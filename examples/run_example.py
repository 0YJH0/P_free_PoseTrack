"""
Demo deform.
Deform template mesh based on input silhouettes and camera pose
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio
import argparse
import shutil

import subprocess
import threading

import sys
current_dir = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_dir)
pp_dir = os.path.dirname(parent_dir)
sys.path.append(pp_dir)

import soft_renderer.functional as srf
import soft_renderer as sr
import utils as ut
import communication
import sam_utils as su
import cv2

from confidence import GradientConsistency, MultiSignalConfidence, apply_mask_confidence_gate
from logging_utils import ConfidenceCsvLogger, ConfidenceRow
from optimization import (
    BoundedLogVariance,
    JointOptimizationConfig,
    PoseDeltaParameterization,
    boundary_alignment_loss,
    evaluate_tracking_risk,
    pose_prior_loss,
    risk_accepts_update,
    temporal_pose_loss,
    weighted_iou_loss,
    RiskCalibratedConfig,
    evaluate_guarded_risk,
    guarded_risk_is_safe,
    heteroscedastic_silhouette_loss,
    select_publication_alpha,
    select_minimum_risk_alpha,
    vertex_gradient_consensus,
)



from segment_anything import sam_model_registry, SamPredictor
from utils import load_config, pjoin


current_dir = os.path.dirname(os.path.realpath(__file__))

data_dir = os.path.join(current_dir, '../data')

import random  
def set_seed(seed):  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    if torch.cuda.is_available():  
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  


def atomic_copy(source, destination):
    """Publish a mesh only after the complete file is present on disk."""
    temporary = destination + ".pending"
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def atomic_save_mesh(mesh, destination):
    temporary = destination + ".pending.obj"
    mesh.save_obj(temporary, save_texture=False)
    os.replace(temporary, destination)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    print("shpe anns", type(anns))

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def show_sam_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3)*255, [1]])
        img[m] = color_mask
    img = img[:,:,:3]  
    img = img.astype(np.uint8)  
    # cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
class Model(nn.Module):
    def __init__(self, template_path):
        super(Model, self).__init__()

        # set template mesh
        self.template_mesh = sr.Mesh.from_obj(template_path, normalization=False)
        self.register_buffer('vertices', self.template_mesh.vertices)
        self.register_buffer('faces', self.template_mesh.faces)
        self.register_buffer('textures', self.template_mesh.textures)

        # optimize for displacement map and center
        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.template_mesh.vertices)))
        self.register_parameter('center', nn.Parameter(torch.zeros(1, 1, 3)))

        # define Laplacian and flatten geometry constraints
        self.laplacian_loss = sr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())
        
        print('vertices shape', self.template_mesh.vertices.shape)	# [1 1032 3]
        print('faces shape', self.template_mesh.faces.shape)	# [1 2600 3]
       
    
    def resetMeshFromObj(self, template_path):
        # set template mesh
        self.template_mesh = sr.Mesh.from_obj(template_path)
        self.register_buffer('vertices', self.template_mesh.vertices * 1.0)
        self.register_buffer('faces', self.template_mesh.faces)
        self.register_buffer('textures', self.template_mesh.textures)

        # optimize for displacement map and center
        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.template_mesh.vertices)))
        self.register_parameter('center', nn.Parameter(torch.zeros(1, 1, 3)))

        # define Laplacian and flatten geometry constraints
        self.laplacian_loss = sr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())
    
    def forward(self, batch_size):
        base = torch.log(self.vertices.abs() / (1 - self.vertices.abs()))
        centroid = torch.tanh(self.center)
        vertices = torch.sigmoid(base + self.displace) * torch.sign(self.vertices)
        vertices = F.relu(vertices) * (1 - centroid) - F.relu(-vertices) * (centroid + 1)
        vertices = vertices + centroid
        
        # apply Laplacian and flatten geometry constraints
        laplacian_loss = self.laplacian_loss(vertices).mean()
        flatten_loss = self.flatten_loss(vertices).mean()

        return sr.Mesh(vertices.repeat(batch_size, 1, 1),
                       self.faces.repeat(batch_size, 1, 1)), laplacian_loss, flatten_loss
     
    def get_face_normals(self, vertices, faces):
        face_normals = []
        for batch_idx in range(faces.shape[0]):
            batch_faces = faces[batch_idx]
            batch_vertices = vertices[batch_idx]
    
            v0 = batch_vertices[batch_faces[:, 0]]
            v1 = batch_vertices[batch_faces[:, 1]]
            v2 = batch_vertices[batch_faces[:, 2]]
    
            normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
            normals = normals / torch.norm(normals, dim=-1, keepdim=True)

            face_normals.append(normals)
        
        face_normals = torch.stack(face_normals)
        return face_normals
    
def neg_iou_loss(predict, target, observation_weights=None):
    dims = tuple(range(predict.ndimension())[1:])  # ndimension = 3
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6

    with open('./loss_iou2.txt', 'a') as f:
         f.write(f"{predict.sum()} '\t' {target.sum()} '\t'  {intersect.sum()} {union.sum()} '\t'  {1. - (intersect / union).sum() / intersect.nelement()}\n")

    per_observation_iou = intersect / union
    if observation_weights is None:
        return 1. - per_observation_iou.mean()
    weights = observation_weights.to(device=predict.device, dtype=predict.dtype)
    if weights.ndim != 1 or weights.numel() != per_observation_iou.numel():
        raise ValueError("observation_weights must contain one value per observation")
    return 1. - (per_observation_iou * weights).sum() / weights.sum().clamp_min(1e-6)


def per_observation_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1.0 - intersect / union

def save_obj_file(vertices, faces, file_path):
    with open(file_path, 'w') as f:
        f.write(f"#coarse_50.obj\n# \n \n")
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()


    
def main():

    set_seed(42) 

	# RBOT dataset
    # summer_config_file = "./config/rbot/a_regular_ape.yml"
	
    # MOPED dataset
    summer_config_file = pjoin("./config/", sys.argv[1], sys.argv[2]) # 0-4
    # summer_config_file = "./config/moped/black_drill/evaluation/00.yml" # 0-4
    # summer_config_file = "./config/moped/duplo_dude/evaluation/00.yml" # 0-6
    # summer_config_file = "./config/moped/rinse_aid/evaluation/00.yml" # 0-5
    # summer_config_file = "./config/moped/toy_plane/evaluation/00.yml" # 0-4
    # summer_config_file = "./config/moped/vim_mug/evaluation/00.yml" # 0-4

    configs = su.LoadConfigSafety(config_file = summer_config_file)
    print("output_dir: ", configs['output_dir'])
    os.makedirs(configs['output_dir'], exist_ok=True)
    confidence_config = configs.get('confidence', {})
    use_confidence_gate = bool(
        confidence_config.get(
            'use_confidence_gate', configs.get('use_confidence_gate', False)
        )
    )
    confidence_threshold = float(
        confidence_config.get(
            'confidence_threshold', configs.get('confidence_threshold', 0.5)
        )
    )
    mask_binary_threshold = float(confidence_config.get('mask_binary_threshold', 0.5))
    confidence_mode = confidence_config.get(
        'mode', 'legacy_hard' if use_confidence_gate else 'disabled'
    )
    if confidence_mode not in {
        'disabled', 'legacy_hard', 'multisignal_hard', 'multisignal_soft',
        'gradient_soft', 'gradient_rollback', 'joint_risk', 'risk_calibrated_soft',
        'risk_calibrated_shadow', 'risk_calibrated_crossfit',
        'risk_calibrated_observable'
    }:
        raise ValueError(f"Unsupported confidence mode: {confidence_mode}")
    fusion_weights = tuple(float(value) for value in confidence_config.get(
        'fusion_weights', [0.40, 0.20, 0.20, 0.20]
    ))
    multisignal_confidence = MultiSignalConfidence(
        base_threshold=float(confidence_config.get('base_threshold', 0.65)),
        adaptive=bool(confidence_config.get('adaptive_threshold', True)),
        adaptive_k=float(confidence_config.get('adaptive_k', 1.5)),
        threshold_min=float(confidence_config.get('threshold_min', 0.45)),
        threshold_max=float(confidence_config.get('threshold_max', 0.85)),
        temperature=float(confidence_config.get('soft_temperature', 0.08)),
        min_weight=float(confidence_config.get('min_weight', 0.05)),
        history_size=int(confidence_config.get('history_size', 32)),
        weights=fusion_weights,
        pose_translation_scale=float(confidence_config.get('pose_translation_scale', 0.03)),
        pose_rotation_scale_deg=float(confidence_config.get('pose_rotation_scale_deg', 10.0)),
        temporal_scale=float(confidence_config.get('temporal_scale', 0.15)),
        min_visible_fraction=float(confidence_config.get('min_visible_fraction', 0.02)),
    )
    gradient_confidence = GradientConsistency(
        alpha=float(confidence_config.get('gradient_alpha', 0.35)),
        history_momentum=float(confidence_config.get('gradient_history_momentum', 0.8)),
        min_weight=float(confidence_config.get('gradient_min_weight', 0.05)),
    )
    rollback_tolerance = float(confidence_config.get('rollback_tolerance', 0.002))
    max_vertex_step = float(confidence_config.get('max_vertex_step_ratio', 0.15))
    joint_config = JointOptimizationConfig(
        outer_steps=int(confidence_config.get('joint_outer_steps', 3)),
        pose_steps=int(confidence_config.get('joint_pose_steps', 20)),
        geometry_steps=int(confidence_config.get('joint_geometry_steps', 60)),
        pose_lr=float(confidence_config.get('joint_pose_lr', 0.002)),
        geometry_lr=float(confidence_config.get('joint_geometry_lr', 0.01)),
        max_rotation_deg=float(confidence_config.get('joint_max_rotation_deg', 5.0)),
        max_translation_ratio=float(confidence_config.get('joint_max_translation_ratio', 0.03)),
        pose_prior_weight=float(confidence_config.get('joint_pose_prior_weight', 0.05)),
        temporal_weight=float(confidence_config.get('joint_temporal_weight', 0.02)),
        boundary_weight=float(confidence_config.get('joint_boundary_weight', 0.10)),
        laplacian_weight=float(confidence_config.get('joint_laplacian_weight', 0.10)),
        geometry_prior_weight=float(confidence_config.get('joint_geometry_prior_weight', 0.02)),
        risk_pose_weight=float(confidence_config.get('joint_risk_pose_weight', 0.10)),
        risk_temporal_weight=float(confidence_config.get('joint_risk_temporal_weight', 0.05)),
        risk_boundary_weight=float(confidence_config.get('joint_risk_boundary_weight', 0.25)),
        risk_geometry_weight=float(confidence_config.get('joint_risk_geometry_weight', 0.10)),
        risk_margin=float(confidence_config.get('joint_risk_margin', 0.002)),
        tail_tolerance=float(confidence_config.get('joint_tail_tolerance', 0.01)),
    )
    v6_config = RiskCalibratedConfig(
        log_variance_radius=float(confidence_config.get('v6_log_variance_radius', 2.0)),
        uncertainty_prior_weight=float(confidence_config.get('v6_uncertainty_prior_weight', 0.10)),
        cvar_quantile=float(confidence_config.get('v6_cvar_quantile', 0.90)),
        cvar_weight=float(confidence_config.get('v6_cvar_weight', 0.35)),
        uncertainty_weight=float(confidence_config.get('v6_uncertainty_weight', 0.25)),
        boundary_weight=float(confidence_config.get('v6_risk_boundary_weight', 0.25)),
        pose_weight=float(confidence_config.get('v6_risk_pose_weight', 0.10)),
        temporal_weight=float(confidence_config.get('v6_risk_temporal_weight', 0.05)),
        geometry_weight=float(confidence_config.get('v6_risk_geometry_weight', 0.10)),
        risk_margin=float(confidence_config.get('v6_risk_margin', 0.001)),
        tail_tolerance=float(confidence_config.get('v6_tail_tolerance', 0.01)),
    )
    confidence_log_path = confidence_config.get(
        'log_csv', os.path.join(configs['output_dir'], 'confidence.csv')
    )
    confidence_logger = ConfidenceCsvLogger(confidence_log_path)
    confidence_logger.reset()
    visualize = bool(configs.get('visualize', False))
    initial_geometry_path = configs.get('initial_geometry_path', '')
    initial_geometry_export_path = configs.get('initial_geometry_export_path', '')
    stop_after_initial_geometry = bool(configs.get('stop_after_initial_geometry', False))
    if initial_geometry_path:
        initial_geometry_path = os.path.abspath(initial_geometry_path)
        if not os.path.isfile(initial_geometry_path):
            raise FileNotFoundError(
                f"Configured initial_geometry_path does not exist: {initial_geometry_path}"
            )
        print("Using frozen initial geometry:", initial_geometry_path)
    

    prepare_start_time = time.time()  
    # differentiable renderer
    from_sphere = False	 # Set this to False and consider the original sphere is not a sphere, so that simplify the code logic
    render_size = configs['renderer_img_size']
    img_nums = su.GenerateArray(first_num=configs['reference_imgs_nums'], second_num=configs['estimated_imgs_nums'], len =100) 
    cur_model_deformed_nums = 0
    max_model_deformed_nums = configs['max_model_deformed_nums'] 
    iteration_nums = configs['iteration_nums_in_each_deform']
    
    batch_size     = configs['batch_size']
    transform      = sr.LookAt(viewing_angle=configs['viewing_angle'])
    lighting       = sr.Lighting()
    rasterizer     = sr.SoftRasterizer(image_size=render_size, sigma_val=1e-4, aggr_func_rgb='hard',near=0.1,far=1000)
    model          = Model(configs['template_mesh']).cuda()
    img_h = configs['height']
    img_w = configs['width']

    # SAM
    sam_checkpoint = pjoin('checkpoints', configs['sam']['checkpoint'])
    model_type     = configs['sam']['model']
    device         = configs['sam']['device']
    sam            = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor      = SamPredictor(sam)

    
    save_obj_path = configs['save_obj_path']
    save_index    = configs['save_index']

	# cpp
    lsc = communication.LocalStorageCommunication(root = configs['communication']['root'], max_count = img_nums[0], from_sphere = from_sphere)
    # tracker and part of scale
    # here call the eg_BIT
    process = subprocess.Popen([configs['tracker'], summer_config_file], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    loop_nums = 0
    pending_shadow = None
        
    while(True):
        status = lsc.Scan(sleep_time = 2) # We sleep here to give the scanner time to scan the local folder, so that the C++ and Python can communicate with each other.
        lsc.set_status(status)
        if not status and process.poll() is not None:
            print(f"Tracker process finished with return code {process.returncode}.")
            if process.returncode != 0:
                raise RuntimeError(
                    f"Tracker process failed with return code {process.returncode}."
                )
            return
        
        dataGroups = lsc.parse_data()
        candidate_frame_ids = list(dataGroups.frame_ids_)
        is_online_update = cur_model_deformed_nums > 0
        shadow_validation = (
            confidence_mode in {
                'risk_calibrated_shadow', 'risk_calibrated_crossfit',
                'risk_calibrated_observable'
            }
            and is_online_update
            and pending_shadow is not None
        )
        
        dataGroups.print_info("datgroups scaning")
        lsc.set_base_datagroups(dataGroups=dataGroups)
        if len(dataGroups.rgbs_) == lsc.max_count():
            lsc.add_new_datagroups_2_base(dataGroups=dataGroups)
            dataGroups = lsc.get_dataGroups()
            dataGroups.print_info()

            segmented_masks = su.segment(predictor=predictor, img_width=img_w, img_height=img_h, dataGroups=dataGroups, configs=configs, save_index=save_index)
            lsc.set_max_count(max_count = img_nums[cur_model_deformed_nums+1])
            segmented_masks = ut.FlipMatImgs(segmented_masks, axis = 0)	
            probs = ut.FlipMatImgs(dataGroups.probs_, axis = 0)	
            cnfds = ut.FlipMatImgs(dataGroups.masks_, axis = 0)	
            
            h, w = segmented_masks[0].shape[-2:]
            if w > h:
                segmented_masks = ut.ExtendMatImgs(segmented_masks, top = (w - h))	
                probs = ut.ExtendMatImgs(probs, top = (w - h))	
                cnfds = ut.ExtendMatImgs(cnfds, top = (w - h))	

            probs = ut.ResizeImgs(probs, width = render_size, height = render_size)
            cnfds = ut.ResizeImgs(cnfds, width = render_size, height = render_size)
            segmented_masks = ut.ResizeImgs(segmented_masks, width = render_size, height = render_size)
            if visualize:
                for segmented_mask in segmented_masks:
                    cv2.imshow("segmented_masks", segmented_mask)
                    cv2.waitKey(100)

            probs = su.BinaringImgs(probs, 100, 255)
			# Note in this the probs actully is the mask
            probs_npy = ut.Mats2Npy(segmented_masks.copy()).astype('float32')/255
            cnfds_npy = ut.Mats2Npy(cnfds.copy()).astype('float32')/255

            Ts  = torch.from_numpy(dataGroups.Ts_)
            Ks  = torch.from_numpy(dataGroups.Ks_).to(dtype=torch.float32)
            use_frozen_initial_geometry = bool(initial_geometry_path) and cur_model_deformed_nums == 0
            if use_frozen_initial_geometry:
                model = Model(initial_geometry_path).cuda()
            elif cur_model_deformed_nums > 0:
                model = Model(save_obj_path).cuda()

            transform.set_img_size(max(img_h, img_w))
            transform.set_K_list(Ks)
            transform.set_T(Ts)

            accepted = np.ones(len(probs_npy), dtype=bool)
            observation_weights = torch.ones(len(probs_npy), dtype=torch.float32)
            candidate_ious = [float('nan')] * len(candidate_frame_ids)
            candidate_pose_scores = [float('nan')] * len(candidate_frame_ids)
            candidate_visibility_scores = [float('nan')] * len(candidate_frame_ids)
            candidate_temporal_scores = [float('nan')] * len(candidate_frame_ids)
            candidate_fused_scores = [float('nan')] * len(candidate_frame_ids)
            candidate_thresholds = [float('nan')] * len(candidate_frame_ids)
            candidate_weights = [1.0] * len(candidate_frame_ids)
            candidate_gradient_scores = [float('nan')] * len(candidate_frame_ids)
            update_accepted = True
            pre_reference_iou = float('nan')
            post_reference_iou = float('nan')
            holdout_offset = None
            pre_holdout_iou = float('nan')
            post_holdout_iou = float('nan')
            max_vertex_step_ratio = float('nan')
            pre_tracking_risk = float('nan')
            post_tracking_risk = float('nan')
            post_tail_iou_loss = float('nan')
            mean_pose_correction = float('nan')
            optimized_log_variance = float('nan')
            effective_samples = float('nan')
            pre_cvar = float('nan')
            post_cvar = float('nan')
            publication_alpha = float('nan')
            full_step_tracking_risk = float('nan')
            shadow_action = ''
            prospective_risk_gain = float('nan')
            active_vertex_fraction = float('nan')
            mean_vertex_agreement = float('nan')
            risk_prior_weights = None
            # Compute confidence in both paired arms.  The baseline differs only
            # in that ``enabled=False`` makes every candidate pass.
            if is_online_update:
                with torch.no_grad():
                    confidence_mesh, _, _ = model(len(probs_npy))
                    confidence_mesh = lighting(confidence_mesh)
                    confidence_mesh = transform(confidence_mesh)
                    rendered_masks = rasterizer(confidence_mesh)[:, 3]
                    segmentation_masks = torch.from_numpy(probs_npy[:, 3]).cuda()
                candidate_count = len(candidate_frame_ids)
                candidate_start = len(probs_npy) - candidate_count
                candidate_masks = segmentation_masks[candidate_start:]
                candidate_rendered = rendered_masks[candidate_start:]
                if confidence_mode in {
                    'multisignal_hard', 'multisignal_soft', 'gradient_soft',
                    'gradient_rollback', 'joint_risk', 'risk_calibrated_soft',
                    'risk_calibrated_shadow', 'risk_calibrated_crossfit',
                    'risk_calibrated_observable'
                }:
                    result = multisignal_confidence.evaluate(
                        candidate_masks,
                        candidate_rendered,
                        Ts[candidate_start:].to(dtype=torch.float32),
                        candidate_frame_ids,
                        mask_binary_threshold=mask_binary_threshold,
                    )
                    if confidence_mode == 'multisignal_hard':
                        candidate_acceptance = (result.confidence >= result.threshold).cpu().numpy()
                        candidate_loss_weights = candidate_acceptance.astype(np.float32)
                    else:
                        candidate_acceptance = np.ones(candidate_count, dtype=bool)
                        candidate_loss_weights = result.weight.clamp_min(
                            multisignal_confidence.min_weight
                        ).cpu().numpy()
                    accepted[candidate_start:] = candidate_acceptance
                    observation_weights[candidate_start:] = torch.from_numpy(candidate_loss_weights)
                    candidate_ious = result.mask_iou.cpu().tolist()
                    candidate_pose_scores = result.pose_score.cpu().tolist()
                    candidate_visibility_scores = result.visibility_score.cpu().tolist()
                    candidate_temporal_scores = result.temporal_score.cpu().tolist()
                    candidate_fused_scores = result.confidence.cpu().tolist()
                    candidate_thresholds = result.threshold.cpu().tolist()
                    candidate_weights = candidate_loss_weights.tolist()
                else:
                    gate_result = apply_mask_confidence_gate(
                        candidate_masks,
                        candidate_rendered,
                        enabled=confidence_mode == 'legacy_hard',
                        confidence_threshold=confidence_threshold,
                        mask_binary_threshold=mask_binary_threshold,
                    )
                    accepted[candidate_start:] = gate_result.accepted.cpu().numpy()
                    candidate_ious = gate_result.mask_iou.cpu().tolist()

            probs_npy = probs_npy[accepted]
            Ts = Ts[accepted]
            Ks = Ks[accepted]
            observation_weights = observation_weights[accepted]
            if len(probs_npy) == 0:
                raise RuntimeError(
                    "Confidence gating removed every observation, including references; "
                    "cannot run geometry optimization."
                )

            transform.set_K_list(Ks)
            transform.set_T(Ts)
            batch_size = len(probs_npy)    
            cur_model_deformed_nums += 1
            previous_model_path = save_obj_path
            save_obj_path = os.path.join(configs['output_dir'], '_realtime_' + str(cur_model_deformed_nums)+'.obj')
            print("save_obj_path****", save_obj_path)
            if use_frozen_initial_geometry:
                # Byte-for-byte reuse guarantees identical initialization in
                # every paired arm and avoids stochastic first-stage fitting.
                atomic_copy(initial_geometry_path, save_obj_path)
                print("Published frozen initial geometry without optimization.")
            else:
                if is_online_update and confidence_mode in {
                    'gradient_soft', 'gradient_rollback', 'joint_risk',
                    'risk_calibrated_soft', 'risk_calibrated_shadow',
                    'risk_calibrated_crossfit', 'risk_calibrated_observable'
                } and not shadow_validation:
                    reference_count = batch_size - candidate_count
                    images_for_gradient = torch.from_numpy(probs_npy).cuda()
                    gradient_mesh, _, _ = model(batch_size)
                    gradient_mesh = lighting(gradient_mesh)
                    gradient_mesh = transform(gradient_mesh)
                    gradient_predictions = rasterizer(gradient_mesh)[:, 3]
                    gradient_losses = per_observation_iou_loss(
                        gradient_predictions, images_for_gradient[:, 3]
                    )
                    gradient_result = gradient_confidence.evaluate(
                        gradient_losses,
                        model.parameters(),
                        reference_count=reference_count,
                        prior_candidate_weights=observation_weights[reference_count:],
                    )
                    observation_weights[reference_count:] = gradient_result.fused_weights.cpu()
                    candidate_gradient_scores = gradient_result.scores.cpu().tolist()
                    candidate_weights = gradient_result.fused_weights.cpu().tolist()
                    del gradient_mesh, gradient_predictions, gradient_losses, images_for_gradient

                if is_online_update and confidence_mode in {
                    'gradient_rollback', 'joint_risk', 'risk_calibrated_soft',
                    'risk_calibrated_shadow', 'risk_calibrated_crossfit',
                    'risk_calibrated_observable'
                } and not shadow_validation:
                    reference_count = batch_size - candidate_count
                    holdout_offset = int(np.argmin(candidate_fused_scores))
                    holdout_index = reference_count + holdout_offset
                    risk_prior_weights = observation_weights.clone()
                    if confidence_mode in {
                        'risk_calibrated_crossfit', 'risk_calibrated_observable'
                    }:
                        # The later half is never used to fit the proposal.  It
                        # provides an internal forward-in-time generalization test.
                        crossfit_offset = max(1, candidate_count // 2)
                        crossfit_holdout_indices = list(range(
                            reference_count + crossfit_offset, batch_size
                        ))
                        if not crossfit_holdout_indices:
                            crossfit_holdout_indices = [holdout_index]
                        observation_weights[crossfit_holdout_indices] = 0.0
                        for index in crossfit_holdout_indices:
                            candidate_weights[index - reference_count] = 0.0
                    else:
                        crossfit_holdout_indices = [holdout_index]
                        observation_weights[holdout_index] = 0.0
                        candidate_weights[holdout_offset] = 0.0
                    with torch.no_grad():
                        pre_vertices = model(1)[0].vertices.detach()[0].clone()
                    transform.set_K_list(Ks[:reference_count])
                    transform.set_T(Ts[:reference_count])
                    with torch.no_grad():
                        validation_targets = torch.from_numpy(probs_npy[:reference_count]).cuda()
                        validation_mesh, _, _ = model(reference_count)
                        validation_mesh = lighting(validation_mesh)
                        validation_mesh = transform(validation_mesh)
                        validation_predictions = rasterizer(validation_mesh)[:, 3]
                        pre_reference_iou = float(
                            (1.0 - per_observation_iou_loss(
                                validation_predictions, validation_targets[:, 3]
                            )).mean()
                        )
                    del validation_mesh, validation_predictions, validation_targets
                    transform.set_K_list(Ks[holdout_index:holdout_index + 1])
                    transform.set_T(Ts[holdout_index:holdout_index + 1])
                    with torch.no_grad():
                        holdout_target = torch.from_numpy(
                            probs_npy[holdout_index:holdout_index + 1]
                        ).cuda()
                        holdout_mesh, _, _ = model(1)
                        holdout_mesh = lighting(holdout_mesh)
                        holdout_mesh = transform(holdout_mesh)
                        holdout_prediction = rasterizer(holdout_mesh)[:, 3]
                        pre_holdout_iou = float(
                            1.0 - per_observation_iou_loss(
                                holdout_prediction, holdout_target[:, 3]
                            ).mean()
                        )
                    del holdout_mesh, holdout_prediction, holdout_target
                    transform.set_K_list(Ks)
                    transform.set_T(Ts)

                if is_online_update and confidence_mode in {
                    'joint_risk', 'risk_calibrated_soft', 'risk_calibrated_shadow',
                    'risk_calibrated_crossfit', 'risk_calibrated_observable'
                } and not shadow_validation:
                    images_gt = torch.from_numpy(probs_npy).cuda()
                    observation_weights_gpu = observation_weights.cuda()
                    object_scale = torch.linalg.vector_norm(
                        pre_vertices.amax(0) - pre_vertices.amin(0)
                    ).clamp_min(1e-12)
                    optimizable_poses = torch.ones(batch_size, dtype=torch.bool, device='cuda')
                    optimizable_poses[0] = False  # anchor the gauge freedom
                    optimizable_poses[crossfit_holdout_indices] = False
                    if confidence_mode == 'risk_calibrated_observable':
                        fit_indices = torch.nonzero(
                            observation_weights > 0, as_tuple=False
                        ).flatten()
                        fit_weights = observation_weights[fit_indices]
                        selected_count = min(8, len(fit_indices))
                        selected_indices = fit_indices[
                            torch.topk(fit_weights, selected_count).indices
                        ]
                        transform.set_T(Ts.cuda())
                        probe_mesh, _, _ = model(batch_size)
                        probe_prediction = rasterizer(
                            transform(lighting(probe_mesh))
                        )[:, 3]
                        probe_losses = per_observation_iou_loss(
                            probe_prediction, images_gt[:, 3]
                        )
                        view_gradients = []
                        for view_index in selected_indices.tolist():
                            gradient = torch.autograd.grad(
                                probe_losses[view_index], model.displace,
                                retain_graph=True, create_graph=False,
                            )[0]
                            view_gradients.append(gradient.detach()[0])
                        vertex_mask, agreements, supports = vertex_gradient_consensus(
                            torch.stack(view_gradients),
                            agreement_threshold=0.60,
                            keep_fraction=0.30,
                        )
                        active_vertex_fraction = float(vertex_mask.float().mean())
                        mean_vertex_agreement = float(
                            agreements[vertex_mask].mean()
                            if vertex_mask.any() else 0.0
                        )
                        displacement_mask = vertex_mask.view(1, -1, 1).to(
                            model.displace
                        )
                        model.displace.register_hook(
                            lambda gradient: gradient * displacement_mask
                        )
                        model.center.register_hook(torch.zeros_like)
                        print(
                            "V8 observable vertices: "
                            f"active={active_vertex_fraction:.4f} "
                            f"agreement={mean_vertex_agreement:.4f} "
                            f"views={selected_count}"
                        )
                        del probe_mesh, probe_prediction, probe_losses, view_gradients
                    pose_deltas = PoseDeltaParameterization(
                        batch_size,
                        object_scale,
                        max_rotation_deg=joint_config.max_rotation_deg,
                        max_translation_ratio=joint_config.max_translation_ratio,
                        optimizable=optimizable_poses,
                        device='cuda',
                    )
                    pose_optimizer = torch.optim.Adam(
                        pose_deltas.parameters(), joint_config.pose_lr
                    )
                    uncertainty_model = None
                    uncertainty_optimizer = None
                    if confidence_mode in {
                        'risk_calibrated_soft', 'risk_calibrated_shadow',
                        'risk_calibrated_crossfit', 'risk_calibrated_observable'
                    }:
                        uncertainty_model = BoundedLogVariance(
                            risk_prior_weights.cuda(),
                            radius=v6_config.log_variance_radius,
                            optimizable=optimizable_poses,
                        ).cuda()
                        uncertainty_optimizer = torch.optim.Adam(
                            uncertainty_model.parameters(), joint_config.pose_lr
                        )
                    geometry_optimizer = torch.optim.Adam(
                        model.parameters(), joint_config.geometry_lr, betas=(0.5, 0.99)
                    )
                    print("begin joint pose-geometry optimization")
                    for outer_step in range(joint_config.outer_steps):
                        for parameter in model.parameters():
                            parameter.requires_grad_(False)
                        pose_loop = tqdm.tqdm(
                            range(joint_config.pose_steps),
                            desc=f"joint pose {outer_step + 1}/{joint_config.outer_steps}",
                        )
                        for _ in pose_loop:
                            corrected_poses = pose_deltas(Ts.cuda())
                            transform.set_T(corrected_poses)
                            mesh, _, _ = model(batch_size)
                            prediction = rasterizer(transform(lighting(mesh)))[:, 3]
                            twists = pose_deltas.twists()
                            if uncertainty_model is not None:
                                fit_mask = observation_weights_gpu > 0
                                loss = (
                                    heteroscedastic_silhouette_loss(
                                        prediction[fit_mask], images_gt[fit_mask, 3],
                                        uncertainty_model()[fit_mask],
                                        boundary_weight=joint_config.boundary_weight,
                                    )
                                    + v6_config.uncertainty_prior_weight
                                    * uncertainty_model.prior_penalty()
                                    + joint_config.pose_prior_weight
                                    * pose_prior_loss(twists, uncertainty_model.precision())
                                    + joint_config.temporal_weight
                                    * temporal_pose_loss(corrected_poses)
                                )
                            else:
                                loss = (
                                    weighted_iou_loss(
                                        prediction, images_gt[:, 3], observation_weights_gpu
                                    )
                                    + joint_config.boundary_weight * boundary_alignment_loss(
                                        prediction, images_gt[:, 3], observation_weights_gpu
                                    )
                                    + joint_config.pose_prior_weight * pose_prior_loss(
                                        twists, observation_weights_gpu
                                    )
                                    + joint_config.temporal_weight * temporal_pose_loss(corrected_poses)
                                )
                            pose_optimizer.zero_grad()
                            if uncertainty_optimizer is not None:
                                uncertainty_optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(pose_deltas.parameters(), 1.0)
                            pose_optimizer.step()
                            if uncertainty_optimizer is not None:
                                torch.nn.utils.clip_grad_norm_(
                                    uncertainty_model.parameters(), 1.0
                                )
                                uncertainty_optimizer.step()
                            pose_loop.set_postfix(loss=f"{loss.item():.4f}")

                        for parameter in model.parameters():
                            parameter.requires_grad_(True)
                        corrected_poses = pose_deltas(Ts.cuda()).detach()
                        transform.set_T(corrected_poses)
                        geometry_loop = tqdm.tqdm(
                            range(joint_config.geometry_steps),
                            desc=f"joint geometry {outer_step + 1}/{joint_config.outer_steps}",
                        )
                        for _ in geometry_loop:
                            mesh, laplacian_loss, _ = model(batch_size)
                            prediction = rasterizer(transform(lighting(mesh)))[:, 3]
                            current_vertices = mesh.vertices[0]
                            geometry_prior = (
                                (current_vertices - pre_vertices).square().sum(1).mean()
                                / object_scale.square()
                            )
                            geometry_weights = (
                                uncertainty_model.precision().detach()
                                if uncertainty_model is not None else observation_weights_gpu
                            )
                            if confidence_mode in {
                                'risk_calibrated_crossfit',
                                'risk_calibrated_observable'
                            }:
                                geometry_weights = geometry_weights * (
                                    observation_weights_gpu > 0
                                )
                            loss = (
                                weighted_iou_loss(
                                    prediction, images_gt[:, 3], geometry_weights
                                )
                                + joint_config.boundary_weight * boundary_alignment_loss(
                                    prediction, images_gt[:, 3], geometry_weights
                                )
                                + joint_config.laplacian_weight * laplacian_loss
                                + joint_config.geometry_prior_weight * geometry_prior
                            )
                            geometry_optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            geometry_optimizer.step()
                            geometry_loop.set_postfix(loss=f"{loss.item():.4f}")
                            loop_nums += 1
                    corrected_Ts = pose_deltas(Ts.cuda()).detach()
                    final_twists = pose_deltas.twists().detach()
                    if uncertainty_model is not None:
                        optimized_log_variance = float(uncertainty_model().mean().detach())
                    print(
                        "Joint pose correction: "
                        f"mean={torch.linalg.vector_norm(final_twists, dim=1).mean().item():.8f} "
                        f"max={torch.linalg.vector_norm(final_twists, dim=1).max().item():.8f}"
                    )
                    transform.set_T(corrected_Ts)
                elif not shadow_validation:
                    optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))
                    loop = tqdm.tqdm(list(range(0, iteration_nums)))
                    print("begin to deform model")
                    for i in loop:
                        images_gt = torch.from_numpy(probs_npy).cuda()
                        mesh, laplacian_loss, flatten_loss = model(batch_size)

                        # render
                        mesh = lighting(mesh)
                        mesh = transform(mesh)
                        images_pred = rasterizer(mesh)
                        loss = neg_iou_loss(
                            images_pred[:, 3], images_gt[:, 3], observation_weights
                        ) + 0.1 * laplacian_loss

                        loop.set_description('Loss: %.4f' % (loss.item()))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        if visualize:
                            intermediate_result = images_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))
                            intermediate_result_flipped = cv2.flip(intermediate_result, 0)
                            cv2.imshow("intermediate_result", intermediate_result_flipped)
                            cv2.waitKey(1)
                        loop_nums = loop_nums + 1

                if is_online_update and confidence_mode == 'gradient_rollback':
                    transform.set_K_list(Ks[:reference_count])
                    transform.set_T(Ts[:reference_count])
                    with torch.no_grad():
                        validation_targets = torch.from_numpy(probs_npy[:reference_count]).cuda()
                        validation_mesh, _, _ = model(reference_count)
                        validation_mesh = lighting(validation_mesh)
                        validation_mesh = transform(validation_mesh)
                        validation_predictions = rasterizer(validation_mesh)[:, 3]
                        post_reference_iou = float(
                            (1.0 - per_observation_iou_loss(
                                validation_predictions, validation_targets[:, 3]
                            )).mean()
                        )
                    update_accepted = post_reference_iou >= pre_reference_iou - rollback_tolerance
                    del validation_mesh, validation_predictions, validation_targets
                    transform.set_K_list(Ks[holdout_index:holdout_index + 1])
                    transform.set_T(Ts[holdout_index:holdout_index + 1])
                    with torch.no_grad():
                        holdout_target = torch.from_numpy(
                            probs_npy[holdout_index:holdout_index + 1]
                        ).cuda()
                        holdout_mesh, _, _ = model(1)
                        holdout_mesh = lighting(holdout_mesh)
                        holdout_mesh = transform(holdout_mesh)
                        holdout_prediction = rasterizer(holdout_mesh)[:, 3]
                        post_holdout_iou = float(
                            1.0 - per_observation_iou_loss(
                                holdout_prediction, holdout_target[:, 3]
                            ).mean()
                        )
                    del holdout_mesh, holdout_prediction, holdout_target
                    transform.set_K_list(Ks)
                    transform.set_T(Ts)
                    update_accepted = update_accepted and (
                        post_holdout_iou >= pre_holdout_iou - rollback_tolerance
                    )
                    with torch.no_grad():
                        post_vertices = model(1)[0].vertices.detach()[0]
                        object_scale = torch.linalg.vector_norm(
                            pre_vertices.amax(0) - pre_vertices.amin(0)
                        ).clamp_min(1e-12)
                        max_vertex_step_ratio = float(
                            torch.linalg.vector_norm(post_vertices - pre_vertices, dim=1).max()
                            / object_scale
                        )
                    update_accepted = update_accepted and max_vertex_step_ratio <= max_vertex_step
                    print(
                        f"Geometry trust check: ref={pre_reference_iou:.6f}->{post_reference_iou:.6f} "
                        f"holdout={pre_holdout_iou:.6f}->{post_holdout_iou:.6f} "
                        f"vertex_step={max_vertex_step_ratio:.6f} "
                        f"accepted={update_accepted}"
                    )
                    if update_accepted:
                        atomic_save_mesh(model(1)[0], save_obj_path)
                    else:
                        atomic_copy(previous_model_path, save_obj_path)
                elif is_online_update and confidence_mode == 'joint_risk':
                    # Score the full causal window, ordering the untouched
                    # holdout last so aggregate fit, required pose correction,
                    # and a genuinely unseen tail check are all available.
                    validation_indices = [
                        index for index in range(batch_size) if index != holdout_index
                    ] + [holdout_index]
                    validation_indices_tensor = torch.as_tensor(
                        validation_indices, dtype=torch.long
                    )
                    validation_targets = torch.from_numpy(
                        probs_npy[validation_indices]
                    ).cuda()[:, 3]
                    with torch.no_grad():
                        post_vertices = model(1)[0].vertices.detach()[0]
                        max_vertex_step_ratio = float(
                            torch.linalg.vector_norm(
                                post_vertices - pre_vertices, dim=1
                            ).max() / object_scale
                        )

                        transform.set_K_list(Ks[validation_indices_tensor])
                        transform.set_T(corrected_Ts[validation_indices_tensor.cuda()])
                        new_mesh, _, _ = model(len(validation_indices))
                        new_predictions = rasterizer(
                            transform(lighting(new_mesh))
                        )[:, 3]
                        new_risk = evaluate_tracking_risk(
                            new_predictions,
                            validation_targets,
                            pose_deltas.twists().detach()[validation_indices_tensor.cuda()],
                            corrected_Ts[validation_indices_tensor.cuda()],
                            max_vertex_step_ratio,
                            joint_config,
                        )

                        old_model = Model(previous_model_path).cuda()
                        transform.set_T(Ts[validation_indices_tensor])
                        old_mesh, _, _ = old_model(len(validation_indices))
                        old_predictions = rasterizer(
                            transform(lighting(old_mesh))
                        )[:, 3]
                        old_risk = evaluate_tracking_risk(
                            old_predictions,
                            validation_targets,
                            torch.zeros(
                                len(validation_indices), 6,
                                device=old_predictions.device,
                            ),
                            Ts[validation_indices_tensor].to(old_predictions.device),
                            0.0,
                            joint_config,
                        )
                        pre_reference_iou = float(
                            1.0 - per_observation_iou_loss(
                                old_predictions[:reference_count],
                                validation_targets[:reference_count]
                            ).mean()
                        )
                        post_reference_iou = float(
                            1.0 - per_observation_iou_loss(
                                new_predictions[:reference_count],
                                validation_targets[:reference_count]
                            ).mean()
                        )
                        pre_holdout_iou = float(
                            1.0 - per_observation_iou_loss(
                                old_predictions[-1:], validation_targets[-1:]
                            ).mean()
                        )
                        post_holdout_iou = float(
                            1.0 - per_observation_iou_loss(
                                new_predictions[-1:], validation_targets[-1:]
                            ).mean()
                        )
                    update_accepted = (
                        max_vertex_step_ratio <= max_vertex_step
                        and risk_accepts_update(old_risk, new_risk, joint_config)
                    )
                    pre_tracking_risk = old_risk.total
                    post_tracking_risk = new_risk.total
                    post_tail_iou_loss = new_risk.tail_iou_loss
                    mean_pose_correction = float(
                        torch.linalg.vector_norm(final_twists, dim=1).mean()
                    )
                    print(
                        f"Joint tracking-risk check: old={old_risk.total:.6f} "
                        f"new={new_risk.total:.6f} "
                        f"tail={old_risk.tail_iou_loss:.6f}->{new_risk.tail_iou_loss:.6f} "
                        f"vertex_step={max_vertex_step_ratio:.6f} "
                        f"accepted={update_accepted}"
                    )
                    transform.set_K_list(Ks)
                    transform.set_T(Ts)
                    del old_model, old_mesh, old_predictions, new_mesh, new_predictions
                    if update_accepted:
                        atomic_save_mesh(model(1)[0], save_obj_path)
                    else:
                        atomic_copy(previous_model_path, save_obj_path)
                elif is_online_update and confidence_mode == 'risk_calibrated_soft':
                    # Compare old and candidate geometry under the same frozen
                    # candidate pose corrections. The common pose penalty cannot
                    # favor either mesh, and past tracker outputs are never rewritten.
                    validation_indices = [
                        index for index in range(batch_size) if index != holdout_index
                    ] + [holdout_index]
                    validation_indices_tensor = torch.as_tensor(
                        validation_indices, dtype=torch.long
                    )
                    validation_targets = torch.from_numpy(
                        probs_npy[validation_indices]
                    ).cuda()[:, 3]
                    validation_poses = corrected_Ts[
                        validation_indices_tensor.cuda()
                    ]
                    validation_weights = risk_prior_weights[
                        validation_indices_tensor
                    ].cuda()
                    validation_twists = final_twists[
                        validation_indices_tensor.cuda()
                    ]
                    with torch.no_grad():
                        candidate_vertices = model(1)[0].vertices.detach()[0]
                        max_vertex_step_ratio = float(
                            torch.linalg.vector_norm(
                                candidate_vertices - pre_vertices, dim=1
                            ).max() / object_scale
                        )
                        transform.set_K_list(Ks[validation_indices_tensor])
                        transform.set_T(validation_poses)

                        old_model = Model(previous_model_path).cuda()
                        old_mesh, _, _ = old_model(len(validation_indices))
                        old_predictions = rasterizer(
                            transform(lighting(old_mesh))
                        )[:, 3]
                        old_risk = evaluate_guarded_risk(
                            old_predictions, validation_targets, validation_weights,
                            validation_twists, validation_poses, 0.0, v6_config,
                        )

                        alpha_risks = {}
                        alpha_predictions = {}
                        for alpha in v6_config.alpha_grid:
                            if alpha <= 0.0 or alpha * max_vertex_step_ratio > max_vertex_step:
                                continue
                            blended_vertices = pre_vertices + float(alpha) * (
                                candidate_vertices - pre_vertices
                            )
                            blended_mesh = sr.Mesh(
                                blended_vertices.unsqueeze(0).repeat(
                                    len(validation_indices), 1, 1
                                ),
                                model.faces.repeat(len(validation_indices), 1, 1),
                            )
                            blended_predictions = rasterizer(
                                transform(lighting(blended_mesh))
                            )[:, 3]
                            alpha_risks[float(alpha)] = evaluate_guarded_risk(
                                blended_predictions, validation_targets,
                                validation_weights, validation_twists, validation_poses,
                                float(alpha) * max_vertex_step_ratio, v6_config,
                            )
                            alpha_predictions[float(alpha)] = blended_predictions

                        publication_alpha, selected_risk = select_publication_alpha(
                            old_risk, alpha_risks, v6_config
                        )
                        selected_predictions = (
                            old_predictions if publication_alpha == 0.0
                            else alpha_predictions[publication_alpha]
                        )
                        full_step_tracking_risk = (
                            alpha_risks[1.0].total if 1.0 in alpha_risks
                            else float('nan')
                        )
                        pre_reference_iou = float(
                            1.0 - per_observation_iou_loss(
                                old_predictions[:reference_count],
                                validation_targets[:reference_count],
                            ).mean()
                        )
                        post_reference_iou = float(
                            1.0 - per_observation_iou_loss(
                                selected_predictions[:reference_count],
                                validation_targets[:reference_count],
                            ).mean()
                        )
                        pre_holdout_iou = float(
                            1.0 - per_observation_iou_loss(
                                old_predictions[-1:], validation_targets[-1:]
                            ).mean()
                        )
                        post_holdout_iou = float(
                            1.0 - per_observation_iou_loss(
                                selected_predictions[-1:], validation_targets[-1:]
                            ).mean()
                        )

                    update_accepted = publication_alpha > 0.0
                    pre_tracking_risk = old_risk.total
                    post_tracking_risk = selected_risk.total
                    post_tail_iou_loss = selected_risk.cvar
                    pre_cvar = old_risk.cvar
                    post_cvar = selected_risk.cvar
                    effective_samples = selected_risk.effective_samples
                    mean_pose_correction = float(
                        torch.linalg.vector_norm(final_twists, dim=1).mean()
                    )
                    print(
                        f"V6 guarded soft publication: old={old_risk.total:.6f} "
                        f"selected={selected_risk.total:.6f} "
                        f"cvar={old_risk.cvar:.6f}->{selected_risk.cvar:.6f} "
                        f"proposal_step={max_vertex_step_ratio:.6f} "
                        f"alpha={publication_alpha:.3f}"
                    )
                    transform.set_K_list(Ks)
                    transform.set_T(Ts)
                    if update_accepted:
                        published_vertices = pre_vertices + publication_alpha * (
                            candidate_vertices - pre_vertices
                        )
                        published_mesh = sr.Mesh(
                            published_vertices.unsqueeze(0), model.faces[:1]
                        )
                        atomic_save_mesh(published_mesh, save_obj_path)
                    else:
                        atomic_copy(previous_model_path, save_obj_path)
                    del old_model, old_mesh, old_predictions, validation_targets
                elif shadow_validation:
                    # The proposal was produced by the preceding update.  This
                    # batch is therefore genuinely prospective and was tracked
                    # exclusively with the unchanged live geometry.
                    validation_targets = torch.from_numpy(probs_npy).cuda()[:, 3]
                    validation_weights = observation_weights.cuda()
                    validation_poses = Ts.cuda()
                    zero_twists = torch.zeros(batch_size, 6, device='cuda')
                    base_vertices = pending_shadow['base_vertices'].cuda()
                    candidate_vertices = pending_shadow['candidate_vertices'].cuda()
                    faces = pending_shadow['faces'].cuda()
                    object_scale = pending_shadow['object_scale'].cuda()
                    max_vertex_step_ratio = float(
                        torch.linalg.vector_norm(
                            candidate_vertices - base_vertices, dim=1
                        ).max() / object_scale
                    )
                    transform.set_K_list(Ks)
                    transform.set_T(validation_poses)
                    with torch.no_grad():
                        base_mesh = sr.Mesh(
                            base_vertices.unsqueeze(0).repeat(batch_size, 1, 1),
                            faces.unsqueeze(0).repeat(batch_size, 1, 1),
                        )
                        old_predictions = rasterizer(
                            transform(lighting(base_mesh))
                        )[:, 3]
                        old_risk = evaluate_guarded_risk(
                            old_predictions, validation_targets, validation_weights,
                            zero_twists, validation_poses, 0.0, v6_config,
                        )
                        alpha_risks = {}
                        alpha_predictions = {}
                        eligible_alphas = pending_shadow.get(
                            'eligible_alphas', v6_config.alpha_grid
                        )
                        for alpha in eligible_alphas:
                            if alpha <= 0.0 or alpha * max_vertex_step_ratio > max_vertex_step:
                                continue
                            vertices = base_vertices + float(alpha) * (
                                candidate_vertices - base_vertices
                            )
                            mesh = sr.Mesh(
                                vertices.unsqueeze(0).repeat(batch_size, 1, 1),
                                faces.unsqueeze(0).repeat(batch_size, 1, 1),
                            )
                            predictions = rasterizer(
                                transform(lighting(mesh))
                            )[:, 3]
                            alpha_risks[float(alpha)] = evaluate_guarded_risk(
                                predictions, validation_targets, validation_weights,
                                zero_twists, validation_poses,
                                float(alpha) * max_vertex_step_ratio, v6_config,
                            )
                            alpha_predictions[float(alpha)] = predictions
                        publication_alpha, selected_risk = select_minimum_risk_alpha(
                            old_risk, alpha_risks, v6_config
                        )
                        selected_predictions = (
                            old_predictions if publication_alpha == 0.0
                            else alpha_predictions[publication_alpha]
                        )
                        pre_reference_iou = float(
                            1.0 - per_observation_iou_loss(
                                old_predictions, validation_targets
                            ).mean()
                        )
                        post_reference_iou = float(
                            1.0 - per_observation_iou_loss(
                                selected_predictions, validation_targets
                            ).mean()
                        )
                    update_accepted = publication_alpha > 0.0
                    pre_tracking_risk = old_risk.total
                    post_tracking_risk = selected_risk.total
                    prospective_risk_gain = old_risk.total - selected_risk.total
                    pre_cvar = old_risk.cvar
                    post_cvar = selected_risk.cvar
                    post_tail_iou_loss = selected_risk.cvar
                    effective_samples = selected_risk.effective_samples
                    full_step_tracking_risk = (
                        alpha_risks[1.0].total if 1.0 in alpha_risks
                        else float('nan')
                    )
                    shadow_action = 'committed' if update_accepted else 'rejected'
                    if update_accepted:
                        vertices = base_vertices + publication_alpha * (
                            candidate_vertices - base_vertices
                        )
                        atomic_save_mesh(
                            sr.Mesh(vertices.unsqueeze(0), faces.unsqueeze(0)),
                            save_obj_path,
                        )
                    else:
                        atomic_copy(previous_model_path, save_obj_path)
                    print(
                        f"V6.1 prospective shadow: old={old_risk.total:.6f} "
                        f"selected={selected_risk.total:.6f} "
                        f"gain={prospective_risk_gain:.6f} "
                        f"alpha={publication_alpha:.3f} action={shadow_action}"
                    )
                    pending_shadow = None
                    del base_mesh, old_predictions, validation_targets
                elif is_online_update and confidence_mode in {
                    'risk_calibrated_shadow', 'risk_calibrated_crossfit',
                    'risk_calibrated_observable'
                }:
                    # Keep the optimized candidate off the live tracking path.
                    # It can only be published after the next update batch.
                    with torch.no_grad():
                        candidate_vertices = model(1)[0].vertices.detach()[0]
                        max_vertex_step_ratio = float(
                            torch.linalg.vector_norm(
                                candidate_vertices - pre_vertices, dim=1
                            ).max() / object_scale
                        )
                    eligible_alphas = tuple(v6_config.alpha_grid)
                    if confidence_mode in {
                        'risk_calibrated_crossfit', 'risk_calibrated_observable'
                    }:
                        holdout_tensor = torch.as_tensor(
                            crossfit_holdout_indices, dtype=torch.long
                        )
                        targets = torch.from_numpy(
                            probs_npy[crossfit_holdout_indices]
                        ).cuda()[:, 3]
                        poses = Ts[holdout_tensor].cuda()
                        weights = risk_prior_weights[holdout_tensor].cuda()
                        zeros = torch.zeros(len(holdout_tensor), 6, device='cuda')
                        transform.set_K_list(Ks[holdout_tensor])
                        transform.set_T(poses)
                        with torch.no_grad():
                            old_mesh = sr.Mesh(
                                pre_vertices.unsqueeze(0).repeat(
                                    len(holdout_tensor), 1, 1
                                ),
                                model.faces[:1].repeat(len(holdout_tensor), 1, 1),
                            )
                            old_pred = rasterizer(transform(lighting(old_mesh)))[:, 3]
                            old_internal = evaluate_guarded_risk(
                                old_pred, targets, weights, zeros, poses, 0.0, v6_config
                            )
                            safe_alphas = []
                            for alpha in v6_config.alpha_grid:
                                if alpha <= 0.0 or alpha * max_vertex_step_ratio > max_vertex_step:
                                    continue
                                vertices = pre_vertices + float(alpha) * (
                                    candidate_vertices - pre_vertices
                                )
                                mesh = sr.Mesh(
                                    vertices.unsqueeze(0).repeat(len(holdout_tensor), 1, 1),
                                    model.faces[:1].repeat(len(holdout_tensor), 1, 1),
                                )
                                pred = rasterizer(transform(lighting(mesh)))[:, 3]
                                candidate_risk = evaluate_guarded_risk(
                                    pred, targets, weights, zeros, poses,
                                    float(alpha) * max_vertex_step_ratio, v6_config,
                                )
                                if guarded_risk_is_safe(
                                    old_internal, candidate_risk, v6_config
                                ):
                                    safe_alphas.append(float(alpha))
                        eligible_alphas = tuple(safe_alphas)
                        transform.set_K_list(Ks)
                        transform.set_T(Ts)
                        print(
                            "V7 internal temporal cross-fit safe alphas: "
                            f"{eligible_alphas}"
                        )
                    pending_shadow = {
                        'base_vertices': pre_vertices.detach().cpu(),
                        'candidate_vertices': candidate_vertices.detach().cpu(),
                        'faces': model.faces.detach()[0].cpu(),
                        'object_scale': object_scale.detach().cpu(),
                        'eligible_alphas': eligible_alphas,
                    }
                    publication_alpha = 0.0
                    update_accepted = False
                    shadow_action = 'proposed'
                    mean_pose_correction = float(
                        torch.linalg.vector_norm(final_twists, dim=1).mean()
                    )
                    atomic_copy(previous_model_path, save_obj_path)
                    print(
                        f"V6.1 shadow proposed: step={max_vertex_step_ratio:.6f}; "
                        "live geometry unchanged"
                    )
                else:
                    atomic_save_mesh(model(1)[0], save_obj_path)

            if cur_model_deformed_nums == 1 and initial_geometry_export_path:
                export_path = os.path.abspath(initial_geometry_export_path)
                os.makedirs(os.path.dirname(export_path), exist_ok=True)
                shutil.copy2(save_obj_path, export_path)
                print("Exported initial geometry artifact:", export_path)

            if cur_model_deformed_nums == 1 and stop_after_initial_geometry:
                print("Initial-geometry preparation complete; stopping tracker.")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return

            if is_online_update:
                candidate_count = len(candidate_frame_ids)
                candidate_start = len(accepted) - candidate_count
                confidence_logger.append(
                    ConfidenceRow(
                        frame_id=frame_id,
                        mask_iou=mask_iou,
                        is_candidate_keyframe=True,
                        is_accepted_keyframe=bool(accepted[candidate_start + offset]),
                        geometry_update=bool(accepted[candidate_start + offset] and update_accepted),
                        pose_score=candidate_pose_scores[offset],
                        visibility_score=candidate_visibility_scores[offset],
                        temporal_score=candidate_temporal_scores[offset],
                        fused_confidence=candidate_fused_scores[offset],
                        adaptive_threshold=candidate_thresholds[offset],
                        observation_weight=candidate_weights[offset],
                        gradient_consistency=candidate_gradient_scores[offset],
                        update_accepted=update_accepted,
                        pre_reference_iou=pre_reference_iou,
                        post_reference_iou=post_reference_iou,
                        is_holdout=offset == holdout_offset,
                        pre_holdout_iou=pre_holdout_iou,
                        post_holdout_iou=post_holdout_iou,
                        max_vertex_step_ratio=max_vertex_step_ratio,
                        pre_tracking_risk=pre_tracking_risk,
                        post_tracking_risk=post_tracking_risk,
                        post_tail_iou_loss=post_tail_iou_loss,
                        mean_pose_correction=mean_pose_correction,
                        optimized_log_variance=optimized_log_variance,
                        effective_samples=effective_samples,
                        pre_cvar=pre_cvar,
                        post_cvar=post_cvar,
                        publication_alpha=publication_alpha,
                        full_step_tracking_risk=full_step_tracking_risk,
                        shadow_action=shadow_action,
                        prospective_risk_gain=prospective_risk_gain,
                        active_vertex_fraction=active_vertex_fraction,
                        mean_vertex_agreement=mean_vertex_agreement,
                    )
                    for offset, (frame_id, mask_iou) in enumerate(
                        zip(candidate_frame_ids, candidate_ious)
                    )
                )

            # Keep the Python controller alive after the final geometry update
            # so that C++ can finish the sequence and report evaluation metrics.

    
if __name__ == '__main__':
    main()
