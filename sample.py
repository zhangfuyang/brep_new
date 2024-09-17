import os
import torch
import argparse
import yaml
from diffusers import DDPMScheduler, PNDMScheduler
from tqdm import tqdm
import numpy as np
from bbox.network import BBoxNet
from face_solid.network import Solid3DNet
from vqvae.vae_network import VQVAE3D
import mcubes
import trimesh
import pickle
from scipy.interpolate import RegularGridInterpolator

base_color = np.array(
    [[255,   0,  0, 255],  # Red
    [  0, 255,   0, 255],  # Green
    [  0,   0, 255, 255],  # Blue
    [255, 255,   0, 255],  # Yellow
    [  0, 255, 255, 255],  # Cyan
    [255,   0, 255, 255],  # Magenta
    [255, 165,   0, 255],  # Orange
    [128,   0, 128, 255],  # Purple
    [255, 192, 203, 255],  # Pink
    [128, 128, 128, 255],  # Gray
    [210, 245, 60, 255], # Lime
    [170, 110, 40, 255], # Brown
    [128, 0, 0, 255], # Maroon
    [0, 128, 128, 255], # Teal
    [0, 0, 128, 255], # Navy
    ],
    dtype=np.uint8
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbox_config', type=str, default='bbox/train.yaml')
    parser.add_argument('--bbox_checkpoint', type=str, default='bbox/logs/debug/lightning_logs/version_1/checkpoints/last.ckpt')
    parser.add_argument('--face_solid_config', type=str, default='face_solid/logs/debug/lightning_logs/version_3/config.yaml')
    parser.add_argument('--face_solid_checkpoint', type=str, default='face_solid/logs/debug/lightning_logs/version_3/checkpoints/epoch=91-loss=0.01.ckpt')
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()
    
    batch_size = 16
    num_bbox = 50
    bbox_threshold = 0.05

    ### 1. load bbox model ###
    with open(args.bbox_config, 'r') as f:
        config = yaml.safe_load(f)
    bbox_net = BBoxNet(config['network_params'])
    # load checkpoint
    state_dict = torch.load(args.bbox_checkpoint, map_location='cpu')['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('network.'):
            new_state_dict[k[8:]] = v
        else:
            new_state_dict[k] = v
    bbox_net.load_state_dict(new_state_dict, strict=True)
    bbox_net = bbox_net.to('cuda').eval()

    ### 2. load face_solid model ###
    with open(args.face_solid_config, 'r') as f:
        config = yaml.safe_load(f)
    face_solid_net = Solid3DNet(config['network_params'])
    # load checkpoint
    state_dict = torch.load(args.face_solid_checkpoint, map_location='cpu')['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('solid_model') or k.startswith('face_model'):
            continue
        if k.startswith('network.'):
            new_state_dict[k[8:]] = v
        else:
            new_state_dict[k] = v
    face_solid_net.load_state_dict(new_state_dict, strict=True)
    face_solid_net = face_solid_net.to('cuda').eval()

    ### 3. load face & solid VAE model ###
    # load face model
    face_yaml = config['trainer_params']['face_model']['config_yaml']
    with open(face_yaml, 'r') as f:
        face_config = yaml.safe_load(f)['network_params']
    face_model = VQVAE3D(**face_config)
    state_dict = torch.load(
        config['trainer_params']['face_model']['checkpoint'], 
        map_location='cpu')['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('network.'):
            new_state_dict[k[8:]] = v
        else:
            new_state_dict[k] = v
    face_model.load_state_dict(new_state_dict, strict=True)
    face_model = face_model.to('cuda').eval()

    # load solid model
    solid_yaml = config['trainer_params']['solid_model']['config_yaml']
    with open(solid_yaml, 'r') as f:
        solid_config = yaml.safe_load(f)['network_params']
    solid_model = VQVAE3D(**solid_config)
    state_dict = torch.load(
        config['trainer_params']['solid_model']['checkpoint'], 
        map_location='cpu')['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('network.'):
            new_state_dict[k[8:]] = v
        else:
            new_state_dict[k] = v
    solid_model.load_state_dict(new_state_dict, strict=True)
    solid_model = solid_model.to('cuda').eval()

    with open(config['trainer_params']['mean_std_path'], 'rb') as f:
        mean_std = pickle.load(f)
    solid_mean, solid_std = mean_std['solid_mean'], mean_std['solid_std']
    face_mean, face_std = mean_std['face_mean'], mean_std['face_std']

    ########## sampling ##########

    pndm_scheduler = PNDMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start=0.0001,
        beta_end=0.02,)

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start=0.0001,
        beta_end=0.02,
        clip_sample=True,
        clip_sample_range=2,)

    ### 1. sample bbox ###
    with torch.no_grad():
        bbox = torch.randn(batch_size, num_bbox, 4).float().to('cuda')

        pndm_scheduler.set_timesteps(200)
        for t in tqdm(pndm_scheduler.timesteps[:158]):
            timesteps = torch.full((batch_size,), t, device='cuda').long()
            pred = bbox_net(bbox, timesteps)
            bbox = pndm_scheduler.step(pred, t, bbox).prev_sample
        
        ddpm_scheduler.set_timesteps(1000)
        for t in tqdm(ddpm_scheduler.timesteps[-250:]):
            timesteps = torch.full((batch_size,), t, device='cuda').long()
            pred = bbox_net(bbox, timesteps)
            bbox = ddpm_scheduler.step(pred, t, bbox).prev_sample
    
    # remove duplicate bbox
    bbox_raw = bbox.cpu().numpy() # (bs, num_bbox, 4) x,y,z,size
    bbox_corner1 = bbox_raw[:,:,:3]-bbox_raw[:,:,3:]/2
    bbox_corner2 = bbox_raw[:,:,:3]+bbox_raw[:,:,3:]/2
    bbox_check = np.concatenate([bbox_corner1[:,:,None], bbox_corner2[:,:,None]], axis=2) 
    # (bs, num_bbox, 2, 3) 

    bbox_deduplicate = []
    for b_i in range(batch_size):
        x = np.round(bbox_check[b_i], 4)
        non_repeat = x[:1]
        non_repeat_idx = [0]
        for i in range(1, len(x)):
            diff = np.max(np.max(np.abs(non_repeat - x[i]), -1), -1)
            same = diff < bbox_threshold
            if same.sum()>=1:
                continue
            non_repeat = np.concatenate([non_repeat, x[i][None]], axis=0)
            non_repeat_idx.append(i)
        
        deduplicate = bbox_raw[b_i][non_repeat_idx]
        bbox_deduplicate.append(deduplicate)
    

    bbox = bbox_deduplicate
    face_num_list = [len(b) for b in bbox]
    max_face_num = max(face_num_list)
    bbox_merge = []
    for i, bbox_i in enumerate(bbox):
        bbox_i = torch.cat([
            torch.from_numpy(bbox_i).float(), 
            torch.zeros((max_face_num-face_num_list[i], 4))], 
        dim=0)
        bbox_merge.append(bbox_i)
    bbox_merge = torch.stack(bbox_merge, dim=0) # (b,m,4)

    # mask
    mask = torch.zeros((batch_size, max_face_num))
    for i, face_num in enumerate(face_num_list):
        mask[i, :face_num] = 1
    

    ### 2. sample face_solid ###
    num_bbox = bbox_merge.shape[1]
    with torch.no_grad():
        bbox = bbox_merge.to('cuda') # (bs, num_bbox, 4)
        mask = mask.to('cuda')

        z_solid = torch.randn(batch_size, 4,4,4,4).to('cuda')
        z_face = torch.randn(batch_size, num_bbox,4,4,4,4).to('cuda')

        ddpm_scheduler.set_timesteps(1000)
        timesteps = ddpm_scheduler.timesteps
        for i, t in tqdm(enumerate(timesteps)):
            timestep = torch.cat([t.unsqueeze(0)]*batch_size, dim=0).to('cuda')
            pred_solid_noise, pred_face_noise = face_solid_net(
                z_solid, z_face, timestep, bbox, mask)
            z_solid = ddpm_scheduler.step(pred_solid_noise, t, z_solid).prev_sample
            z_face = ddpm_scheduler.step(pred_face_noise, t, z_face).prev_sample

    # render
    for i in range(batch_size):
        print(f'sample {i}')
        save_root = os.path.join(args.output_dir, f'{i:03d}')
        os.makedirs(save_root, exist_ok=True)
        
        solid_latent = z_solid[i][None] # (1,4,4,4,4)
        face_latent = z_face[i] # (num_bbox,4,4,4,4)
        

        solid_latent = solid_latent * solid_std + solid_mean
        face_latent = face_latent * face_std + face_mean

        with torch.no_grad():
            solid_voxel = solid_model.quantize_decode(solid_latent)[0,0] # (n,n,n)
            face_voxel = face_model.quantize_decode(face_latent)[mask[i]==1,0] # (m,n,n,n)

        bbox_i = bbox[i, mask[i]==1].cpu().numpy()
        solid_voxel_i = solid_voxel.cpu().numpy()
        face_voxel_i = face_voxel.cpu().numpy()
        solid_voxel_i = solid_voxel_i.transpose(1,0,2)
        face_voxel_i = face_voxel_i.transpose(0,2,1,3)

        vertices, triangle = mcubes.marching_cubes(solid_voxel_i, 0)
        vertices = vertices / solid_voxel_i.shape[0] * 2 - 1 # [-1,1] (k,3)
        solid_mesh = trimesh.Trimesh(vertices, triangle)
        solid_mesh.export(os.path.join(save_root, 'solid.obj'))

        center_bbox = bbox_i[:,:3]
        length_bbox = bbox_i[:,3:]

        offset = vertices[:,None] - center_bbox[None] # (k,m,3)
        relative_pos = offset + length_bbox[None]/2 # (k,m,3)
        relative_pos = relative_pos / length_bbox[None] # (k,m,3)

        coord = relative_pos * face_voxel_i.shape[1] # (k,m,3)
        coord = coord.clip(0, face_voxel_i.shape[1]-1)

        dist_ = []
        for face_i in range(face_voxel_i.shape[0]):
            dist_interpolater = RegularGridInterpolator(
                (np.arange(face_voxel_i.shape[1]), np.arange(face_voxel_i.shape[2]), np.arange(face_voxel_i.shape[3])),
                face_voxel_i[face_i], bounds_error=False, fill_value=10)
            v_dist2face = dist_interpolater(coord[:, face_i]) # k
            v_dist2face[v_dist2face>0.8] = 10
            v_dist2face = v_dist2face * length_bbox[face_i,0]
            dist_.append(v_dist2face)
        dist_ = np.stack(dist_, axis=1)

        v_face_id = np.argmin(dist_, axis=1) # k

        # save each face
        all_pc = None
        for face_i in range(face_voxel_i.shape[0]):
            v = vertices[v_face_id==face_i]
            pc = trimesh.points.PointCloud(v)
            if pc.vertices.shape[0] == 0:
                continue
            pc.visual.vertex_colors = base_color[face_i % len(base_color)]
            if all_pc is None:
                all_pc = pc
            else:
                all_pc = all_pc + pc
            
            pc.export(os.path.join(save_root, f'face_{face_i}.obj'))
        all_pc.export(os.path.join(save_root, 'all_face.obj'))
        
            
            


        

