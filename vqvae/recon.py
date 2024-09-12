import os
import torch
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from dataset import ReconDataset
from vae_network import VQVAE3D
import mcubes
import trimesh
import pickle
import numpy as np

class ReconVAEExperiment(pl.LightningModule):
    def __init__(self, config, solid_model, face_model):
        super(ReconVAEExperiment, self).__init__()
        self.config = config
        self.solid_model = solid_model
        self.face_model = face_model
        self.solid_model.eval()
        self.face_model.eval()
    
    def validation_step(self, batch, batch_idx):
        solid_voxel, faces_voxel, filenames, faces_num = \
            batch['solid_voxel'], batch['faces_voxel'], batch['filename'], batch['faces_num']
        faces_bbox = batch['faces_bbox']
        
        solid_latent = self.solid_model.encode(solid_voxel) # bs, dim, N, N, N

        # minibatch for face model
        faces_mini_batch = 32
        face_latent = []
        for i in range(0, faces_voxel.shape[0], faces_mini_batch):
            result = self.face_model.encode(faces_voxel[i:i+faces_mini_batch])
            face_latent.append(result)
        face_latent = torch.cat(face_latent, dim=0) # ?, dim, N, N, N

        # save pkl
        face_num_start = 0
        for i in range(solid_voxel.shape[0]):
            filename = filenames[i]
            save_path = os.path.join(self.logger.log_dir, 'pkl', f'{filename}')
            save_path = save_path.replace('npz', 'pkl')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            data = {}
            data['voxel_sdf'] = solid_latent[i].cpu().numpy() # dim, N, N, N
            num = faces_num[i]
            face_result = face_latent[face_num_start:face_num_start+num] #?, dim, N, N, N
            face_result = face_result.permute(1, 2, 3, 4, 0) #dim, N, N, N, ?
            data['faces_udf_norm'] = face_result.cpu().numpy()
            data['face_bboxes'] = faces_bbox[face_num_start:face_num_start+num].cpu().numpy()
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
            
            face_num_start += num
        

        ##### decode #####
        solid_recon = self.solid_model.quantize_decode(solid_latent) # bs, 1, 64, 64, 64
        # minibatch for face model
        face_recon = []
        for i in range(0, face_latent.shape[0], faces_mini_batch):
            result = self.face_model.decode(face_latent[i:i+faces_mini_batch])
            face_recon.append(result)
        face_recon = torch.cat(face_recon, dim=0) # ?, 1, 64, 64, 64

        # render
        solid_recon = solid_recon.cpu().numpy()
        face_recon = face_recon.cpu().numpy()
        face_num_start = 0
        for i in range(solid_recon.shape[0]):
            filename = filenames[i]
            save_root = os.path.join(self.logger.log_dir, 'render', filename[:-4])
            os.makedirs(save_root, exist_ok=True)
            v, t = mcubes.marching_cubes(solid_recon[i, 0], 0)
            mcubes.export_obj(v, t, os.path.join(save_root, 'solid.obj'))

            num = faces_num[i]
            points_all = []
            for j in range(num):
                points = np.where(face_recon[face_num_start+j, 0]<0.2)
                points = np.array(points).T
                pointcloud = trimesh.points.PointCloud(points)
                pointcloud.export(os.path.join(save_root, f'face_{j}.obj'))

                points_all.append(points)
            points_all = np.concatenate(points_all, axis=0)
            pointcloud = trimesh.points.PointCloud(points_all)
            pointcloud.export(os.path.join(save_root, 'faces_all.obj'))
            face_num_start += num


        


def load_model(model_class, model_yaml_config, pretrained_model_path=None, exclude_prefix='network.'):
    with open(model_yaml_config, 'r') as f:
        model_config = yaml.safe_load(f)['network_params']
    model = model_class(**model_config)
    state_dict = torch.load(pretrained_model_path, map_location='cpu')['state_dict']
    # modify the key names
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(exclude_prefix):
            new_state_dict[k[len(exclude_prefix):]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='vqvae/recon.yaml')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

solid_model = load_model(VQVAE3D, 
                        config['trainer_params']['solid_model']['config_yaml'],
                        config['trainer_params']['solid_model']['checkpoint'],
                        exclude_prefix='network.')
face_model = load_model(VQVAE3D,
                        config['trainer_params']['face_model']['config_yaml'],
                        config['trainer_params']['face_model']['checkpoint'],
                        exclude_prefix='network.')
experiment = ReconVAEExperiment(config['trainer_params'], solid_model, face_model)

val_dataset = ReconDataset(config['data_params']['train'])
print(len(val_dataset))
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=16,
    shuffle=False, num_workers=0,#8,
    collate_fn=val_dataset.collate_fn)

trainer = pl.Trainer(
    accelerator='gpu', 
    precision=32, 
    devices=1, strategy='ddp',
    default_root_dir=config['trainer_params']['default_root_dir'])

# cp yaml file
if trainer.is_global_zero:
    os.makedirs(trainer.log_dir, exist_ok=True)
    os.system(f'cp {args.config} {trainer.log_dir}/config.yaml')

trainer.validate(experiment, val_dataloader)

