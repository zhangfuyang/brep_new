import os
import torch
import yaml
import argparse
import pytorch_lightning as pl
from network import Solid3DNet
from dataset import LatentDataset
from diffusers import DDPMScheduler
import pickle
from tqdm import tqdm
import sys
import mcubes
import trimesh
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vqvae')))
from vae_network import VQVAE3D


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

class Trainer(pl.LightningModule):
    def __init__(self, config, network, solid_model):
        super(Trainer, self).__init__()
        self.config = config
        self.network = network

        self.solid_model = solid_model
        self.solid_model.eval()
        for param in self.solid_model.parameters():
            param.requires_grad = False
        # load mean_std
        with open(config['mean_std_path'], 'rb') as f:
            mean_std = pickle.load(f)
        self.solid_mean = mean_std['solid_mean']
        self.solid_std = mean_std['solid_std']
        
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False)
        
    @torch.no_grad()
    def preprocess(self, batch):
        solid_voxel, face_voxel, bbox, mask = batch
        bs = solid_voxel.shape[0]
        if solid_voxel.shape[1] == 1:
            # solid_voxel: (b,1,n,n,n)  face_voxel: (b,m,1,n,n,n)  
            # bbox: (b,m,4)  mask: (b,m)
            with torch.no_grad():
                solid_latent = self.solid_model.encode(solid_voxel)
        else:
            solid_latent = solid_voxel # (b,dim,n,n,n)
        
        # normalize
        solid_latent = (solid_latent - self.solid_mean) / self.solid_std

        return solid_latent
    
    def training_step(self, batch, batch_idx):
        solid_latent = self.preprocess(batch)
        device = solid_latent.device
        timesteps = torch.randint(0, 1000, (solid_latent.shape[0],), device=device).long()
        solid_noise = torch.randn_like(solid_latent)
        solid_noisy = self.scheduler.add_noise(solid_latent, solid_noise, timesteps)

        pred_solid_noise = self.network(solid_noisy, timesteps)
        
        gt_noise = solid_noise
        pred_noise = pred_solid_noise
        loss = torch.nn.functional.mse_loss(pred_noise, gt_noise)

        self.log('loss', loss, prog_bar=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        solid_latent_gt = self.preprocess(batch)

        z_solid = torch.randn_like(solid_latent_gt) # (b,dim,n,n,n)

        self.scheduler.set_timesteps(1000)
        timesteps = self.scheduler.timesteps
        bs = solid_latent_gt.shape[0]

        for i, t in tqdm(enumerate(timesteps)):
            timestep = torch.cat([t.unsqueeze(0)]*bs, dim=0).to(solid_latent_gt.device)
            pred_solid_noise = self.network(z_solid, timestep)
            z_solid = self.scheduler.step(pred_solid_noise, t, z_solid).prev_sample

        if self.trainer.is_global_zero:
            for i in range(bs):
                #### gt
                solid_voxel_i = self.latent_to_voxel(solid_latent_gt[i:i+1], 'solid')
                solid_voxel_i = solid_voxel_i[0,0].cpu().numpy() # (n,n,n)

                solid_mesh = self.render_mesh(solid_voxel_i)
                save_root = os.path.join(self.logger.log_dir, 'img')
                os.makedirs(save_root, exist_ok=True)
                solid_mesh.export(os.path.join(save_root, f'{self.global_step}_{batch_idx}_{i}_solid_gt.obj'))
                #### pred
                solid_voxel_i = self.latent_to_voxel(z_solid[i:i+1], 'solid')
                solid_voxel_i = solid_voxel_i[0,0].cpu().numpy() # (n,n,n)

                solid_mesh = self.render_mesh(solid_voxel_i)
                save_root = os.path.join(self.logger.log_dir, 'img')
                os.makedirs(save_root, exist_ok=True)
                solid_mesh.export(os.path.join(save_root, f'{self.global_step}_{batch_idx}_{i}_solid_pred.obj'))


    def latent_to_voxel(self, latent, phase):
        if phase == 'solid':
            latent = latent * self.solid_std + self.solid_mean
            with torch.no_grad():
                voxel = self.solid_model.quantize_decode(latent)
        elif phase == 'face':
            latent = latent * self.face_std + self.face_mean
            with torch.no_grad():
                voxel = self.face_model.quantize_decode(latent)

        return voxel

    def render_mesh(self, solid_voxel):
        # solid_voxel: n,n,n  face_voxel: m,n,n,n  bbox: m,4

        solid_voxel = solid_voxel.transpose(1,0,2) # n,n,n

        vertices, triangles = mcubes.marching_cubes(solid_voxel, 0)
        vertices = vertices / solid_voxel.shape[0] * 2 - 1 # [-1, 1] (k,3)
        solid_mesh = trimesh.Trimesh(vertices, triangles)

        return solid_mesh
    
    def on_train_epoch_end(self):
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', cur_lr, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.config['lr'],
            weight_decay=self.config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.config['lr_decay'])
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='solid_only/train.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    network = Solid3DNet(config['network_params'])

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


    experiment = Trainer(config['trainer_params'], network, solid_model)

    dataset = LatentDataset(config['data_params'])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config['trainer_params']['batch_size'], 
        shuffle=True, num_workers=16, collate_fn=dataset.collate_fn)
    
    val_dataset = LatentDataset(config['data_params'])
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8, 
        shuffle=True, num_workers=8, collate_fn=dataset.collate_fn)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1, monitor='loss', mode='min',
        save_last=True, filename='{epoch}-{loss:.2f}')

    lr_monitor = pl.callbacks.LearningRateMonitor()

    trainer_config = config['trainer_params']
    trainer = pl.Trainer(
        accelerator=trainer_config['accelerator'],
        max_epochs=trainer_config['max_epochs'],
        num_nodes=1, devices=trainer_config['devices'],
        strategy=trainer_config['strategy'],
        log_every_n_steps=trainer_config['log_every_n_steps'],
        default_root_dir=trainer_config['default_root_dir'],
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=trainer_config['gradient_clip_val'],
        num_sanity_val_steps=1,
        val_check_interval=1.,
        limit_val_batches=1,
    )

    if trainer.is_global_zero:
        os.makedirs(trainer.log_dir, exist_ok=True)
        os.system(f'cp {args.config} {trainer.log_dir}/config.yaml')

    trainer.fit(experiment, dataloader, val_dataloader)




