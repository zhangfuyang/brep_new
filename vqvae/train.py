import os
import torch
import yaml
import argparse
import pytorch_lightning as pl
from vae_network import VQVAE3D
from dataset import Dataset
import trimesh
import mcubes
import numpy as np

class Trainer(pl.LightningModule):
    def __init__(self, config, network):
        super(Trainer, self).__init__()
        self.config = config
        self.network = network

    def training_step(self, batch, batch_idx):
        phase = batch[1][0]
        batch = batch[0]
        results = self.network(batch, self.config['vq_weight'])
        recon = results[0]
        vq_loss = results[1]
        code_indices = results[2]

        loss_dict = self.network.loss_function(recon, batch, vq_loss, **self.config)
        loss = loss_dict['loss']
        recon_loss = loss_dict['Reconstruction_Loss']
        vq_loss = loss_dict['VQ_Loss']
        self.log('recon_loss', recon_loss, prog_bar=True, rank_zero_only=True)
        self.log('vq_loss', vq_loss, prog_bar=True, rank_zero_only=True)
        self.log('loss', loss, prog_bar=True, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx):
        phase = batch[1][0]
        batch = batch[0]
        results = self.network(batch, self.config['vq_weight'])
        recon = results[0]
        vq_loss = results[1]
        code_indices = results[2]

        loss_dict = self.network.loss_function(recon, batch, vq_loss, **self.config)
        loss = loss_dict['loss']
        self.log('val_loss', loss, prog_bar=True)

        if self.trainer.is_global_zero:
            if batch_idx == 0:
                for i in range(len(batch)):
                    save_name = os.path.join(
                        self.logger.log_dir, 'images', f'{self.global_step}_gt_{i}.obj'
                    )
                    self.render_(batch[i][0], save_name, phase)

                    save_name = os.path.join(
                        self.logger.log_dir, 'images', f'{self.global_step}_recon_{i}.obj'
                    )
                    self.render_(recon[i][0], save_name, phase)
        
    def render_(self, voxel, save_name, phase):
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        voxel = voxel.cpu().numpy()
        if phase == 'face':
            points = np.where(voxel < 0.2)
            points = np.array(points).T
            pointcloud = trimesh.points.PointCloud(points)
            pointcloud.export(save_name)
        else:
            vertices, triangles = mcubes.marching_cubes(voxel, 0.)
            mcubes.export_obj(vertices, triangles, save_name)

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
    parser.add_argument('--config', type=str, default='vqvae/train_solid.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    network = VQVAE3D(**config['network_params'])

    experiment = Trainer(config['trainer_params'], network)

    dataset = Dataset(config['data_params'])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config['trainer_params']['batch_size'], 
        shuffle=True, num_workers=16)

    val_dataset = Dataset(config['data_params'])
    val_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=6, 
        shuffle=True, num_workers=16)
    
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
        limit_val_batches=1,
    )

    if trainer.is_global_zero:
        os.makedirs(trainer.log_dir, exist_ok=True)
        os.system(f'cp {args.config} {trainer.log_dir}/config.yaml')

    trainer.fit(experiment, dataloader, val_dataloader)




