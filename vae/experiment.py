import os
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import pytorch_lightning as pl
import mcubes

class VAEExperiment(pl.LightningModule):
    def __init__(self, config, vae_model):
        super(VAEExperiment, self).__init__()
        self.config = config
        self.vae_model = vae_model
    
    def training_step(self, batch, batch_idx):
        results = self.vae_model(batch)
        recon = results[0]
        mu = results[1]
        logvar = results[2]

        loss_dict = self.vae_model.loss_function(
            recon, batch, mu, logvar, **self.config)
        loss = loss_dict['loss']
        recon_loss = loss_dict['Reconstruction_Loss']
        kld = loss_dict['KLD']
        self.log('recon_loss', recon_loss, rank_zero_only=True, prog_bar=True)
        self.log('kld', kld, rank_zero_only=True, prog_bar=True)
        self.log('train_loss', loss, rank_zero_only=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        results = self.vae_model(batch)
        recon = results[0]
        mu = results[1]
        logvar = results[2]

        loss_dict = self.vae_model.loss_function(recon, batch, mu, logvar, **self.config)
        loss = loss_dict['loss']
        self.log('val_loss', loss, rank_zero_only=True, prog_bar=True)
        if self.trainer.is_global_zero:
            if batch_idx == 0:
                for i in range(4):
                    save_name = os.path.join(self.logger.log_dir, 'images', 
                                             f'{self.global_step}_gt_{i}.obj')
                    self.render_mesh(batch[i][0], save_name)

                    save_name = os.path.join(self.logger.log_dir, 'images', 
                                             f'{self.global_step}_recon_{i}.obj')
                    self.render_mesh(recon[i][0], save_name)

    
    def on_validation_end(self):
        if self.trainer.is_global_zero:
            self.sample_images()
    
    def sample_images(self):
        samples = self.vae_model.sample(2, self.device)
        for i, sample in enumerate(samples):
            save_name = os.path.join(self.logger.log_dir, 'images', 
                                     f'{self.global_step}_sample_{i}.obj')
            self.render_mesh(sample[0], save_name)
        
    def render_mesh(self, voxel, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        voxel = voxel.cpu().numpy()
        voxel = voxel / 3
        vertices, triangles = mcubes.marching_cubes(voxel, 0)
        mcubes.export_obj(vertices, triangles, filename)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.vae_model.parameters(), 
                                     lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_decay'])
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
            
    