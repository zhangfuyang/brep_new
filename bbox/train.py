import os
import torch
import yaml
import argparse
import pytorch_lightning as pl
from network import BBoxNet
from dataset import Dataset
from diffusers import DDPMScheduler

class Trainer(pl.LightningModule):
    def __init__(self, config, network):
        super(Trainer, self).__init__()
        self.config = config
        self.network = network

        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,)
    
    def training_step(self, batch, batch_idx):
        device = batch.device
        timesteps = torch.randint(0, 1000, (batch.shape[0],), device=device).long()
        noise = torch.randn_like(batch)
        x_noisy = self.scheduler.add_noise(batch, noise, timesteps)

        pred_noise = self.network(x_noisy, timesteps)
        loss = torch.nn.functional.mse_loss(pred_noise, noise)

        self.log('loss', loss, prog_bar=True)
        return loss
    
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
    parser.add_argument('--config', type=str, default='bbox/train.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    network = BBoxNet(config['network_params'])

    experiment = Trainer(config['trainer_params'], network)

    dataset = Dataset(config['data_params'])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config['trainer_params']['batch_size'], 
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
    )

    if trainer.is_global_zero:
        os.makedirs(trainer.log_dir, exist_ok=True)
        os.system(f'cp {args.config} {trainer.log_dir}/config.yaml')

    trainer.fit(experiment, dataloader)



