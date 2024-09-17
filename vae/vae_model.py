"""Adapted from https://github.com/milesial/Pytorch-UNet/tree/master/unet"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE3D(nn.Module):
    def __init__(self, n_channels, with_conv=True, num_res_blocks=2, ch=64,
                 channels=(2, 4, 8, 8), voxel_size=64, data_scale=1.):
        super(VAE3D, self).__init__()
        _ch = ch
        _channels = tuple(channels)
        self.n_channels = n_channels
        self.ch = ch
        self.channels = [int(c) for c in _channels]
        self.with_conv = with_conv
        self.num_res_blocks = num_res_blocks
        self.voxel_size = voxel_size
        self.data_scale = data_scale

        self.conv_in = nn.Conv3d(n_channels, _ch, 
                                kernel_size=3, stride=1, padding=1)

        self.encoder = nn.ModuleList()
        in_channels = (1, ) + _channels
        for i in range(len(_channels)):
            if i != len(_channels)-1:
                self.encoder.append(
                    Down(in_channels[i]*_ch, in_channels[i+1]*_ch, 
                         num_res_blocks=self.num_res_blocks, with_conv=self.with_conv)
                )
            else:
                self.encoder.append(
                    Mid(in_channels[i]*_ch, in_channels[i+1]*_ch, 
                        num_res_blocks=self.num_res_blocks)
                )
            
        self.fc_mu = nn.Conv3d(in_channels[-1]*_ch, in_channels[-1]*_ch, 1)
        self.fc_logvar = nn.Conv3d(in_channels[-1]*_ch, in_channels[-1]*_ch, 1)

        self.decoder_input = nn.Conv3d(in_channels[-1]*_ch, in_channels[-1]*_ch, 1)
        self.decoder = nn.ModuleList()
        for i in range(len(_channels), 0, -1):
            if i == len(_channels):
                self.decoder.append(
                    Mid(in_channels[i]*_ch, in_channels[i-1]*_ch, 
                        num_res_blocks=self.num_res_blocks)
                )
            else:
                self.decoder.append(
                    Up(in_channels[i]*_ch, in_channels[i-1]*_ch, 
                       num_res_blocks=self.num_res_blocks, with_conv=self.with_conv)
                )

        self.final_layer = nn.Sequential(
            nn.Conv3d(in_channels[0]*_ch, n_channels, kernel_size=1),
            nn.Tanh()
        )

    def encode(self, x):
        x = self.conv_in(x)
        for model in self.encoder:
            x = model(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar
    
    def decode(self, z):
        z = self.decoder_input(z)
        for model in self.decoder:
            z = model(z)
        result = self.final_layer(z) * self.data_scale
        return result
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar, **kwargs):
        kld_weight = kwargs['kld_weight']
        #if kwargs['recon_loss_type'] == 'l1':
        #    recons_loss = F.l1_loss(recon_x, x)
        #else:
        #    recons_loss = F.mse_loss(recon_x, x)
        recons_loss = F.mse_loss(recon_x, x)

        logvar = logvar.reshape(logvar.shape[0], -1)
        mu = mu.reshape(mu.shape[0], -1)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss

        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
    
    def sample(self, num_samples, device):
        latent_size = self.voxel_size // (2**(len(self.channels)-1))
        z = torch.randn(num_samples, 
                        self.channels[-1]*self.ch, 
                        latent_size, latent_size, latent_size).to(device)
        return self.decode(z)
    
class Mid(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=2):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(ResNetBlock(in_channels, out_channels))
            in_channels = out_channels
        
    def forward(self, x):
        for model in self.res_blocks:
            x = model(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=2, with_conv=True):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(ResNetBlock(in_channels, out_channels))
            in_channels = out_channels
        
        self.with_conv = with_conv
        if with_conv:
            self.downsample = nn.Conv3d(in_channels, out_channels, 
                                    kernel_size=3, stride=2, padding=0)
        else:
            self.downsample = nn.AvgPool3d(2)

    def forward(self, x):
        for model in self.res_blocks:
            x = model(x)
        if self.with_conv:
            pad = (0,1,0,1,0,1)
            x = F.pad(x, pad, "constant", 0)
        return self.downsample(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=2, with_conv=True):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(ResNetBlock(in_channels, out_channels))
            in_channels = out_channels
        
        self.with_conv = with_conv
        if with_conv:
            self.upsample = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x):
        for model in self.res_blocks:
            x = model(x)
        return self.upsample(x)

def Normalize(in_channels, num_groups=32):
    return torch.nn.BatchNorm3d(num_features=in_channels)
    #return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = Normalize(out_channels)
        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                 out_channels,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1)
        
        self.nonlinearity = nn.SiLU()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x+h

