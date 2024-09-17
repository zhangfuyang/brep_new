import torch
import torch.nn as nn
import math

def sincos_embedding(input, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param input: a N-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim //2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) /half
    ).to(device=input.device)
    for _ in range(len(input.size())):
        freqs = freqs[None]
    args = input.unsqueeze(-1).float() * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Solid3DNet(nn.Module):
    def __init__(self, config):
        super(Solid3DNet, self).__init__()
        self.config = config
        self.embed_dim = config['d_model']

        self.solid_model = UNet3DModel(config['solid_model'])

        self.time_embed = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.LayerNorm(config['d_model']),
            nn.SiLU(),
            nn.Linear(config['d_model'], config['d_model'])
        )

    def forward(self, solid, timesteps):
        # solid: b,dim,n,n,n
        # timesteps: b
        bs = solid.shape[0]
        time_embed = self.time_embed(sincos_embedding(timesteps, self.embed_dim))

        solid_ori = solid

        ##### 1. encode
        #bbox_embed_avg = (bbox_embed * mask[...,None]).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        bbox_embed_avg = None
        solid_latent, solid_res_blocks = self.solid_model.encode(solid, time_embed, bbox_embed_avg)
        solid_latent = self.solid_model.mid(solid_latent, time_embed, bbox_embed_avg)

        ##### 2. decode
        solid_latent = self.solid_model.decode(solid_latent, solid_res_blocks, time_embed, bbox_embed_avg)
        solid_pred = solid_ori + solid_latent

        return solid_pred 


def Normalize(in_channels, num_groups=32, eps=1e-5):
    return nn.GroupNorm(num_groups, in_channels, eps)

class ResNetBlockTimeEmbed(nn.Module):
    def __init__(self, in_channels, out_channels=None, 
                 temb_channels=512, bbox_channels=512,
                 act_fn="silu", num_groups=32, eps=1e-5, kernel_size=3,
                 stride=1, padding=1):
        super(ResNetBlockTimeEmbed, self).__init__()
        self.in_channels = in_channels
        out_channels = out_channels if out_channels is not None else in_channels
        self.out_channels = out_channels
        
        self.norm1 = Normalize(in_channels, num_groups=num_groups, eps=eps)
        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding)
        self.norm2 = Normalize(out_channels, num_groups=num_groups, eps=eps)
        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding)
        if temb_channels is not None:
            self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None
        
        if bbox_channels is not None:
            self.bbox_emb_proj = torch.nn.Linear(bbox_channels, out_channels)
        else:
            self.bbox_emb_proj = None
        
        if self.in_channels != self.out_channels:
            self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                 out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding)
        self.nonlinearity = torch.nn.SiLU()

    def forward(self, x, time_embed, bbox_embed):
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)
        h = self.conv1(h)

        if self.time_emb_proj is not None:
            temb = self.time_emb_proj(time_embed)
            temb = temb[:,:,None,None,None]
            h = h + temb
        if self.bbox_emb_proj is not None:
            temb = self.bbox_emb_proj(bbox_embed)
            temb = temb[:,:,None,None,None]
            h = h + temb
        
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)
        
        return x + h

class DownSample(nn.Module):
    def __init__(self, dims, dim):
        super(DownSample, self).__init__()
        self.conv = nn.Conv3d(dims, dim, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, dims, dim):
        super(Upsample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = nn.Conv3d(dims, dim, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return self.conv(self.up(x))

class UNet3DModel(nn.Module):
    def __init__(self, config):
        super(UNet3DModel, self).__init__()
        self.config = config

        in_channels = 4
        block_channels = self.config['block_channels']
        encode_modules = [EncodeDownBlock3D, EncodeDownBlock3D, EncodeBlock3D]
        decode_modules = [DecodeBlock3D, DecodeUpBlock3D, DecodeUpBlock3D]
        time_embed_channels = config['d_model']
        bbox_embed_channels = self.config['bbox_embed_channels']

        # input
        self.conv_in = nn.Conv3d(in_channels, block_channels[0], kernel_size=3, stride=1, padding=1)

        self.encode_blocks = nn.ModuleList([])
        self.mid_block = nn.ModuleList([])
        self.decode_blocks = nn.ModuleList([])

        # encode
        output_channel = block_channels[0]
        channel_start = 0
        skip_dims = [block_channels[0]]
        for i in range(len(block_channels)):
            input_channel = output_channel
            output_channel = block_channels[i]
            encode_block = encode_modules[i](
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_channels,
                bbox_channels=bbox_embed_channels,
                num_layers=2,
                resnet_act_fn='silu',
                num_groups=32,
                eps=1e-5,
                has_attention=False,
            )
            self.encode_blocks.append(encode_block)
            for j in range(len(encode_block.resnets)):
                skip_dims.append(output_channel)
        
        # mid
        mid_block = EncodeBlock3D(
            in_channels=block_channels[-1],
            out_channels=block_channels[-1],
            temb_channels=time_embed_channels,
            bbox_channels=bbox_embed_channels,
            num_layers=2,
            resnet_act_fn='silu',
            num_groups=32,
            eps=1e-5,
            has_attention=False,
        )
        self.mid_block.append(mid_block)

        # decode
        reverse_block_channels = block_channels[::-1] + [block_channels[0],]
        reverse_skip_channels = skip_dims[::-1]
        for i in range(len(reverse_block_channels)-1):
            input_channel = reverse_block_channels[i]
            output_channel = reverse_block_channels[i+1]
            decode_block = decode_modules[i](
                in_channels=input_channel,
                out_channels=output_channel,
                prev_out_channels=reverse_skip_channels[:2],
                temb_channels=time_embed_channels,
                bbox_channels=bbox_embed_channels,
                num_layers=2,
                resnet_act_fn='silu',
                num_groups=32,
                eps=1e-5,
                has_attention=False,
            )
            self.decode_blocks.append(decode_block)
            reverse_skip_channels = reverse_skip_channels[2:]
        
        # out
        self.conv_norm_out = Normalize(block_channels[0], num_groups=32, eps=1e-5)
        self.conv_out_act = nn.SiLU()
        self.conv_out = nn.Conv3d(block_channels[0], 4, kernel_size=3, stride=1, padding=1)

    def encode(self, x, time_embed, bbox_embed=None):
        x = self.conv_in(x)

        down_block_res_samples = (x, )
        for encode_block_i in range(len(self.encode_blocks)):
            x, res_samples = self.encode_blocks[encode_block_i](x, time_embed, bbox_embed)
            down_block_res_samples += res_samples
    
        return x, down_block_res_samples
    
    def mid(self, x, time_embed, bbox_embed=None):
        x, _ = self.mid_block[0](x, time_embed, bbox_embed)
        return x
    
    def decode(self, x, down_block_res_samples, time_embed, bbox_embed=None):
        for decode_block_i in range(len(self.decode_blocks)):
            decode_block = self.decode_blocks[decode_block_i]
            res_samples = down_block_res_samples[-len(decode_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(decode_block.resnets)]

            x = decode_block(x, time_embed, res_samples, bbox_embed)
        
        x = self.conv_norm_out(x)
        x = self.conv_out_act(x)
        x = self.conv_out(x)

        return x
    
    def forward(self, x, time_embed):
        skip_x = x
        x, down_block_res_samples = self.encode(x, time_embed)

        x = self.mid(x, time_embed)

        x = self.decode(x, down_block_res_samples, time_embed)

        x += skip_x

        return x

class EncodeBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 temb_channels, bbox_channels,
                 num_layers, resnet_act_fn, 
                 num_groups, eps, has_attention=False,):
        super(EncodeBlock3D, self).__init__()
        resnets = []
        attentions = []

        if has_attention:
            assert False, "Not implemented"
        
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResNetBlockTimeEmbed(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    bbox_channels=bbox_channels,
                    act_fn=resnet_act_fn,
                    num_groups=num_groups,
                    eps=eps,
                )
            )
            if has_attention:
                assert False, "Not implemented"
        
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

    def forward(self, x, time_embed, bbox_channels=None):
        output_states = ()

        for model_i in range(len(self.resnets)):
            resnet = self.resnets[model_i]
            x = resnet(x, time_embed, bbox_channels)
            if len(self.attentions) > 0:
                attn = self.attentions[model_i]
                x = attn(x)
            output_states = output_states + (x, )
        return x, output_states

class EncodeDownBlock3D(EncodeBlock3D):
    def __init__(self, in_channels, out_channels, 
                 temb_channels, bbox_channels,
                 num_layers, resnet_act_fn, 
                 num_groups, eps, has_attention=False,):
        super(EncodeDownBlock3D, self).__init__(in_channels, out_channels, 
                                                temb_channels, bbox_channels,
                                                num_layers, resnet_act_fn, 
                                                num_groups, eps, has_attention)
        self.downsample = DownSample(out_channels, out_channels)
    
    def forward(self, x, time_embed, bbox_channels=None):
        x, output_states = super().forward(x, time_embed, bbox_channels)
        x = self.downsample(x)
        return x, output_states

class DecodeBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 prev_out_channels, temb_channels, 
                 bbox_channels, num_layers,
                 resnet_act_fn, num_groups, eps, has_attention=False):
        super(DecodeBlock3D, self).__init__()
        resnets = []
        attentions = []

        if has_attention:
            assert False, "Not implemented"
        
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.prev_out_channels = prev_out_channels
        for i in range(num_layers):
            res_skip_channels = prev_out_channels[i]
            resnet_in_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResNetBlockTimeEmbed(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    bbox_channels=bbox_channels,
                    act_fn=resnet_act_fn,
                    num_groups=num_groups,
                    eps=eps,
                )
            )
            if has_attention:
                assert False, "Not implemented"
        
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
    
    def forward(self, x, time_embed, res_samples, bbox_channels=None):
        for model_i in range(len(self.resnets)):
            resnet = self.resnets[model_i]
            res_hidden_states = res_samples[-1]
            res_samples = res_samples[:-1]

            x = torch.cat([x, res_hidden_states], dim=1)
            x = resnet(x, time_embed, bbox_channels)
            if len(self.attentions) > 0:
                attn = self.attentions[model_i]
                x = attn(x)
        return x

class DecodeUpBlock3D(DecodeBlock3D):
    def __init__(self, in_channels, out_channels, 
                 prev_out_channels, temb_channels, 
                 bbox_channels, num_layers,
                 resnet_act_fn, num_groups, eps, has_attention=False):
        super(DecodeUpBlock3D, self).__init__(in_channels, out_channels, 
                                              prev_out_channels, temb_channels, 
                                              bbox_channels, num_layers,
                                              resnet_act_fn, num_groups, eps, has_attention)
        self.upsample = Upsample(in_channels, in_channels)
    
    def forward(self, x, time_embed, res_samples, bbox_channels=None):
        x = self.upsample(x)
        x = super().forward(x, time_embed, res_samples, bbox_channels)
        return x

