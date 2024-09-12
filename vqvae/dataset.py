import glob
import torch
import pickle
import os
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        pkl_path = self.config['pkl_path']
        self.data_id_list = pickle.load(open(pkl_path, 'rb'))['train']
    
    def __len__(self):
        return len(self.data_id_list)
    
    def __getitem__(self, idx):
        data_path = self.data_id_list[idx]
        npz_path = os.path.join(*data_path.split('/')[:-1], 'solid_0.npz')
        data = np.load(npz_path)
        if self.config['key'] == 'face':
            face_idx = int(data_path.split('/')[-1].split('_')[2])
            voxel = data['faces_udf_norm'][..., face_idx] # (64, 64, 64)
        else:
            voxel = data['voxel_sdf']
        
        voxel = torch.from_numpy(voxel).float()
        voxel = voxel.unsqueeze(0) # (1, 64, 64, 64)
        voxel = voxel * 10.

        return voxel, self.config['key']

class ReconDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        pkl_path = self.config['pkl_path']
        self.data_id_list = pickle.load(open(pkl_path, 'rb'))['train']
    
    def __len__(self):
        return len(self.data_id_list)
    
    def __getitem__(self, idx):
        data_path = self.data_id_list[idx]
        npz_path = os.path.join(*data_path.split('/')[:-1], 'solid_0.npz')
        data = np.load(npz_path)
        solid_voxel = torch.from_numpy(data['voxel_sdf']).float().unsqueeze(0) * 10.
        faces_voxel = torch.from_numpy(data['faces_udf_norm']).float().unsqueeze(0) * 10.
        faces_voxel = faces_voxel.permute(4, 0, 1, 2, 3)
        filename = '/'.join(data_path.split('/')[-2:])
        
        faces_bboxes = data['face_bboxes']
        faces_bboxes = torch.from_numpy(faces_bboxes).float()
        return solid_voxel, faces_voxel, filename, faces_bboxes
    
    def collate_fn(self, batch):
        solid_voxels, faces_voxels, filenames, faces_bboxes = zip(*batch)
        solid_voxels = torch.stack(solid_voxels) # bs, 1, 64, 64, 64

        faces_num = [x.shape[0] for x in faces_voxels]
        faces_voxels = torch.cat(faces_voxels, dim=0)

        faces_bboxes = torch.cat(faces_bboxes, dim=0) # ?, 4

        return {'solid_voxel': solid_voxels, 
                'faces_voxel': faces_voxels, 
                'faces_bbox': faces_bboxes,
                'filename': filenames, 
                'faces_num': faces_num}


if __name__ == "__main__":
    import yaml
    with open('vqvae/recon.yaml', 'r') as f:
        config = yaml.safe_load(f)
    dataset = ReconDataset(config['data_params']['train'])
    for i in range(len(dataset)):
        data = dataset[i]
        print(i)
        
        



