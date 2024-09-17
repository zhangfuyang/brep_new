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
        data = np.load(data_path)

        bbox = data['face_bboxes']
        solid_sdf = data['voxel_sdf'] # (n,n,n)
        face_udf = data['faces_udf_norm'] # (n,n,n,  m)

        max_bbox_num = self.config['max_bbox_num']

        bbox = torch.from_numpy(bbox).float()
        solid_sdf = torch.from_numpy(solid_sdf).float()
        face_udf = torch.from_numpy(face_udf).float()

        if len(bbox) > max_bbox_num:
            return self.__getitem__(torch.randint(0, len(bbox), (1,)).item())
        
        # shuffle
        idx = torch.randperm(bbox.shape[0]) # (m,4)
        bbox = bbox[idx]
        face_udf = face_udf[...,idx]

        return solid_sdf, face_udf, bbox
    
    def collate_fn(self, batch):
        solid_sdf, face_udf, bbox = zip(*batch)
        # solid_sdf: n,n,n  face_udf: n,n,n,m  bbox: m,4
        bs = len(solid_sdf)
        size = solid_sdf[0].shape[0]

        face_num_list = [len(b) for b in bbox]
        max_face_num = max(face_num_list)

        solid_sdf_merge = torch.stack(solid_sdf, dim=0) # (b,n,n,n)
        solid_sdf_merge = solid_sdf_merge[:,None] # (b,1,n,n,n)
        solid_sdf_merge = solid_sdf_merge * 10.

        face_udf_merge = []
        for i, face_udf_i in enumerate(face_udf):
            face_udf_i = torch.cat([
                face_udf_i, 
                torch.zeros((size, size, size, max_face_num-face_num_list[i]))], 
            dim=-1)
            face_udf_merge.append(face_udf_i)
        face_udf_merge = torch.stack(face_udf_merge, dim=0) # (b,n,n,n,m)
        face_udf_merge = face_udf_merge.permute(0, 4, 1, 2, 3) # (b,m,n,n,n)
        face_udf_merge = face_udf_merge[:,:,None] # (b,m,1,n,n,n)
        face_udf_merge = face_udf_merge * 10.

        bbox_merge = []
        for i, bbox_i in enumerate(bbox):
            bbox_i = torch.cat([
                bbox_i, 
                torch.zeros((max_face_num-face_num_list[i], 4))], 
            dim=0)
            bbox_merge.append(bbox_i)
        bbox_merge = torch.stack(bbox_merge, dim=0) # (b,m,4)

        # mask
        mask = torch.zeros((bs, max_face_num))
        for i, face_num in enumerate(face_num_list):
            mask[i, :face_num] = 1
        
        return solid_sdf_merge, face_udf_merge, bbox_merge, mask

        
class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        data_dir = self.config['latent_dir']
        self.data_id_list = glob.glob(os.path.join(data_dir, '*', '*.pkl'))
        self.data_id_list = sorted(self.data_id_list)
        
    def __len__(self):
        return len(self.data_id_list) * 25
    
    def __getitem__(self, idx):
        idx = idx % len(self.data_id_list)
        data_path = self.data_id_list[idx]

        data = pickle.load(open(data_path, 'rb'))

        bbox = data['face_bboxes'] # (m,4)
        solid_sdf = data['voxel_sdf'] # (dim,n,n,n)
        face_udf = data['faces_udf_norm'] # (dim,n,n,n,m)

        max_bbox_num = self.config['max_bbox_num']

        bbox = torch.from_numpy(bbox).float()
        solid_sdf = torch.from_numpy(solid_sdf).float()
        face_udf = torch.from_numpy(face_udf).float()

        if len(bbox) > max_bbox_num:
            return self.__getitem__(torch.randint(0, len(bbox), (1,)).item())
        
        # shuffle
        #idx = torch.randperm(bbox.shape[0]) 
        #bbox = bbox[idx]
        #face_udf = face_udf[...,idx]

        return solid_sdf, face_udf, bbox
    
    def collate_fn(self, batch):
        solid_sdf, face_udf, bbox = zip(*batch)
        # solid_sdf: n,n,n  face_udf: n,n,n,m  bbox: m,4
        bs = len(solid_sdf)
        size = solid_sdf[0].shape[0]

        face_num_list = [len(b) for b in bbox]
        max_face_num = max(face_num_list)

        solid_sdf_merge = torch.stack(solid_sdf, dim=0) # (b,dim,n,n,n)

        face_udf_merge = []
        for i, face_udf_i in enumerate(face_udf):
            face_udf_i = torch.cat([
                face_udf_i, 
                torch.zeros((*face_udf_i.shape[:-1], max_face_num-face_num_list[i]))], 
            dim=-1)
            face_udf_merge.append(face_udf_i)
        face_udf_merge = torch.stack(face_udf_merge, dim=0) # (b,dim,n,n,n,m)
        face_udf_merge = face_udf_merge.permute(0, 5, 1, 2, 3, 4) # (b,m,dim,n,n,n)

        bbox_merge = []
        for i, bbox_i in enumerate(bbox):
            bbox_i = torch.cat([
                bbox_i, 
                torch.zeros((max_face_num-face_num_list[i], 4))], 
            dim=0)
            bbox_merge.append(bbox_i)
        bbox_merge = torch.stack(bbox_merge, dim=0) # (b,m,4)

        # mask
        mask = torch.zeros((bs, max_face_num))
        for i, face_num in enumerate(face_num_list):
            mask[i, :face_num] = 1
        
        return solid_sdf_merge, face_udf_merge, bbox_merge, mask

        
if __name__ == "__main__":
    import yaml
    with open('face_solid/train.yaml', 'r') as f:
        config = yaml.safe_load(f)
    dataset = LatentDataset(config['data_params'])
    for i in range(len(dataset)):
        data = dataset[i]
        print(i)
        
        


