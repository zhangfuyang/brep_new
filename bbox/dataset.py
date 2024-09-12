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

        max_bbox_num = self.config['max_bbox_num']

        bbox = torch.from_numpy(bbox).float()

        if len(bbox) > max_bbox_num:
            return self.__getitem__(torch.randint(0, len(bbox), (1,)).item())
        elif len(bbox) < max_bbox_num:
            # padding
            repeat_time = max_bbox_num // len(bbox)
            sep = max_bbox_num - len(bbox) * repeat_time
            a = torch.cat([bbox[:sep],]*(repeat_time+1), dim=0)
            b = torch.cat([bbox[sep:],]*(repeat_time), dim=0)
            bbox = torch.cat([a, b], dim=0)
        
        # shuffle bbox
        idx = torch.randperm(bbox.shape[0])
        bbox = bbox[idx]

        return bbox


if __name__ == "__main__":
    import yaml
    with open('bbox/train.yaml', 'r') as f:
        config = yaml.safe_load(f)
    dataset = Dataset(config['data_params'])
    for i in range(len(dataset)):
        data = dataset[i]
        print(i)
        
        

