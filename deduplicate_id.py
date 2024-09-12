import math
import pickle 
import argparse
from tqdm import tqdm
from hashlib import sha256
#from convert_utils import *
import glob
import trimesh
import os
import numpy as np


def real2bit(data, n_bits=8, min_range=-1, max_range=1):
    """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
    range_quantize = 2**n_bits - 1
    data_quantize = (data - min_range) * range_quantize / (max_range - min_range)
    data_quantize = np.clip(data_quantize, a_min=0, a_max=range_quantize) # clip values
    return data_quantize.astype(int) 

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='data/deepcad')
parser.add_argument("--bit",  type=int, default=6, help='Deduplicate precision')
parser.add_argument("--output", type=str, default='deepcad_face_deduplicated_id.pkl')
args = parser.parse_args()

solid_paths = []
face_paths = []
for solid_name in sorted(os.listdir(args.data_path)):
    if os.path.exists(os.path.join(args.data_path, solid_name, 'solid_0.npz')):
        solid_paths.append(os.path.join(args.data_path, solid_name))
        face_objs = glob.glob(os.path.join(args.data_path, solid_name, 'face_*_norm.obj'))
        face_objs = sorted(face_objs)
        face_paths.extend(face_objs)
# Remove duplicate for the training set 
train_path = []
unique_hash = set()
total = 0

for path_idx, face_path in tqdm(enumerate(face_paths)):
    total += 1

    mesh = trimesh.load_mesh(face_path)
    vertices = mesh.vertices
    # sort the vertices
    order = np.lexsort((vertices[:,2], vertices[:,1], vertices[:,0]))
    vertices = vertices[order]
    
    # hash the vertices
    vertices_bit = real2bit(vertices, n_bits=args.bit)  # bits
    data_hash = sha256(vertices_bit.tobytes()).hexdigest()

    # Save non-duplicate shapes
    prev_len = len(unique_hash)
    unique_hash.add(data_hash)  
    if prev_len < len(unique_hash):
        train_path.append(face_path)
    else:
        continue
        
    if path_idx % 2000 == 0:
        print(len(unique_hash)/total)

# save data 
data_path = {
    'train':train_path,
}
print('Saving data...')
print('Total number of training data:', len(train_path))
with open(args.output, "wb") as tf:
    pickle.dump(data_path, tf)


