import pickle
import numpy as np
import os
import glob

data_dir = 'vqvae/logs/recon/lightning_logs/version_0/pkl'
# load face unique
with open('data/deepcad_face_deduplicated_id.pkl', 'rb') as f:
    face_unique = pickle.load(f)['train']

face_unique_dict = {}
for face_name in face_unique:
    solid_id = face_name.split('/')[-2]
    face_id = int(face_name.split('_')[-2])
    if solid_id not in face_unique_dict:
        face_unique_dict[solid_id] = [face_id,]
    else:
        face_unique_dict[solid_id].append(face_id)

solid_all = []
face_all = []

for pkl_name in glob.glob(os.path.join(data_dir, '*', '*.pkl')):
    with open(pkl_name, 'rb') as f:
        data = pickle.load(f)
    solid_sdf = data['voxel_sdf'][...,None] # dim,n,n,n,1
    solid_all.append(solid_sdf)

    face_udf = data['faces_udf_norm'] # dim,n,n,n,m
    solid_id = pkl_name.split('/')[-2]
    if solid_id not in face_unique_dict:
        continue
    face_udf = face_udf[...,face_unique_dict[solid_id]] # dim,n,n,n,m
    face_all.append(face_udf)

solid_all = np.concatenate(solid_all, axis=-1)
face_all = np.concatenate(face_all, axis=-1)

solid_mean, solid_std = solid_all.mean(), solid_all.std()
face_mean, face_std = face_all.mean(), face_all.std()

print('solid_mean:', solid_mean, 'solid_std:', solid_std)
print('face_mean:', face_mean, 'face_std:', face_std)

# save pkl
data = {
    'solid_mean': solid_mean,
    'solid_std': solid_std,
    'face_mean': face_mean,
    'face_std': face_std,
}

with open(os.path.join(data_dir, '..', 'mean_std.pkl'), 'wb') as f:
    pickle.dump(data, f)

