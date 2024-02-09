import torch
import spconv.utils
from spconv.pytorch.utils import PointToVoxel, gather_features_by_pc_voxel_id
import numpy as np

# select device to run (cuda or cpu)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


gen = PointToVoxel(vsize_xyz=[0.1, 0.1, 0.1],
                       coors_range_xyz=[-80, -80, -6, 80, 80, 6],
                       num_point_features=3,
                       max_num_voxels=5000,
                       max_num_points_per_voxel=5,
                       device=device)
sample_pc =np.random.rand(25000,3).astype(np.float32)
torch_pc = torch.from_numpy(sample_pc).to(device)
voxels, coordinates, num_points = gen(torch_pc)

print(voxels.shape)
print(coordinates.shape)
print(num_points.shape)

