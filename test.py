import torch
import spconv.utils
from spconv.pytorch.utils import PointToVoxel, gather_features_by_pc_voxel_id
import numpy as np

# select device to run (cuda or cpu)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def main_pytorch_voxel_gen_cuda():
    np.random.seed(50051)
    # voxel gen source code: spconv/csrc/sparse/pointops.py
    pc = np.random.uniform(-2, 8, size=[1000, 3]).astype(np.float32)

    for device in [torch.device("cuda:0"), torch.device("cpu:0")]:
        gen = PointToVoxel(vsize_xyz=[0.25, 0.25, 0.25],
                        coors_range_xyz=[0, 0, 0, 10, 10, 10],
                        num_point_features=3,
                        max_number_of_voxels=5000,
                        max_num_points_per_voxel=5,
                        device=device)

        pc_th = torch.from_numpy(pc).to(device)
        voxels_th, indices_th, num_p_in_vx_th = gen(pc_th)
        voxels_np = voxels_th.cpu().numpy()
        indices_np = indices_th.cpu().numpy()
        num_p_in_vx_np = num_p_in_vx_th.cpu().numpy()
        print(f"------{device} Raw Voxels {voxels_np.shape[0]}-------")
        print(voxels_np[0])
        # run voxel gen and FILL MEAN VALUE to voxel remain
        voxels_tv, indices_tv, num_p_in_vx_tv = gen(pc_th, empty_mean=True)
        voxels_np = voxels_tv.cpu().numpy()
        indices_np = indices_tv.cpu().numpy()
        num_p_in_vx_np = num_p_in_vx_tv.cpu().numpy()
        print(f"------{device} Voxels with mean filled-------")
        print(voxels_np[0])
        voxels_th, indices_th, num_p_in_vx_th, pc_voxel_id = gen.generate_voxel_with_id(pc_th, empty_mean=True)
        print(f"------{device} Reconstruct Indices From Voxel ids for every point-------")
        indices_th_float = indices_th.float()
        # we gather indices by voxel_id to see correctness of voxel id.
        indices_th_voxel_id = gather_features_by_pc_voxel_id(indices_th_float, pc_voxel_id)
        indices_th_voxel_id_np = indices_th_voxel_id[:10].cpu().numpy()
        print(pc[:10])
        print(indices_th_voxel_id_np[:, ::-1] / 4)

main_pytorch_voxel_gen_cuda()

gen = PointToVoxel(vsize_xyz=[0.1, 0.1, 0.1],
                       coors_range_xyz=[-80, -80, -6, 80, 80, 6],
                       num_point_features=3,
                       max_number_of_voxels=5000,
                       max_num_points_per_voxel=5,
                       device=device)
sample_pc =np.random.rand(25000,3).astype(np.float32)
torch_pc = torch.from_numpy(sample_pc).to(device)
voxels, coordinates, num_points = gen(torch_pc)

print(voxels.shape)
print(coordinates.shape)
print(num_points.shape)

