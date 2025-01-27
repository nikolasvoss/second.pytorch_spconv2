import numpy as np
import torch
from spconv.pytorch.utils import PointToVoxel
# from spconv.pytorch.utils import Point2VoxelCPU
# from second.protos import voxel_generator_pb2


def build(voxel_config):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        TODO: raises nothing, voxel_generator_pb2.VoxelGenerator need to be changed
        old:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
#    if not isinstance(voxel_config, (voxel_generator_pb2.VoxelGenerator)):
#        raise ValueError('input_reader_config not of type '
#                         'input_reader_pb2.InputReader.')
    voxel_generator = PointToVoxel(
        vsize_xyz=list(voxel_config.voxel_size),
        coors_range_xyz=list(voxel_config.point_cloud_range),
        num_point_features=4, # 3 is for points xyz?
        max_num_voxels=20000,
        max_num_points_per_voxel=voxel_config.max_number_of_points_per_voxel,
        device=torch.device("cpu:0"))
    return voxel_generator
