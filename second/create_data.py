import copy
from pathlib import Path
import pickle

import fire

import second.data.kitti_dataset as kitti_ds
import second.data.nuscenes_dataset as nu_ds
from second.data.all_dataset import create_groundtruth_database

def kitti_data_prep(root_path):
    kitti_ds.create_kitti_info_file(root_path)
    kitti_ds.create_reduced_point_cloud(root_path)
    create_groundtruth_database("KittiDataset", root_path, Path(root_path) / "kitti_infos_train.pkl")

def nuscenes_data_prep(root_path, version, dataset_name, max_sweeps=10, output_path=None):
    nu_ds.create_nuscenes_infos(root_path, version=version, max_sweeps=max_sweeps, output_path=output_path)
    name = "infos_train.pkl"
    if version == "v1.0-test":
        name = "infos_test.pkl"
    create_groundtruth_database(dataset_name, root_path, Path(root_path) / name, db_info_save_path=output_path)

if __name__ == '__main__':
    fire.Fire()
