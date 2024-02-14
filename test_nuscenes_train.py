from second.pytorch import train

#nuscenes_data_prep('/home/niko/Documents/nuscenes/v1.0-mini', "v1.0-mini", dataset_name="NuscenesDataset", max_sweeps=10)
train.train(config_path="/home/niko/second.pytorch_spconv2/second/configs/nuscenes/all.pp.mida.config",
    model_dir="/home/niko/Documents/models",
    resume=True)



