import copy
import json
import os
from pathlib import Path
import pickle
import shutil
import time
import re 
import fire
import numpy as np
import torch
from google.protobuf import text_format

import second.data.kitti_common as kitti
import torchplus
from second.builder import target_assigner_builder, voxel_builder
from second.core import box_np_ops
from second.data.preprocess import merge_second_batch, merge_second_batch_multigpu
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.utils.log_tool import SimpleModelLog
from second.utils.progress_bar import ProgressBar
import psutil

def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "importance"
    ]
    for k, v in example.items():
        if k in float_names:
            # slow when directly provide fp32 data with dtype=torch.half
            example_torch[k] = torch.tensor(
                v, dtype=torch.float32, device=device).to(dtype)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.uint8, device=device)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = torch.tensor(
                    v1, dtype=dtype, device=device).to(dtype)
            example_torch[k] = calib
        elif k == "num_voxels":
            example_torch[k] = torch.tensor(v)
        else:
            example_torch[k] = v
    return example_torch


def build_network(model_cfg, measure_time=False):
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = [
        voxel_generator.coors_range[0],
        voxel_generator.coors_range[1],
        voxel_generator.coors_range[3],
        voxel_generator.coors_range[4]
    ]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    box_coder.custom_ndim = target_assigner._anchor_generators[0].custom_ndim
    net = second_builder.build(
        model_cfg, voxel_generator, target_assigner, measure_time=measure_time)
    return net

def _worker_init_fn(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)
    print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])

def freeze_params(params: dict, include: str=None, exclude: str=None):
    assert isinstance(params, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    remain_params = []
    for k, p in params.items():
        if include_re is not None:
            if include_re.match(k) is not None:
                continue 
        if exclude_re is not None:
            if exclude_re.match(k) is None:
                continue 
        remain_params.append(p)
    return remain_params

def freeze_params_v2(params: dict, include: str=None, exclude: str=None):
    assert isinstance(params, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    for k, p in params.items():
        if include_re is not None:
            if include_re.match(k) is not None:
                p.requires_grad = False
        if exclude_re is not None:
            if exclude_re.match(k) is None:
                p.requires_grad = False

def filter_param_dict(state_dict: dict, include: str=None, exclude: str=None):
    assert isinstance(state_dict, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    res_dict = {}
    for k, p in state_dict.items():
        if include_re is not None:
            if include_re.match(k) is None:
                continue
        if exclude_re is not None:
            if exclude_re.match(k) is not None:
                continue 
        res_dict[k] = p
    return res_dict


def train(config_path,
          model_dir,
          result_path=None,
          create_folder=False,
          display_step=50,
          summary_step=5,
          pretrained_path=None,
          pretrained_include=None,
          pretrained_exclude=None,
          freeze_include=None,
          freeze_exclude=None,
          multi_gpu=False,
          measure_time=False,
          resume=False):
    """train a VoxelNet model specified by a config file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure the model directory is an absolute path
    model_dir = str(Path(model_dir).resolve())
    # If 'create_folder' flag is set, make a new folder if necessary
    if create_folder:
        if Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)
    model_dir = Path(model_dir)
    # Raise an error if not resuming training and the model directory already exists
    if not resume and model_dir.exists():
        raise ValueError("model dir exists and you don't specify resume.")
    # Create model directory if it does not exist
    model_dir.mkdir(parents=True, exist_ok=True)
    # Set default result path if not provided
    if result_path is None:
        result_path = model_dir / 'results'
    # Backup the configuration file for the model
    config_file_bkp = "pipeline.config"
    # Read the configuration for the model based on whether a filepath or object have been provided
    if isinstance(config_path, str): # If a filepath is provided, read config from it
        # directly provide a config object. this usually used
        # when you want to train with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:  # Config is directly provided as an object
        config = config_path
        proto_str = text_format.MessageToString(config, indent=2)
    # Write the configuration to a file in the model directory
    with (model_dir / config_file_bkp).open("w") as f:
        f.write(proto_str)

    # Parse config for different parts of the model
    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    # Build the network and move it to the device (GPU or CPU)
    net = build_network(model_cfg, measure_time).to(device)
    # if train_cfg.enable_mixed_precision:
    #     net.half()
    #     net.metrics_to_float()
    #     net.convert_norm_to_float(net)
    # Initialize variables for the target assigner and voxel generator
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    # Print the total number of parameters in the network
    print("num parameters:", len(list(net.parameters())))
    # Try to restore the latest checkpoints (if any)
    torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    # Load pretrained model if the path is given
    if pretrained_path is not None:
        model_dict = net.state_dict()
        pretrained_dict = torch.load(pretrained_path)
        pretrained_dict = filter_param_dict(pretrained_dict, pretrained_include, pretrained_exclude)
        new_pretrained_dict = {}
        # Update the model with weights from the pretrained model where shapes match
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                new_pretrained_dict[k] = v        
        print("Load pretrained parameters:")
        for k, v in new_pretrained_dict.items():
            print(k, v.shape)
        model_dict.update(new_pretrained_dict) 
        net.load_state_dict(model_dict)
        # Freeze parameters if specified
        freeze_params_v2(dict(net.named_parameters()), freeze_include, freeze_exclude)
        # Reset global step and metrics in case of fine-tuning from pretrained model
        net.clear_global_step()
        net.clear_metrics()
    if multi_gpu:  # Set up DataParallel if multiple GPUs are being used
        net_parallel = torch.nn.DataParallel(net)
    else:
        net_parallel = net
    # Set up the optimizer according to the configuration
    optimizer_cfg = train_cfg.optimizer
    loss_scale = train_cfg.loss_scale_factor
    fastai_optimizer = optimizer_builder.build(
        optimizer_cfg,
        net,
        mixed=False,
        loss_scale=loss_scale)
    # Set dynamic loss scale if configured
    if loss_scale < 0:
        loss_scale = "dynamic"
    # Set up the learning rate scheduler, enable mixed precision training if specified
    if train_cfg.enable_mixed_precision:
        max_num_voxels = input_cfg.preprocess.max_number_of_voxels * input_cfg.batch_size
        assert max_num_voxels < 65535, "spconv fp16 training only support this"
        from apex import amp
        net, amp_optimizer = amp.initialize(net, fastai_optimizer,
                                        opt_level="O2",
                                        keep_batchnorm_fp32=True,
                                        loss_scale=loss_scale
                                        )
        net.metrics_to_float()
    else:
        amp_optimizer = fastai_optimizer
    # Try to restore the latest checkpoints for the optimizer (if any)
    torchplus.train.try_restore_latest_checkpoints(model_dir,
                                                   [fastai_optimizer])
    # Initialize the learning rate scheduler
    lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, amp_optimizer,
                                              train_cfg.steps)
    # Determine the data type for tensors based on whether mixed precision is configured
    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    # Prepare for multi-gpu training by setting the appropriate collate function and number of GPUs
    if multi_gpu:
        num_gpu = torch.cuda.device_count()
        print(f"MULTI-GPU: use {num_gpu} gpu")
        collate_fn = merge_second_batch_multigpu
    else:
        collate_fn = merge_second_batch
        num_gpu = 1

    ######################
    # PREPARE INPUT
    ######################
    # Instantiate a dataset reader for training data
    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        multi_gpu=multi_gpu)
    # Instantiate a dataset reader for evaluation data
    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)

    # Create the DataLoader for the training dataset, with defined collate function and other parameters
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=input_cfg.batch_size * num_gpu,  # Adjust batch size for multiple GPUs if enabled
        shuffle=True,  # Shuffle the training data
        num_workers=input_cfg.preprocess.num_workers * num_gpu,  # Number of subprocesses for data loading
        pin_memory=False,  # Whether to use pinned memory
        collate_fn=collate_fn,  # Function to merge a list of samples into mini-batch
        worker_init_fn=_worker_init_fn,  # Custom function to initialize worker subprocesses
        drop_last=not multi_gpu)  # Drop the last incomplete batch if it's not multi-GPU training

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_input_cfg.batch_size, # only support multi-gpu train
        shuffle=False,  # Evaluation data is not shuffled
        num_workers=eval_input_cfg.preprocess.num_workers,  # Number of subprocesses for evaluation data loading
        pin_memory=False,  # Usage of pinned memory for evaluation data
        collate_fn=merge_second_batch)  # Function to merge samples for evaluation data
    ######################
    # TRAINING
    ######################
    # Initialize an object to handle simple logging for the model
    model_logging = SimpleModelLog(model_dir)
    model_logging.open()
    model_logging.log_text(proto_str + "\n", 0, tag="config")  # Write the model config to the log

    # Retrieve the starting step of the training, total number of steps, and the interval for evaluation
    start_step = net.get_global_step()
    total_step = train_cfg.steps
    t = time.time()
    steps_per_eval = train_cfg.steps_per_eval
    clear_metrics_every_epoch = train_cfg.clear_metrics_every_epoch

    # Commence the optimization process with gradient set to zero
    amp_optimizer.zero_grad()
    step_times = []  # To track the time taken for each step
    step = start_step  # Initialize current step with the starting step

    # Begin the main training loop
    try:
        while True:
            # The content within this while loop constitutes the steps within a single epoch
            if clear_metrics_every_epoch:
                net.clear_metrics() # Clear metrics after each epoch if specified

            # Iterate over the training data using the DataLoader
            for example in dataloader:
                lr_scheduler.step(net.get_global_step())  # Update the learning rate scheduler
                time_metrics = example["metrics"]  # Extract timing metrics if available
                example.pop("metrics")  # Remove metrics from the example to clean up data
                example_torch = example_convert_to_torch(example, float_dtype)  # Convert data to torch tensors

                batch_size = example["anchors"].shape[0]

                ret_dict = net_parallel(example_torch)
                cls_preds = ret_dict["cls_preds"] # Class predictions
                loss = ret_dict["loss"].mean() # Compute the mean loss
                cls_loss_reduced = ret_dict["cls_loss_reduced"].mean() # Compute the mean reduced class loss
                loc_loss_reduced = ret_dict["loc_loss_reduced"].mean() # Compute the mean reduced location loss
                cls_pos_loss = ret_dict["cls_pos_loss"].mean() # Compute the mean positive class loss
                cls_neg_loss = ret_dict["cls_neg_loss"].mean() # Compute the mean negative class loss
                loc_loss = ret_dict["loc_loss"] # Location loss
                cls_loss = ret_dict["cls_loss"] # Class loss
                
                cared = ret_dict["cared"] # Booleans indicating which indices should be considered for metrics
                labels = example_torch["labels"] # Ground-truth labels for the batch
                if train_cfg.enable_mixed_precision: # Backpropagate while managing scaling for mixed precision
                    with amp.scale_loss(loss, amp_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else: # Backpropagate normally
                    loss.backward()
                # Clip gradients to a maximum norm of 10 to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                amp_optimizer.step() # Optimizer step to update weights
                amp_optimizer.zero_grad() # Reset gradients to zero
                net.update_global_step() # Increment the network's global step count
                # Update model metrics with the reduced losses and predictions
                net_metrics = net.update_metrics(cls_loss_reduced,
                                                 loc_loss_reduced, cls_preds,
                                                 labels, cared)

                step_time = (time.time() - t) # Calculate the time taken for the step
                step_times.append(step_time) # Append to the list of step times
                t = time.time() # Reset the clock

                # Collect various metrics from the current training step
                metrics = {}
                num_pos = int((labels > 0)[0].float().sum().cpu().numpy()) # Number of positive labels
                num_neg = int((labels == 0)[0].float().sum().cpu().numpy()) # Number of negative labels
                if 'anchors_mask' not in example_torch:
                    num_anchors = example_torch['anchors'].shape[1]
                else:
                    num_anchors = int(example_torch['anchors_mask'][0].sum())
                global_step = net.get_global_step() # Current global step

                # Log metrics to the console and file if the step is at the specified display interval
                if global_step % display_step == 0:
                    if measure_time:
                        for name, val in net.get_avg_time_dict().items():
                            print(f"avg {name} time = {val * 1000:.3f} ms")

                    loc_loss_elem = [ # Compute location loss for each element
                        float(loc_loss[:, :, i].sum().detach().cpu().numpy() /
                              batch_size) for i in range(loc_loss.shape[-1])
                    ]
                    metrics["runtime"] = {
                        "step": global_step,
                        "steptime": np.mean(step_times),
                    }
                    metrics["runtime"].update(time_metrics[0]) # Add timing metrics to the log
                    step_times = [] # Reset the list of step times
                    metrics.update(net_metrics) # Update metrics with the network metrics
                    # Add additional loss and runtime information to the metrics
                    metrics["loss"]["loc_elem"] = loc_loss_elem
                    metrics["loss"]["cls_pos_rt"] = float(
                        cls_pos_loss.detach().cpu().numpy())
                    metrics["loss"]["cls_neg_rt"] = float(
                        cls_neg_loss.detach().cpu().numpy())
                    if model_cfg.use_direction_classifier:
                        dir_loss_reduced = ret_dict["dir_loss_reduced"].mean()
                        metrics["loss"]["dir_rt"] = float(
                            dir_loss_reduced.detach().cpu().numpy())
                    # Additional miscellaneous metrics
                    metrics["misc"] = {
                        "num_vox": int(example_torch["voxels"].shape[0]), # Number of voxels in the batch
                        "num_pos": int(num_pos), # Number of positive labels
                        "num_neg": int(num_neg), # Number of negative labels
                        "num_anchors": int(num_anchors), # Number of anchors
                        "lr": float(amp_optimizer.lr), # Learning rate
                        "mem_usage": psutil.virtual_memory().percent, # Memory usage
                    }
                    model_logging.log_metrics(metrics, global_step) # Log current metrics to the console and file

                # Save model and perform evaluation at specific steps
                if global_step % steps_per_eval == 0:
                    torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                                net.get_global_step())
                    net.eval() # Put model in evaluation mode
                    result_path_step = result_path / f"step_{net.get_global_step()}"
                    result_path_step.mkdir(parents=True, exist_ok=True)
                    model_logging.log_text("#################################",
                                        global_step)
                    model_logging.log_text("# EVAL", global_step)
                    model_logging.log_text("#################################",
                                        global_step)
                    model_logging.log_text("Generate output labels...", global_step)

                    t = time.time() # Reset the clock
                    detections = [] # List to store the detection results
                    prog_bar = ProgressBar()
                    net.clear_timer()
                    prog_bar.start((len(eval_dataset) + eval_input_cfg.batch_size - 1)
                                // eval_input_cfg.batch_size)

                    # Process each batch of evaluation data
                    for example in iter(eval_dataloader):
                        example = example_convert_to_torch(example, float_dtype)  # Convert to Torch tensors
                        detections += net(example) # Append detections to list
                        prog_bar.print_bar()

                    sec_per_ex = len(eval_dataset) / (time.time() - t) # Calculate time per example
                    model_logging.log_text(
                        f'generate label finished({sec_per_ex:.2f}/s). start eval:',
                        global_step)
                    # Evaluate and log the results
                    result_dict = eval_dataset.dataset.evaluation(
                        detections, str(result_path_step))
                    for k, v in result_dict["results"].items():
                        model_logging.log_text("Evaluation {}".format(k), global_step)
                        model_logging.log_text(v, global_step)
                    model_logging.log_metrics(result_dict["detail"], global_step)
                    # Save the detections to a file
                    with open(result_path_step / "result.pkl", 'wb') as f:
                        pickle.dump(detections, f)

                    net.train() # Put model back in training mode
                step += 1 # Increment step counter
                if step >= total_step: # If reached total steps, exit loop
                    break
            if step >= total_step: # Additional check to ensure training doesn't exceed total steps
                break

    # Handle exceptions during training, log them, and save the current state
    except Exception as e:
        print(json.dumps(example["metadata"], indent=2))
        model_logging.log_text(str(e), step)
        model_logging.log_text(json.dumps(example["metadata"], indent=2), step)
        torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                    step)
        raise e
    # Ensure logging is tidied up upon exiting the training loop
    finally:
        model_logging.close()
    # Save final model state at the end of training
    torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                net.get_global_step())


def evaluate(config_path,
             model_dir=None,
             result_path=None,
             ckpt_path=None,
             measure_time=False,
             batch_size=None,
             **kwargs):
    """Don't support pickle_result anymore. if you want to generate kitti label file,
    please use kitti_anno_to_label_file and convert_detection_to_kitti_annos
    in second.data.kitti_dataset.
    """
    assert len(kwargs) == 0
    model_dir = str(Path(model_dir).resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_name = 'eval_results'
    if result_path is None:
        model_dir = Path(model_dir)
        result_path = model_dir / result_name
    else:
        result_path = Path(result_path)
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to eval with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    net = build_network(model_cfg, measure_time=measure_time).to(device)
    if train_cfg.enable_mixed_precision:
        net.half()
        print("half inference!")
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator

    if ckpt_path is None:
        assert model_dir is not None
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)
    batch_size = batch_size or input_cfg.batch_size
    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    net.eval()
    result_path_step = result_path / f"step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)
    t = time.time()
    detections = []
    print("Generate output labels...")
    bar = ProgressBar()
    bar.start((len(eval_dataset) + batch_size - 1) // batch_size)
    prep_example_times = []
    prep_times = []
    t2 = time.time()

    for example in iter(eval_dataloader):
        if measure_time:
            prep_times.append(time.time() - t2)
            torch.cuda.synchronize()
            t1 = time.time()
        example = example_convert_to_torch(example, float_dtype)
        if measure_time:
            torch.cuda.synchronize()
            prep_example_times.append(time.time() - t1)
        with torch.no_grad():
            detections += net(example)
        bar.print_bar()
        if measure_time:
            t2 = time.time()

    sec_per_example = len(eval_dataset) / (time.time() - t)
    print(f'generate label finished({sec_per_example:.2f}/s). start eval:')
    if measure_time:
        print(
            f"avg example to torch time: {np.mean(prep_example_times) * 1000:.3f} ms"
        )
        print(f"avg prep time: {np.mean(prep_times) * 1000:.3f} ms")
    for name, val in net.get_avg_time_dict().items():
        print(f"avg {name} time = {val * 1000:.3f} ms")
    with open(result_path_step / "result.pkl", 'wb') as f:
        pickle.dump(detections, f)
    result_dict = eval_dataset.dataset.evaluation(detections,
                                                  str(result_path_step))
    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print("Evaluation {}".format(k))
            print(v)

def helper_tune_target_assigner(config_path, target_rate=None, update_freq=200, update_delta=0.01, num_tune_epoch=5):
    """get information of target assign to tune thresholds in anchor generator.
    """    
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to train with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path
        proto_str = text_format.MessageToString(config, indent=2)

    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    net = build_network(model_cfg, False)
    # if train_cfg.enable_mixed_precision:
    #     net.half()
    #     net.metrics_to_float()
    #     net.convert_norm_to_float(net)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        multi_gpu=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=merge_second_batch,
        worker_init_fn=_worker_init_fn,
        drop_last=False)
    
    class_count = {}
    anchor_count = {}
    class_count_tune = {}
    anchor_count_tune = {}
    for c in target_assigner.classes:
        class_count[c] = 0
        anchor_count[c] = 0
        class_count_tune[c] = 0
        anchor_count_tune[c] = 0


    step = 0
    classes = target_assigner.classes
    if target_rate is None:
        num_tune_epoch = 0
    for epoch in range(num_tune_epoch):
        for example in dataloader:
            gt_names = example["gt_names"]
            for name in gt_names:
                class_count_tune[name] += 1
            
            labels = example['labels']
            for i in range(1, len(classes) + 1):
                anchor_count_tune[classes[i - 1]] += int(np.sum(labels == i))
            if target_rate is not None:
                for name, rate in target_rate.items():
                    if class_count_tune[name] > update_freq:
                        # calc rate
                        current_rate = anchor_count_tune[name] / class_count_tune[name]
                        if current_rate > rate:
                            target_assigner._anchor_generators[classes.index(name)].match_threshold += update_delta
                            target_assigner._anchor_generators[classes.index(name)].unmatch_threshold += update_delta
                        else:
                            target_assigner._anchor_generators[classes.index(name)].match_threshold -= update_delta
                            target_assigner._anchor_generators[classes.index(name)].unmatch_threshold -= update_delta
                        anchor_count_tune[name] = 0
                        class_count_tune[name] = 0
            step += 1
    for c in target_assigner.classes:
        class_count[c] = 0
        anchor_count[c] = 0
    total_voxel_gene_time = 0
    count = 0

    for example in dataloader:
        gt_names = example["gt_names"]
        total_voxel_gene_time += example["metrics"][0]["voxel_gene_time"]
        count += 1

        for name in gt_names:
            class_count[name] += 1
        
        labels = example['labels']
        for i in range(1, len(classes) + 1):
            anchor_count[classes[i - 1]] += int(np.sum(labels == i))
    print("avg voxel gene time", total_voxel_gene_time / count)

    print(json.dumps(class_count, indent=2))
    print(json.dumps(anchor_count, indent=2))
    if target_rate is not None:
        for ag in target_assigner._anchor_generators:
            if ag.class_name in target_rate:
                print(ag.class_name, ag.match_threshold, ag.unmatch_threshold)

def mcnms_parameters_search(config_path,
          model_dir,
          preds_path):
    pass


if __name__ == '__main__':
    fire.Fire()
