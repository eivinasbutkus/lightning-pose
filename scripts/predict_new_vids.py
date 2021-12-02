import glob
import os
import numpy as np
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from pose_est_nets.utils.plotting_utils import (
    predict_videos,
)
from pose_est_nets.utils.io import (
    get_absolute_hydra_path_from_hydra_str,
    ckpt_path_from_base_path,
    verify_absolute_path,
)


import argparse
from pathlib import Path



#@hydra.main(config_path="configs", config_name="config")
def make_predictions(extraction_method, dataset, run, model, seed):
    """this script will work with a path to a trained model's hydra folder
    from that folder it'll read the info about the model, get the checkpoint, and predict on a new vid"""
    """note, by decorating with hydra, the current working directory will be become the new folder os.path.join(os.getcwd(), "/outputs/YYYY-MM-DD/hour-info")"""
    # TODO: supporting only the zeroth index of cfg.eval.path_to_test_videos[0]
    # go to folders up to the "outputs" folder, and search for hydra_path from cfg

    #absolute_cfg_path = get_absolute_hydra_path_from_hydra_str(hydra_relative_path)

    absolute_cfg_path = f'outputs/{extraction_method}_{dataset}_{run}_{model}_{seed}/'
    # extraction_method = cfg.data.extraction_method
    # dataset = cfg.data.dataset
    # run = cfg.data.run
    # model = cfg.model.resnet_version
    # seed = cfg.training.rng_seed_model_pt

    #absolute_cfg_path = _get_absolute_cfg_path(hydra_relative_path)
    
    model_cfg = OmegaConf.load(
        os.path.join(absolute_cfg_path, ".hydra/config.yaml")
    )

    ckpt_file = ckpt_path_from_base_path(
        base_path=absolute_cfg_path, model_name=model_cfg.model.model_name
    )
    
    datasets_path = '/home/eivinas/dev/dlc-frame-selection/datasets/'
    test_video_path = f'{datasets_path}/{dataset}/test_video'
    Path(test_video_path, 'predictions').mkdir(parents=True, exist_ok=True)
    predictions_path = f'{datasets_path}/{dataset}/test_video/predictions/{extraction_method}_{dataset}_{run}_{model}_{seed}.csv'

    #absolute_path_to_test_videos = verify_absolute_path(test_video_path)

    predict_videos(
        video_path=test_video_path,
        ckpt_file=ckpt_file,
        cfg_file=model_cfg,
        save_file=predictions_path,
        sequence_length=64,
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--extraction-method', type=str)
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-r', '--run', type=int)
    parser.add_argument('-m', '--model', type=int)
    parser.add_argument('-s', '--seed', type=int)

    args = parser.parse_args()
    make_predictions(args.extraction_method, args.dataset, args.run, args.model, args.seed)
