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
from itertools import product


#@hydra.main(config_path="configs", config_name="config")
def make_predictions(extraction_method, dataset, run, model, seed):
    """this script will work with a path to a trained model's hydra folder
    from that folder it'll read the info about the model, get the checkpoint, and predict on a new vid"""
    """note, by decorating with hydra, the current working directory will be become the new folder os.path.join(os.getcwd(), "/outputs/YYYY-MM-DD/hour-info")"""
    # TODO: supporting only the zeroth index of cfg.eval.path_to_test_videos[0]
    # go to folders up to the "outputs" folder, and search for hydra_path from cfg

    absolute_cfg_path = f'outputs/{extraction_method}_{dataset}_{run}_{model}_{seed}/'
    
    model_cfg = OmegaConf.load(
        os.path.join(absolute_cfg_path, ".hydra/config.yaml")
    )

    ckpt_file = ckpt_path_from_base_path(
        base_path=absolute_cfg_path, model_name=model_cfg.model.model_name
    )
    
    datasets_path = '/home/eivinas/dev/dlc-frame-selection/datasets/'
    test_video_dir = f'{datasets_path}/{dataset}/test_video'
    
    predictions_csv_dir = '/home/eivinas/dev/dlc-frame-selection/predictions/csv'
    save_dir = f'{predictions_csv_dir}/{extraction_method}_{dataset}_{run}_{model}_{seed}'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    predict_videos(
        video_dir=test_video_dir,
        ckpt_file=ckpt_file,
        cfg_file=model_cfg,
        save_dir=save_dir,
        sequence_length=64,
    )


if __name__ == "__main__":
    #extraction_methods = ['uniform', 'kmeans', 'umap']
    extraction_methods = ['umap']
    dataset = 'mouse_wheel'
    n_runs = 1
    models = [50]
    n_seeds = 1

    runs = range(1, n_runs+1)
    seeds = range(1, n_seeds+1)
    seeds = [3]
    
    rmses = {}
    combs = product(extraction_methods, runs, models, seeds)
    for comb in combs:
        (e, r, m, s) = comb
        make_predictions(e, dataset, r, m, s)
