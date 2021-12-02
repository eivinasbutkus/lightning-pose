
"""
Creates a text file with all the different job commands
and submits that as a slurm job array
"""

from pathlib import Path
from itertools import product

#extraction_methods = ['uniform', 'kmeans', 'umap']
extraction_methods = ['umap']
datasets = ['mouse_wheel']
n_runs = 1
models = [50] # resnet 50 only
n_seeds = 5


def main():
    slurm_jobs_file_path = Path("slurm_jobs.txt")
    slurm_jobs_file = open(slurm_jobs_file_path, "w")
    
    runs = range(1, n_runs+1)
    seeds = range(1, n_seeds+1)

    for (e, d, r, m, s) in product(extraction_methods, datasets, runs, models, seeds):
        output_dir = f'{e}_{d}_{r}_{m}_{s}'
        cmd = f'conda activate lp && '
        cmd += f'python scripts/train_hydra.py data=data_{d} data.extraction_method={e} data.run={r} '
        cmd += f'training.rng_seed_model_pt={s} model.resnet_version={m} hydra.run.dir={output_dir}\n'        
        slurm_jobs_file.write(cmd)

    slurm_jobs_file.close()
    

if __name__ == '__main__':
    main()
