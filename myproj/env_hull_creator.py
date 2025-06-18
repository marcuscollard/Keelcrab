# Generic Imports
import argparse
import random
import numpy as np
import sys
import os, glob

# for ship
import tempfile
from ShipD.HullParameterization import Hull_Parameterization as HP

# config
import hydra
from omegaconf import DictConfig, OmegaConf

from isaaclab.app import AppLauncher
import argparse

# Optional CLI args
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
# 4. Override default values BEFORE parsing
default_args = [
    "--headless"
]

args = parser.parse_args(default_args)


args = parser.parse_args()

# Launch Isaac Sim headlessly and without UI
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def create_hulls(cfg: DictConfig):
    #set random for numpy - do for torch too!
    np.random.seed(cfg.random.seed)

    # load the parameters csv
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(SCRIPT_DIR, cfg.ships.filepath)

    print(csv_path)

    vectors = np.array([])

    if os.path.isfile(csv_path):
        try:
            vectors = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
        except Exception as e:
            print(f"Failed to load vectors: {e}")
            vectors = np.empty((0,))
    else:
        print('not valid csv file')


    # randomly select N from valid indices
    size = vectors.shape[0] if cfg.ships.limit is None else cfg.ships.limit
    
    chosen_indices = np.random.choice(np.arange(size), 
                                      size=4, replace=False)

    vecs = list(vectors[chosen_indices])
    for i, vec in enumerate(vecs):
        hull = HP(vec)
        base = os.path.join("myproj/temp/", f'hull_{i}')
        hull.gen_USD(
            NUM_WL=50,
            PointsPerWL=300,
            bit_AddTransom=1,
            bit_AddDeckLid=1,
            bit_RefineBowAndStern=0,
            namepath=base
        )
        usd_file = base + '.usd'
        if not os.path.isfile(usd_file):
            print(f"USD generation failed: {usd_file} not found")
            return
        
        
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    
    # create hulls
    create_hulls(cfg)

    # exit the script
    sys.exit(0)


if __name__ == "__main__":

    # Run the main function with the loaded configuration
    main()