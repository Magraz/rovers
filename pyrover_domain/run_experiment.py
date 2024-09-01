from pyrover_domain.ccea.ccea_lib import runCCEA
import torch.multiprocessing as mp
from torch import set_grad_enabled

import argparse


if __name__ == "__main__":
    # Set for CUDA to allow multiprocessing
    mp.set_start_method("forkserver")

    # Disable gradient calculations
    set_grad_enabled(False)

    # Arg parser variables
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpc", help="use hpc config files", action="store_true")
    args = parser.parse_args()

    # Set base_config path
    config_dir = "/home/magraz/rovers/pyrover_domain/experiments/yamls"
    if args.hpc:
        config_dir = "/nfs/hpc/share/agrazvam/experiments/yamls"

    # Set configuration file
    config_dir = f"{config_dir}/multi_agent_regular_mlp_teaming.yaml"
    # config_dir = "/home/magraz/rovers/pyrover_domain/config/single_agent_decay.yaml"
    # config_dir = "/home/magraz/rovers/pyrover_domain/config/multi_agent_decay_mlp.yaml"
    # config_dir = "/home/magraz/rovers/pyrover_domain/config/multi_agent_decay_cnn.yaml"
    # config_dir = "/home/magraz/rovers/pyrover_domain/config/multi_agent_decay_gru.yaml"

    # Run learning algorithm
    runCCEA(config_dir=config_dir)
