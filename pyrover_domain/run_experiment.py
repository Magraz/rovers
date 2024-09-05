from pyrover_domain.algorithms.ccea import runCCEA
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
    parser.add_argument("--hpc", default=False, help="use hpc config files", action="store_true")
    parser.add_argument("--teaming", default=False, help="use teaming", action="store_true")
    parser.add_argument("--modality", default="multi", help="multi/single", type=str)
    parser.add_argument("--poi_type", default="static", help="static/decay", type=str)
    parser.add_argument("--model", default="mlp", help="mlp/gru/cnn", type=str)

    args = vars(parser.parse_args())

    # Set base_config path
    config_dir = (
        "/nfs/hpc/share/agrazvam/experiments/yamls"
        if args["hpc"]
        else "/home/magraz/rovers/pyrover_domain/experiments/yamls"
    )

    teaming = "_teaming" if args["teaming"] else ""

    # Set configuration file
    config_dir = f"{config_dir}/{args['modality']}_{args['poi_type']}_{args['model']}{teaming}.yaml"

    # Run learning algorithm
    runCCEA(config_dir=config_dir)
