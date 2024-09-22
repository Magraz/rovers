from pyrover_domain.algorithms.ccea import runCCEA
import torch.multiprocessing as mp
from torch import set_grad_enabled

import argparse


if __name__ == "__main__":
    # Set for CUDA to allow multiprocessing
    mp.set_start_method("forkserver")

    # Arg parser variables
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpc", default=False, help="use hpc config files", action="store_true")
    parser.add_argument("--teaming", default=False, help="use teaming", action="store_true")
    parser.add_argument("--fitness_critic", default=False, help="use fitness_critic", action="store_true")
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
    fitness_critic = "_fit_crit" if args["fitness_critic"] else ""

    # Set configuration file
    config_dir = f"{config_dir}/{args['poi_type']}_{args['model']}{teaming}{fitness_critic}.yaml"

    # config_dir = f"{config_dir}/order_mlp.yaml"

    # Run learning algorithm
    runCCEA(config_dir=config_dir)
