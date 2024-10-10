from pyrover_domain.algorithms.ccea import runCCEA
import torch.multiprocessing as mp
import os

import argparse

if __name__ == "__main__":
    # Set for CUDA to allow multiprocessing
    mp.set_start_method("forkserver")

    # Arg parser variables
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpc", default=False, help="use hpc config files", action="store_true")
    parser.add_argument("--load_checkpoint", default=False, help="loads checkpoint", action="store_true")
    parser.add_argument("--poi_type", default="static", help="static/decay", type=str)
    parser.add_argument("--model", default="mlp", help="mlp/gru/cnn/att", type=str)
    parser.add_argument("--experiment_type", default="", help="standard/fitness_critic/teaming", type=str)
    parser.add_argument("--trial_id", default=0, help="Sets trial ID", type=int)

    args = vars(parser.parse_args())

    # Set base_config path
    dir_path = os.path.dirname(os.path.realpath(__file__))

    if args["hpc"]:
        dir_path = "/nfs/hpc/share/agrazvam"

    config_dir = os.path.join(dir_path, "experiments", "yamls", args["experiment_type"])

    # Set configuration file
    experiment_name = "_".join((args["experiment_type"], args["poi_type"], args["model"]))
    yaml_filename = "_".join((args["poi_type"], args["model"])) + ".yaml"
    config_dir = os.path.join(config_dir, yaml_filename)

    # Run learning algorithm
    runCCEA(config_dir=config_dir, experiment_name=experiment_name, trial_id=args["trial_id"], load_checkpoint=args["load_checkpoint"])
