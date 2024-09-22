from pyrover_domain.algorithms.ccea import runCCEA
import torch.multiprocessing as mp

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
    parser.add_argument("--fitness_critic_model", default="mlp", help="mlp/att", type=str)

    args = vars(parser.parse_args())

    # Set base_config path
    config_dir = (
        "/nfs/hpc/share/agrazvam/experiments/yamls"
        if args["hpc"]
        else "/home/magraz/rovers/pyrover_domain/experiments/yamls"
    )

    teaming = "_teaming" if args["teaming"] else ""

    fit_crit = "_fit_crit" if args["fitness_critic"] else ""

    fit_crit_model = f"_{args['fitness_critic_model']}" if len(args["fitness_critic_model"]) > 0 else ""

    # Set configuration file
    config_dir = f"{config_dir}/{args['poi_type']}_{args['model']}{teaming}{fit_crit}{fit_crit_model}.yaml"

    # config_dir = f"{config_dir}/static_mlp_fit_crit.yaml"

    # Run learning algorithm
    runCCEA(config_dir=config_dir)
