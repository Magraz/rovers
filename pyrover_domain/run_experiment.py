from pyrover_domain.ccea.ccea_lib import runCCEA
import torch.multiprocessing as mp
from torch import set_grad_enabled

if __name__ == "__main__":
    # Set for CUDA to allow multiprocessing
    mp.set_start_method("forkserver")

    # Disable gradient calculations
    set_grad_enabled(False)

    # Set configuration file
    config_dir = "/home/magraz/rovers/pyrover_domain/config/multi_agent_regular_mlp.yaml"
    # config_dir = "/home/magraz/rovers/pyrover_domain/config/single_agent_decay.yaml"
    # config_dir = "/home/magraz/rovers/pyrover_domain/config/multi_agent_decay_mlp.yaml"
    # config_dir = "/home/magraz/rovers/pyrover_domain/config/multi_agent_decay_cnn.yaml"
    # config_dir = "/home/magraz/rovers/pyrover_domain/config/multi_agent_decay_gru.yaml"

    # Run learning algorithm
    runCCEA(config_dir=config_dir)
