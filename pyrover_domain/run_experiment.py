from pyrover_domain.ccea.ccea_lib import runCCEA
import torch.multiprocessing as mp

if __name__ == '__main__':
    #Set for CUDA to allow multiprocessing
    mp.set_start_method('forkserver')

    #Set configuration file
    config_dir = "/home/magraz/rovers/pyrover_domain/config/experiment_1.yaml"

    #Run learning algorithm
    runCCEA(config_dir=config_dir)
