from retro_branching.utils import check_if_network_params_equal, seed_stochastic_modules_globally
from retro_branching.networks import BipartiteGCN
from retro_branching.agents import DQNAgent
from retro_branching.environments import EcoleBranching
from retro_branching.learners import DQNLearner
from retro_branching.utils import generate_craballoc

import ecole
import torch

import random
from tqdm import trange

import os
import argparse

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import shutil
from pathlib import Path

hydra.HYDRA_FULL_ERROR = 1


def init_save_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


@hydra.main(config_path='configs', config_name='config.yaml')
def run(cfg: DictConfig):
    # seeding
    if 'seed'in cfg.experiment:
        seed = cfg.experiment['seed']
    else:
        seed = random.randint(0, 10000)
    seed_stochastic_modules_globally(seed)

    # print info
    print('\n\n\n')
    print(f'~' * 80)
    print(f'Config:\n{OmegaConf.to_yaml(cfg)}')
    print(f'~' * 80)

    # initialise instance generator
    if 'path_to_instances' in cfg.instances:
        instances = ecole.instance.FileGenerator(cfg.instances.path_to_instances,
                                                 sampling_mode=cfg.instances.sampling_mode)
    else:
        if cfg.instances.co_class == 'set_covering':
            instances = ecole.instance.SetCoverGenerator(**cfg.instances.co_class_kwargs)
        elif cfg.instances.co_class == 'combinatorial_auction':
            instances = ecole.instance.CombinatorialAuctionGenerator(**cfg.instances.co_class_kwargs)
        elif cfg.instances.co_class == 'capacitated_facility_location':
            instances = ecole.instance.CapacitatedFacilityLocationGenerator(**cfg.instances.co_class_kwargs)
        elif cfg.instances.co_class == 'maximum_independent_set':
            instances = ecole.instance.IndependentSetGenerator(**cfg.instances.co_class_kwargs)
        elif cfg.instances.co_class == 'crabs':
            instances = generate_craballoc(**cfg.instances.co_class_kwargs)
        else:
            raise Exception(f'Unrecognised co_class {cfg.instances.co_class}')
    print(f'Initialised instance generator.')

    # initialise branch-and-bound environment
    env = EcoleBranching(observation_function=cfg.environment.observation_function,
                         information_function=cfg.environment.information_function,
                         reward_function=cfg.environment.reward_function,
                         scip_params=cfg.environment.scip_params)
    print(f'Initialised environment.')
    
    init_save_dir(cfg.experiment.path_to_load_instances)
    # data generation
    for i in trange(55, 100):
        done = True
        while done:
            instance = next(instances)
            obs, act, rew, done, info = env.reset(instance.copy_orig())
        instance.write_problem(
            f'{cfg.experiment.path_to_load_instances}/'
            f'/{i}.mps'
        )


if __name__ == '__main__':
    run()
