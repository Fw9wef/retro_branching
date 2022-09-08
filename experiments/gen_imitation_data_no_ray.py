from retro_branching.utils import gen_co_name, ExploreThenStrongBranch, PureStrongBranch, seed_stochastic_modules_globally
from retro_branching.scip_params import gasse_2019_scip_params, default_scip_params

import ecole

import gzip
import pickle
import numpy as np
from pathlib import Path
import time
import os
import glob
import random

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import shutil
hydra.HYDRA_FULL_ERROR = 1

import threading
from threading import Thread
import queue
from queue import Queue
from tqdm import trange

n_parallel_process = 32


def run_sampler(path, sample_n_queue, co_class, co_class_kwargs, branching, max_steps=None):
    '''
    Args:
        branching (str): Branching scheme to use. Must be one of 'explore_then_strong_branch',
            'pure_strong_branch'
        max_steps (None, int): If not None, will terminate episode after max_steps.
    '''
    # N.B. Need to init instances and env here since ecole objects are not
    # serialisable and ray requires all args passed to it to be serialisable
    if co_class == 'set_covering':
        instance_gen = ecole.instance.SetCoverGenerator(**co_class_kwargs)
    elif co_class == 'combinatorial_auction':
        instance_gen = ecole.instance.CombinatorialAuctionGenerator(**co_class_kwargs)
    elif co_class == 'capacitated_facility_location':
        instance_gen = ecole.instance.CapacitatedFacilityLocationGenerator(**co_class_kwargs)
    elif co_class == 'maximum_independent_set':
        instance_gen = ecole.instance.IndependentSetGenerator(**co_class_kwargs)
    else:
        raise Exception(f'Unrecognised co_class {co_class}')

    # scip_params = default_scip_params
    scip_params = gasse_2019_scip_params

    if branching == 'explore_then_strong_branch':
        env = ecole.environment.Branching(observation_function=(ExploreThenStrongBranch(expert_probability=0.05),
                                                                ecole.observation.NodeBipartite()),
                                          scip_params=scip_params)
    elif branching == 'pure_strong_branch':
        env = ecole.environment.Branching(observation_function=(PureStrongBranch(),
                                                                ecole.observation.NodeBipartite()),
                                          scip_params=scip_params)
    else:
        raise Exception('Unrecognised branching {}'.format(branching))

    n = sample_n_queue.get(timeout=5)

    while True:
        instance = next(instance_gen)
        observation, action_set, _, done, _ = env.reset(instance)
        t = 0
        while not done:
            if branching == 'explore_then_strong_branch':
                # only save samples if they are coming from the expert (strong branching)
                (scores, save_samples), node_observation = observation
            elif branching == 'pure_strong_branch':
                # always save samples since always using strong branching
                save_samples = True
                scores, node_observation = observation
            else:
                raise Exception('Unrecognised branching {}'.format(branching))

            action = action_set[scores[action_set].argmax()]

            if save_samples:
                data = [node_observation, action, action_set, scores]
                filename = f'{path}sample_{n}.pkl'
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(data, f)
                sample_n_queue.task_done()

                try:
                    n = sample_n_queue.get(timeout=5)
                except queue.Empty:
                    curr_thread = threading.current_thread()
                    print(f'thread {curr_thread} finished')
                    return

            observation, action_set, _, done, _ = env.step(action)
            t += 1
            if max_steps is not None:
                if t >= max_steps:
                    # stop episode
                    break

def init_save_dir(path, name):
    _path = path + name + '/'
    counter = 1
    foldername = '{}_{}/'
    while os.path.isdir(_path+foldername.format(name, counter)):
        counter += 1
    foldername = foldername.format(name, counter)
    Path(_path+foldername).mkdir(parents=True, exist_ok=True)
    return _path+foldername


@hydra.main(config_path='configs', config_name='config.yaml')
def run(cfg: DictConfig):
    # seeding
    #if 'seed' not in cfg.experiment:
    cfg.experiment['seed'] = random.randint(0, 10000)
    seed_stochastic_modules_globally(cfg.experiment.seed)

    # print info
    print('\n\n\n')
    print(f'~'*80)
    print(f'Config:\n{OmegaConf.to_yaml(cfg)}')
    print(f'~'*80)

    #path = cfg.experiment.path_to_save + f'/{cfg.experiment.branching}/{cfg.instances.co_class}/max_steps_{cfg.experiment.max_steps}/{gen_co_name(cfg.instances.co_class, cfg.instances.co_class_kwargs)}/'
    path = '/home/al/prjs/retro_branching/outputs/' + f'/{cfg.experiment.branching}/{cfg.instances.co_class}/max_steps_{cfg.experiment.max_steps}/{gen_co_name(cfg.instances.co_class, cfg.instances.co_class_kwargs)}/'
    print(path)
    path = init_save_dir(path, 'samples')
    print(path)
    print('Generating >={} samples in parallel on {} CPUs and saving to {}'.format(cfg.experiment.min_samples, n_parallel_process, os.path.abspath(path)))

    path = "/home/al/prjs/retro_branching/outputs/explore_then_strong_branch/capacitated_facility_location/max_steps_None/capacitated_facility_location_n_customers_15_n_facilities_15/samples/samples_1/"
    already_done = max(len(os.listdir(os.path.abspath(path))) - 50, 0)

    #ecole.seed(cfg.experiment.seed)
    ecole.seed(already_done)
    sample_n_queue = Queue(maxsize=64)

    threads = list()
    for _ in range(n_parallel_process):
        process = Thread(target=run_sampler, args=(path, sample_n_queue,
                                                   cfg.instances.co_class,
                                                   cfg.instances.co_class_kwargs,
                                                   cfg.experiment.branching,
                                                   cfg.experiment.max_steps))
        process.start()
        threads.append(process)

    for i in trange(already_done, cfg.experiment.min_samples):
        sample_n_queue.put(i)
        print(f'{i} of {cfg.experiment.min_samples} index queued up')

    for process in threads:
        process.join()
    sample_n_queue.join()

if __name__ == '__main__':
    run()