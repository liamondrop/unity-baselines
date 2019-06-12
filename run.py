import os
import sys
import multiprocessing
import numpy as np
import random

from baselines.common.cmd_util import common_arg_parser
from baselines.common.vec_env import VecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.run import parse_cmdline_kwargs
from baselines.run import get_env_type, get_learn_function, get_learn_function_defaults

from gym_unity.envs import UnityEnv

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def train(args, extra_args):
    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = {}
    alg_kwargs.update(extra_args)

    env_type = 'unity'
    env = make_vec_env(args.env,
                       args.num_env or 1,
                       args.seed,
                       reward_scale=args.reward_scale,
                       wrapper_kwargs=extra_args)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = 'mlp'

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, args.env, alg_kwargs))
    print(args)

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env

def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i  + 1000 * rank if i is not None else None
    try:
        import tensorflow as tf
        tf.set_random_seed(myseed)
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)

def make_vec_env(env_filename, num_env, seed, wrapper_kwargs=None, start_index=0, reward_scale=1.0):
    if wrapper_kwargs is None: wrapper_kwargs = {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            worker_id = 32 + rank
            env = UnityEnv(env_filename, worker_id, **wrapper_kwargs)
            env.seed(seed + 10000*mpi_rank + rank if seed is not None else None)
            return env
        return _thunk
    set_global_seeds(seed)
    if num_env > 1: return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
    else: return DummyVecEnv([make_env(start_index)])

def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    rank = 0
    model, env = train(args, extra_args)

    if args.save_path is not None and rank == 0:
        save_path = os.path.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        episode_rew = 0
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions)
            episode_rew += rew[0] if isinstance(env, VecEnv) else rew
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done
            if done:
                print('episode_rew={}'.format(episode_rew))
                episode_rew = 0
                obs = env.reset()

    env.close()
    return model


if __name__ == '__main__':
    main(sys.argv)