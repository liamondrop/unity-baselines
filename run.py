import os
import sys
import datetime
import multiprocessing
import numpy as np
import pprint
import random
import threading
import time
import yaml
import tensorflow as tf

from baselines import logger
from baselines.bench import Monitor
from baselines.common.cmd_util import common_arg_parser
from baselines.common.tf_util import get_session
from baselines.common.vec_env import VecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.run import parse_cmdline_kwargs, configure_logger
from baselines.run import get_learn_function, get_learn_function_defaults

from gym_unity.envs import UnityEnv

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0
    myseed = i + 1000 * rank if i is not None else None
    tf.set_random_seed(myseed)
    np.random.seed(myseed)
    random.seed(myseed)


def train(args, extra_args):
    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)

    wrapper_kwargs = dict(
        use_visual=False,
        uint8_visual=True,
        multiagent=False,
        flatten_branched=False,
        no_graphics=False,
        allow_multiple_visual_obs=False,)
    for k in wrapper_kwargs:
        if extra_args.get(k) is not None:
            wrapper_kwargs[k] = extra_args[k]
            del extra_args[k]

    alg_kwargs = {}
    alg_kwargs.update(extra_args)

    env = make_vec_env(args.env,
                       args.num_env or 1,
                       args.seed,
                       reward_scale=args.reward_scale,
                       wrapper_kwargs=wrapper_kwargs)

    if args.save_video_interval != 0:
        env = VecVideoRecorder(env,
            os.path.join(logger.get_dir(), 'videos'),
            record_video_trigger=lambda x: x % args.save_video_interval == 0,
            video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = 'mlp'

    print('Training {} in unity_env:{} with arguments:'.format(args.alg, args.env))
    pprint.pprint(alg_kwargs)

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def make_vec_env(env_filename, num_env, seed,
                 wrapper_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 subrank=0):
    if wrapper_kwargs is None: wrapper_kwargs = {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    logger_dir = logger.get_dir()
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            worker_id = 32 + rank
            env = UnityEnv(env_filename, worker_id, **wrapper_kwargs)
            env.seed(seed + 10000*mpi_rank + rank if seed is not None else None)
            env = Monitor(env, logger_dir and os.path.join(
                            logger_dir, str(rank)),
                          allow_early_resets=True)
            return env
        return _thunk
    set_global_seeds(seed)
    return DummyVecEnv([make_env(start_index)])


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    rank = 0
    log_path = os.path.join(args.log_path if args.log_path is not None else 'log',
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    configure_logger(log_path, format_strs=['stdout', 'csv'])

    model, env = train(args, extra_args)

    if args.save_path is not None and rank == 0:
        save_path = os.path.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
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

if __name__ == '__main__':
    argv = sys.argv.copy()
    with open(argv.pop(1), 'r') as stream:
        conf = yaml.safe_load(stream)

    assert 'env' in conf, 'The path to the Unity "env" is required'

    for key, val in conf.items():
        argv.extend(['--%s' % key, str(val)])
    main(argv)
