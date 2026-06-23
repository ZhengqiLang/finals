import argparse
import copy
import os
from datetime import datetime
from pathlib import Path

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint
import torch
from flax.training import orbax_utils
from sb3_contrib import ARS, TQC, TRPO
from stable_baselines3 import PPO, TD3, SAC, A2C, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.sac.policies import SACPolicy

import models
from core.jax_utils import MLP, create_train_state, orbax_set_config
from core.plot import plot_traces

gym.register(
    id='LinearSystem',
    entry_point='models.linearsystem:LinearSystem',
    max_episode_steps=250
)

gym.register(
    id='MyPendulum',
    entry_point='models.pendulum:Pendulum',
    max_episode_steps=500
)

gym.register(
    id='CollisionAvoidance',
    entry_point='models.collision_avoidance:CollisionAvoidance',
    max_episode_steps=250
)

gym.register(
    id='MyMountainCar',
    entry_point='models.mountain_car:MountainCar',
    max_episode_steps=250
)

gym.register(
    id='PlanarRobot',
    entry_point='models.planar_robot:PlanarRobot',
    max_episode_steps=250
)

gym.register(
    id='Cartpole',
    entry_point='models.cartpole:Cartpole',
    max_episode_steps=500
)


def torch_to_jax(jax_policy_state, weights, biases):
    for i, (w, b) in enumerate(zip(weights, biases)):
        w = w.cpu().detach().numpy()
        b = b.cpu().detach().numpy()

        # Copy weights and biases from each layer from Pytorch to JAX
        jax_policy_state.params['params']["Dense_" + str(i)]['kernel'] = w.T  # Note: Transpose between torch and jax!
        jax_policy_state.params['params']["Dense_" + str(i)]['bias'] = b

    return jax_policy_state


def train_stable_baselines(vec_env, RL_method, policy_size, policy_size_jax, activation_fn_torch, activation_fn_jax,
                           total_steps, allow_tanh):
    # Create JAX policy network
    print(f'- Create JAX MLP for RL algorithm {RL_method}')
    print(f'-- Size: {policy_size_jax}')
    print(f'-- Act func: {activation_fn_jax}')

    jax_policy_model = MLP(policy_size_jax, activation_fn_jax)
    jax_policy_state = create_train_state(
        model=jax_policy_model,
        act_funcs=activation_fn_jax,
        rng=jax.random.PRNGKey(1),
        in_dim=vec_env.reset().shape[1],
        learning_rate=5e-5,
    )

    if RL_method == "PPO":
        policy_kwargs = dict(activation_fn=activation_fn_torch,
                             net_arch=policy_size)

        model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)

        # Train
        model.learn(total_timesteps=total_steps)

        # PPO Should return an actor critic policy
        assert isinstance(model.policy, ActorCriticPolicy)

        # Get weights
        weights = [model.policy.mlp_extractor.policy_net[int(i * 2)].weight for i in range(len(policy_size))]
        weights += [model.policy.action_net.weight]
        # Get biases
        biases = [model.policy.mlp_extractor.policy_net[int(i * 2)].bias for i in range(len(policy_size))]
        biases += [model.policy.action_net.bias]
        # Convert Torch to JAX model
        jax_policy_state = torch_to_jax(jax_policy_state, weights, biases)

    elif RL_method == "TD3":
        policy_kwargs = dict(activation_fn=activation_fn_torch,
                             net_arch=policy_size)

        # The noise objects for TD3
        n_actions = vec_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.0001 * np.ones(n_actions))

        model = TD3("MlpPolicy", vec_env, action_noise=action_noise, policy_kwargs=policy_kwargs, verbose=1)

        # Remove the tanh activation function, which TD3 sets by default
        if not allow_tanh:
            model.actor.mu = model.actor.mu[:-1]

        # Train
        model.learn(total_timesteps=total_steps)

        # Get weights
        weights = [model.actor.mu[int(i * 2)].weight for i in range(len(policy_size) + 1)]
        # Get biases
        biases = [model.actor.mu[int(i * 2)].bias for i in range(len(policy_size) + 1)]
        # Convert Torch to JAX model
        jax_policy_state = torch_to_jax(jax_policy_state, weights, biases)

    elif RL_method == "SAC":
        policy_kwargs = dict(activation_fn=activation_fn_torch,
                             net_arch=policy_size)

        model = SAC("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)

        # Train
        model.learn(total_timesteps=total_steps)

        # SAC Should return an SACPolicy object
        assert isinstance(model.policy, SACPolicy)

        # Get weights
        weights = [model.actor.latent_pi[int(i * 2)].weight for i in range(len(policy_size))]
        weights += [model.actor.mu.weight]
        # Get biases
        biases = [model.actor.latent_pi[int(i * 2)].bias for i in range(len(policy_size))]
        biases += [model.actor.mu.bias]
        # Convert Torch to JAX model
        jax_policy_state = torch_to_jax(jax_policy_state, weights, biases)

    elif RL_method == "A2C":
        policy_kwargs = dict(activation_fn=activation_fn_torch,
                             net_arch=policy_size)

        model = A2C("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)

        # Train
        model.learn(total_timesteps=total_steps)

        # A2C Should return an actor critic policy
        assert isinstance(model.policy, ActorCriticPolicy)

        # Get weights
        weights = [model.policy.mlp_extractor.policy_net[int(i * 2)].weight for i in range(len(policy_size))]
        weights += [model.policy.action_net.weight]
        # Get biases
        biases = [model.policy.mlp_extractor.policy_net[int(i * 2)].bias for i in range(len(policy_size))]
        biases += [model.policy.action_net.bias]
        # Convert Torch to JAX model
        jax_policy_state = torch_to_jax(jax_policy_state, weights, biases)

    elif RL_method == "DDPG":
        policy_kwargs = dict(activation_fn=activation_fn_torch,
                             net_arch=policy_size)

        # The noise objects for DDPG
        n_actions = vec_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = DDPG("MlpPolicy", vec_env, action_noise=action_noise, policy_kwargs=policy_kwargs, verbose=1)

        # Remove the tanh activation function, which DDPG sets by default
        if not allow_tanh:
            model.actor.mu = model.actor.mu[:-1]

        # Train
        model.learn(total_timesteps=total_steps)

        # Get weights
        weights = [model.actor.mu[int(i * 2)].weight for i in range(len(policy_size) + 1)]
        # Get biases
        biases = [model.actor.mu[int(i * 2)].bias for i in range(len(policy_size) + 1)]
        # Convert Torch to JAX model
        jax_policy_state = torch_to_jax(jax_policy_state, weights, biases)

    elif RL_method == "ARS":
        policy_kwargs = dict(activation_fn=activation_fn_torch,
                             net_arch=policy_size)

        model = ARS("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)

        # Remove the tanh activation function, which ARS sets by default
        if not allow_tanh:
            model.policy.action_net = model.policy.action_net[:-1]

        # Train
        model.learn(total_timesteps=total_steps)

        # Get weights
        weights = [model.policy.action_net[int(i * 2)].weight for i in range(len(policy_size) + 1)]
        # Get biases
        biases = [model.policy.action_net[int(i * 2)].bias for i in range(len(policy_size) + 1)]
        # Convert Torch to JAX model
        jax_policy_state = torch_to_jax(jax_policy_state, weights, biases)

    elif RL_method == "TQC":
        policy_kwargs = dict(activation_fn=activation_fn_torch,
                             net_arch=policy_size)

        model = TQC("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)

        # Train
        model.learn(total_timesteps=total_steps)

        # Get weights
        weights = [model.actor.latent_pi[int(i * 2)].weight for i in range(len(policy_size))]
        weights += [model.actor.mu.weight]
        # Get biases
        biases = [model.actor.latent_pi[int(i * 2)].bias for i in range(len(policy_size))]
        biases += [model.actor.mu.bias]
        # Convert Torch to JAX model
        jax_policy_state = torch_to_jax(jax_policy_state, weights, biases)

    elif RL_method == "TRPO":
        policy_kwargs = dict(activation_fn=activation_fn_torch,
                             net_arch=policy_size)

        model = TRPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)

        # Train
        model.learn(total_timesteps=total_steps)

        # PPO Should return an actor critic policy
        assert isinstance(model.policy, ActorCriticPolicy)

        # Get weights
        weights = [model.policy.mlp_extractor.policy_net[int(i * 2)].weight for i in range(len(policy_size))]
        weights += [model.policy.action_net.weight]
        # Get biases
        biases = [model.policy.mlp_extractor.policy_net[int(i * 2)].bias for i in range(len(policy_size))]
        biases += [model.policy.action_net.bias]
        # Convert Torch to JAX model
        jax_policy_state = torch_to_jax(jax_policy_state, weights, biases)

    else:
        print(f'ERROR: Invalid RL algorithm provided ({RL_method})')
        return None

    return model, jax_policy_state


def pretrain_policy(args, env_name, cwd, RL_method, seed, num_envs, total_steps, policy_size,
                    activation_fn_jax, activation_fn_txt, allow_tanh):
    start_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    activation_fn_torch = torch.nn.ReLU

    # Set random seed in stable baselines
    set_random_seed(seed)

    # Generate environment. TD3 and DDPG are episodic and can thus not use multiple environments.
    vec_env = make_vec_env(env_name, n_envs=(1 if RL_method in ["TD3", "DDPG"] else num_envs),
                           env_kwargs={'args': args}, seed=seed)

    if RL_method in ["TD3", "DDPG", "ARS", "TRPO"] and allow_tanh:
        print(f'(!) Set policy output activation function to tanh (used by default for "{RL_method}")')
        activation_fn_jax += [nn.tanh]
        activation_fn_txt += ['tanh']
    else:
        print(f'(!) Set no policy output activation function (used by default for "{RL_method}")')
        activation_fn_jax += [None]
        activation_fn_txt += ['None']

    # For JAX, we also need to specify the output size (not necessary for torch)
    n_actions = vec_env.action_space.shape[-1]
    policy_size_jax = policy_size + [n_actions]

    model, jax_policy_state = train_stable_baselines(vec_env,
                                                     RL_method, policy_size, policy_size_jax, activation_fn_torch,
                                                     activation_fn_jax, total_steps, allow_tanh)

    print('- Training complete')

    ######
    # Export JAX policy as Orbax checkpoint
    ckpt_export_file = f"ckpt/{env_name}_layout={args.layout}_alg={RL_method}_layers={args.hidden_layers}_neurons={args.neurons_per_layer}_outfn={activation_fn_txt[-1]}_seed={seed}_steps={total_steps}"
    checkpoint_path = Path(cwd, ckpt_export_file)

    # Additional configuration info (stored in checkpoint)
    config = orbax_set_config(start_datetime=start_datetime, env_name=env_name, layout=args.layout, seed=seed,
                              RL_method=RL_method, total_steps=total_steps,
                              neurons_per_layer=policy_size_jax, activation_fn_txt=activation_fn_txt)

    print('- Export configured')

    # Checkpoint consists of policy state and config dictionary
    ckpt = {'model': jax_policy_state, 'config': config}

    orbax_checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    orbax_checkpointer.save(checkpoint_path, ckpt,
                            save_args=flax.training.orbax_utils.save_args_from_target(ckpt), force=True)
    print(f'- Exported checkpoint for method "{RL_method}" (seed {seed}) to file: {checkpoint_path}')

    return vec_env, model, jax_policy_state, checkpoint_path


if __name__ == "__main__":

    # Use CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    parser = argparse.ArgumentParser(prefix_chars='--')
    parser.add_argument('--model', type=str, default="LinearSystem",
                        help="Dynamical model to train on")
    parser.add_argument('--layout', type=int, default=0,
                        help="Select a particular layout for the benchmark model (if this option exists)")
    parser.add_argument('--algorithm', type=str, default="ALL",
                        help="RL algorithm to train with")
    parser.add_argument('--total_steps', type=int, default=100000,
                        help="Number of steps to train for")
    parser.add_argument('--seed', type=int, default=1,
                        help="Random number generator seed")
    parser.add_argument('--num_envs', type=int, default=10,
                        help="Number of parallel environments to train with (>1 does not work for all algorithms)")

    ### NEURAL NETWORK ARCHITECTURE
    parser.add_argument('--neurons_per_layer', type=int, default=128,
                        help="Number of neurons per (hidden) layer.")
    parser.add_argument('--hidden_layers', type=int, default=3,
                        help="Number of hidden layers.")

    parser.add_argument('--allow_tanh', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, allow the use of tanh output activation function on policies")

    args = parser.parse_args()
    args.cwd = os.getcwd()

    if args.algorithm == "ALL":
        METHODS = ["PPO", "SAC", "A2C", "ARS", "TRPO", "TQC", "TD3", "DDPG"]
    elif args.algorithm == "ALL_fast":
        METHODS = ["PPO", "SAC", "A2C", "ARS", "TRPO", "TQC"]
    elif args.algorithm == "ALL_paper":
        METHODS = ["SAC", "A2C", "TRPO", "TQC"]
    else:
        METHODS = [str(args.algorithm)]

    policy_size = [args.neurons_per_layer for _ in range(args.hidden_layers)]
    activation_fn_jax = [nn.relu for _ in range(args.hidden_layers)]
    activation_fn_txt = ['relu' for _ in range(args.hidden_layers)]
    activation_fn_torch = torch.nn.ReLU

    model = {}
    jax_policy_state = {}
    checkpoint_path = {}

    for z, RL_method in enumerate(METHODS):
        print(f'\n=== Algorithm {z}: {RL_method} ===')

        model[RL_method] = {}
        jax_policy_state[RL_method] = {}
        checkpoint_path[RL_method] = {}

        seed = args.seed
        print(f'\n--- Seed: {seed} ---')

        env = models.get_model_fun(args.model)(args)  # Also generate non-vectorized environment to get access to other parameters

        vec_env, model[RL_method][seed], jax_policy_state[RL_method][seed], checkpoint_path[RL_method][seed] = \
            pretrain_policy(args=args,
                            env_name=args.model,
                            cwd=args.cwd,
                            RL_method=RL_method,
                            seed=seed,
                            num_envs=args.num_envs,
                            total_steps=args.total_steps,
                            policy_size=policy_size,
                            activation_fn_jax=copy.copy(activation_fn_jax),
                            activation_fn_txt=copy.copy(activation_fn_txt),
                            allow_tanh=args.allow_tanh)

        print('\n--- Training completed and checkpoint exported ---')

        ######
        # Plot
        H = 100
        ax = plt.figure().add_subplot()

        print('\n--- Create plot... ---')

        for j in range(10):
            traces = np.zeros((H + 1, args.num_envs, vec_env.observation_space.shape[-1]))
            actions = np.zeros((H, args.num_envs, vec_env.action_space.shape[-1]))

            traces[0] = vec_env.reset()

            for i in range(H):
                actions[i], _states = model[RL_method][seed].predict(traces[i], deterministic=True)

                traces[i + 1], rewards, dones, info = vec_env.step(actions[i])

                actions_jax = jax_policy_state[RL_method][seed].apply_fn(jax_policy_state[RL_method][seed].params,
                                                                         traces[i])
                # print('- Difference:', actions[i] - actions_jax)

            for i in range(args.num_envs):
                plt.plot(traces[:, i, env.plot_dim[0]], traces[:, i, env.plot_dim[1]], color="gray", linewidth=1, markersize=1)
                plt.plot(traces[0, i, env.plot_dim[0]], traces[0, i, env.plot_dim[1]], 'ro')
                plt.plot(traces[-1, i, env.plot_dim[0]], traces[-1, i, env.plot_dim[1]], 'bo')

        ax.set_title(f"Initialized policy ({RL_method}, {args.total_steps} steps, seed={seed})", fontsize=10)

        filename = f"output/policyInit_{args.model}_alg={RL_method}_steps={int(args.total_steps)}_seed={seed}_tanh={args.allow_tanh}"
        filepath = Path(args.cwd, filename).with_suffix('.png')
        plt.savefig(filepath, format='png', bbox_inches='tight', dpi=300)

        filename = f"output/policyInit_{args.model}_alg={RL_method}_steps={int(args.total_steps)}_seed={seed}_tanh={args.allow_tanh}_B"
        plot_traces(env, jax_policy_state[RL_method][seed], key=jax.random.PRNGKey(2), folder=args.cwd, filename=filename,
                    title=True)
