import argparse
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint

# Import all benchmark models
import models
from LipBaB import LipBaB
from core.commons import Namespace
from core.jax_utils import lipschitz_coeff, orbax_parse_activation_fn, load_policy_config, \
    create_nn_states


def lipschitz_LipBaB(params, state_space):
    wts = []
    bs = []
    for layer in params["params"].values():
        if "kernel" in layer:
            wts.append(layer["kernel"])
            bs.append(layer["bias"])

    L = jax.pure_callback(LipBaB, jax.ShapeDtypeStruct((), jnp.float32), jax.lax.stop_gradient(wts),
                          jax.lax.stop_gradient(bs), state_space)
    return L, None


def run_LipBaB(checkpoint_path):
    print(f'- Use checkpoint in folder "{checkpoint_path}"')

    Policy_config = load_policy_config(checkpoint_path, key='Policy_config')
    V_config = load_policy_config(checkpoint_path, key='V_config')
    general_config = load_policy_config(checkpoint_path, key='general_config')

    # Create gym environment (jax/flax version)
    envfun = models.get_model_fun(Policy_config['env_name'])


    # Define empty namespace and store layout attribute
    args = Namespace
    args.layout = Policy_config['layout']

    # Build environment
    env = envfun(args)

    V_neurons_withOut = V_config['neurons_per_layer']
    V_act_fn_withOut_txt = V_config['activation_fn']
    V_act_fn_withOut = orbax_parse_activation_fn(V_act_fn_withOut_txt)

    pi_neurons_withOut = Policy_config['neurons_per_layer']
    pi_act_funcs_txt = Policy_config['activation_fn']
    pi_act_funcs_jax = orbax_parse_activation_fn(pi_act_funcs_txt)

    # Load policy configuration
    V_state, Policy_state, Policy_config, Policy_neurons_withOut = create_nn_states(env, Policy_config,
                                                                                    V_neurons_withOut,
                                                                                    V_act_fn_withOut,
                                                                                    pi_neurons_withOut)

    # Restore state of policy and certificate network
    orbax_checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    target = {'general_config': general_config, 'V_state': V_state, 'Policy_state': Policy_state, 'V_config': V_config,
              'Policy_config': Policy_config}

    Policy_state = orbax_checkpointer.restore(checkpoint_path, item=target)['Policy_state']
    V_state = orbax_checkpointer.restore(checkpoint_path, item=target)['V_state']
    state_space = [[env.state_space.low[i], env.state_space.high[i]] for i in range(env.state_space.dimension)]

    # policy network
    print("Policy network")
    flags = [[True, True]]
    for f in flags:
        time0 = time.time()
        print(f, lipschitz_coeff(Policy_state.params, *f, False), time.time() - time0)

    # do it again to see difference with/without jit compilation time
    for f in flags:
        time0 = time.time()
        print(f, lipschitz_coeff(Policy_state.params, *f, False), time.time() - time0)

    time0 = time.time()
    print(lipschitz_LipBaB(Policy_state.params, state_space))
    print(time.time() - time0)

    # certificate network
    print("Certificate network")
    flags = [[True, True]]
    for f in flags:
        time0 = time.time()
        print(f, lipschitz_coeff(V_state.params, *f, False), time.time() - time0)

    time0 = time.time()
    print(lipschitz_LipBaB(V_state.params, state_space))
    print(time.time() - time0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='',
                        help="File to load orbax checkpoint from")
    args = parser.parse_args()
    args.cwd = os.getcwd()

    checkpoint_path = Path(args.cwd, args.checkpoint)
    run_LipBaB(checkpoint_path=checkpoint_path)
