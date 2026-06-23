from functools import partial
from typing import Callable
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from flax import struct
from flax.training.train_state import TrainState


def load_policy_config(checkpoint_path, key):
    '''
    Load the configuration of a saved neural network checkpoint.

    :param checkpoint_path: file path to a Orbax checkpoint file.
    :param key: string 'V_config' or 'Policy_config', describing whether the certificate or the policy should be loaded. 
    :return: dictionary, giving the relevant configuration.
    '''

    # First read only the config from the orbax checkpoint
    orbax_checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    ckpt_restored = orbax_checkpointer.restore(checkpoint_path)
    Policy_config = ckpt_restored[key]

    return Policy_config


def create_nn_states(env, Policy_config, V_neurons_withOut, V_act_fn_withOut, pi_neurons_per_layer,
                     Policy_lr=5e-5, V_lr=5e-4):
    '''
    Create Jax state objects (for both policy and certificate).

    :param env: Benchmark model.
    :param Policy_config: Configuration for policy.
    :param V_neurons_withOut: Number of neurons per layer of certificate network (including the output dimension).
    :param V_act_fn_withOut: Activation function per layer of certificate network (including the output act. func.).
    :param pi_neurons_per_layer: Number of neurons per layer of policy network (excluding the output dimension).
    :param Policy_lr: Policy network learning rate. 
    :param V_lr: Certificate learning rate.
    :return:
        - V_state: Initialized certificate network 
        - Policy_state: Initialized policy network
        - Policy_config: Configuration for policy.
        - Policy_neurons_withOut: Number of neurons per layer of policy network (including the output dimension).
    '''

    # Initialize certificate network
    certificate_model = MLP(V_neurons_withOut, V_act_fn_withOut)
    V_state = create_train_state(
        model=certificate_model,
        act_funcs=V_act_fn_withOut,
        rng=jax.random.PRNGKey(1),
        in_dim=env.state_dim,
        learning_rate=V_lr,
    )

    # Parse policy activation functions (txt -> jax functions)
    Policy_act_fn_withOut = orbax_parse_activation_fn(Policy_config['activation_fn'])
    Policy_neurons_withOut = pi_neurons_per_layer + [len(env.action_space.low)]

    # Create policy state object
    policy_model = MLP(Policy_neurons_withOut, Policy_act_fn_withOut)
    Policy_state = create_train_state(
        model=policy_model,
        act_funcs=Policy_act_fn_withOut,
        rng=jax.random.PRNGKey(1),
        in_dim=env.state_dim,
        learning_rate=Policy_lr,
    )

    return V_state, Policy_state, Policy_config, Policy_neurons_withOut


def orbax_set_config(start_datetime=None, env_name=None, layout=None, seed=None, RL_method=None, total_steps=None,
                     neurons_per_layer=None, activation_fn_txt=None):
    '''
    Set configuration of Orbax neural network checkpoint. 

    :param start_datetime: Time at which training was started.
    :param env_name: Name of the environment.
    :param layout: Layout of the environment.
    :param seed: Random seed used.
    :param RL_method: RL algorithm that was used for the pretraining. 
    :param total_steps: Number of steps for which the network was (pre)trained. 
    :param neurons_per_layer: Number of neurons per layer of the network. 
    :param activation_fn_txt: List of strings describing the activation functions.
    :return: configuration dictionary
    '''

    config = {
        'date_created': start_datetime,
        'env_name': env_name,
        'layout': layout,
        'seed': seed,
        'algorithm': RL_method,
        'total_steps': total_steps,
        'neurons_per_layer': neurons_per_layer,
        'activation_fn': activation_fn_txt
    }

    return config


def orbax_parse_activation_fn(activation_fn_txt):
    '''
    Parse list of activation functions as flax functions.

    :param activation_fn_txt: List of strings describing the activation functions.
    :return: List of flax functions describing the activation functions.
    '''

    activation_fn = [None] * len(activation_fn_txt)
    for i, fn in enumerate(activation_fn_txt):
        if fn == 'relu':
            activation_fn[i] = nn.relu
        elif fn == 'tanh':
            activation_fn[i] = nn.tanh
        elif fn == 'softplus':
            activation_fn[i] = nn.softplus
        elif fn == 'None':
            activation_fn[i] = None
        else:
            print(f'(!!!) Warning: unknown activation function ({fn}) in checkpoint config encountered')

    return activation_fn


def create_batches(data_length, batch_size):
    '''
    Create batches for the given data and batch size. Returns the start and end indices to iterate over.

    :param data_length: Total number of data points.
    :param batch_size: Number of points per batch.
    :return: Each batch is represented by the slice [starts[i]:ends[i]].
    '''

    num_batches = np.ceil(data_length / batch_size).astype(int)
    starts = np.arange(num_batches) * batch_size
    ends = np.minimum(starts + batch_size, data_length)

    return starts, ends


def apply_ibp_rectangular(act_fns, params, mean, radius):
    '''
    Implementation of the interval bound propagation (IBP) method from https://arxiv.org/abs/1810.12715.
    We use IBP to compute upper and lower bounds for (hyper)rectangular input sets.

    This function returns the same result as jax_verify.interval_bound_propagation(apply_fn, initial_bounds). However,
    the jax_verify version is generally slower, because it is written to handle more general neural networks.

    :param act_fns: List of flax.nn activation functions.
    :param params: Parameter dictionary of the network.
    :param mean: 2d array, with each row being an input point of dimension n.
    :param radius: 1d array, specifying the radius of the input in every dimension.
    :return: lb and ub (both 2d arrays of the same shape as `mean`)
    '''

    # Broadcast radius to match shape of the mean numpy array
    radius = jnp.broadcast_to(radius, mean.shape)

    # Enumerate over the layers of the network
    for i, act_fn in enumerate(act_fns):
        layer = 'Dense_' + str(i)

        # Compute mean and radius after the current fully connected layer
        mean = mean @ params['params'][layer]['kernel'] + params['params'][layer]['bias']
        radius = radius @ jnp.abs(params['params'][layer]['kernel'])

        if act_fn is not None:
            # Then, apply the activation function and determine the lower and upper bounds
            lb = act_fn(mean - radius)
            ub = act_fn(mean + radius)

            # Use these upper bounds to determine the mean and radius after the layer
            mean = (ub + lb) / 2
            radius = (ub - lb) / 2

        else:
            lb = mean - radius
            ub = mean + radius

    return lb, ub


class AgentState(TrainState):
    '''
    Class inherited from the TrainState class from flax. 
    It sets default values for agent functions to make TrainState work in jitted function.
    '''    
    
    ibp_fn: Callable = struct.field(pytree_node=False)


def create_train_state(model, act_funcs, rng, in_dim, learning_rate=0.01, ema=0, params=None):
    '''
    Create a flax TrainState object.

    :param model: MLP object describing number of neurons per layer and the activation functions. 
    :param act_funcs: List of activation functions (required separately for the IBP function). 
    :param rng: random number generator object. 
    :param in_dim: Input dimension of the network.
    :param learning_rate: Learning rate. 
    :param ema: Rate of the EMA (exponential moving average) used in the optimizer. 
    :param params: Additional parameters to be passed to the (flax) TrainState class.
    :return: flax TrainState object.
    '''

    if params is None:
        params = model.init(rng, jnp.ones([1, in_dim]))
    else:
        params = params

    tx = optax.adam(learning_rate)
    if ema > 0:
        tx = optax.chain(tx, optax.ema(ema))
    return AgentState.create(apply_fn=jax.jit(model.apply), params=params, tx=tx,
                             ibp_fn=jax.jit(partial(apply_ibp_rectangular, act_funcs)))


@partial(jax.jit, static_argnums=(1, 2, 3,))
def lipschitz_coeff(params, weighted, CPLip, Linfty):
    '''
    Function to compute Lipschitz constants using the techniques presented in the paper.

    :param params: Neural network parameters.
    :param weighted: If true, use weighted norms.
    :param CPLip: If true, use the average activation operators (cplip) improvement.
    :param Linfty: If true, use Linfty norm; If false, use L1 norm (currently only L1 norm is used).
    :return: Lipschitz constant and list of weights (or None if weighted is False).
    '''

    if Linfty:
        axis = 0
    else:
        axis = 1

    minweight = jnp.float32(1e-6)
    maxweight = jnp.float32(1e6)

    if (not weighted and not CPLip):
        L = jnp.float32(1)
        # Compute Lipschitz coefficient by iterating through layers
        for layer in params["params"].values():
            # Involve only the 'kernel' dictionaries of each layer in the network, which are the weight matrices
            if "kernel" in layer:
                L *= jnp.max(jnp.sum(jnp.abs(layer["kernel"]), axis=axis))

    elif (not weighted and CPLip):
        L = jnp.float32(0)
        matrices = []
        for layer in params["params"].values():
            # Collect all weight matrices of the network
            if "kernel" in layer:
                matrices.append(layer["kernel"])

        nmatrices = len(matrices)
        # Create a list with all products of consecutive weight matrices
        # products[i][j] is the matrix product matrices[i + j] ... matrices[j]
        products = [matrices]
        prodnorms = [[jnp.max(jnp.sum(jnp.abs(mat), axis=axis)) for mat in matrices]]
        for nprods in range(1, nmatrices):
            prod_list = []
            for idx in range(nmatrices - nprods):
                prod_list.append(jnp.matmul(products[nprods - 1][idx], matrices[idx + nprods]))
            products.append(prod_list)
            prodnorms.append([jnp.max(jnp.sum(jnp.abs(mat), axis=axis)) for mat in prod_list])

        ncombs = 1 << (nmatrices - 1)
        for idx in range(ncombs):
            # To iterate over all possible ways of putting norms or products between the layers, 
            #  interpret idx as binary number of length (nmatrices - 1),
            # where the jth bit determines whether to put a norm or a product between layers j and j+1
            # We use that the (nmatrices - 1)th bit of such number is always 0, which implies that
            # each layer is taken into account for each term in the sum. 
            jprev = 0
            Lloc = jnp.float32(1)
            for jcur in range(nmatrices):
                if idx & (1 << jcur) == 0: 
                    Lloc *= prodnorms[jcur - jprev][jprev]
                    jprev = jcur + 1

            L += Lloc / ncombs


    elif (weighted and not CPLip and not Linfty):
        L = jnp.float32(1)
        matrices = []
        for layer in params["params"].values():
            # Collect all weight matrices of the network
            if "kernel" in layer:
                matrices.append(layer["kernel"])
        matrices.reverse()

        weights = [jnp.ones(jnp.shape(matrices[0])[1])]
        for mat in matrices:
            colsums = jnp.sum(jnp.multiply(jnp.abs(mat), weights[-1][jnp.newaxis, :]), axis=1)
            lip = jnp.maximum(jnp.max(colsums), minweight)
            weights.append(jnp.maximum(colsums / lip, minweight))
            L *= lip

    elif (weighted and not CPLip and Linfty):
        L = jnp.float32(1)
        matrices = []
        for layer in params["params"].values():
            # Collect all weight matrices of the network
            if "kernel" in layer:
                matrices.append(layer["kernel"])

        weights = [jnp.ones(jnp.shape(matrices[0])[0])]
        for mat in matrices:
            rowsums = jnp.sum(jnp.multiply(jnp.abs(mat), jnp.float32(1) / weights[-1][:, jnp.newaxis]), axis=0)
            lip = jnp.max(rowsums)
            weights.append(jnp.minimum(lip / rowsums, maxweight))
            L *= lip

    elif (weighted and CPLip and not Linfty):
        L = jnp.float32(0)
        matrices = []
        for layer in params["params"].values():
            # Collect all weight matrices of the network
            if "kernel" in layer:
                matrices.append(layer["kernel"])
        matrices.reverse()

        weights = [jnp.ones(jnp.shape(matrices[0])[1])]
        for mat in matrices:
            colsums = jnp.sum(jnp.multiply(jnp.abs(mat), weights[-1][jnp.newaxis, :]), axis=1)
            lip = jnp.maximum(jnp.max(colsums), minweight)
            weights.append(jnp.maximum(colsums / lip, minweight))

        matrices.reverse()
        nmatrices = len(matrices)
        # Create a list with all products of consecutive weight matrices
        # products[i][j] is the matrix product matrices[i + j] ... matrices[j]
        products = [matrices]
        prodnorms = [[jnp.max(jnp.multiply(jnp.sum(jnp.multiply(jnp.abs(matrices[idx]),
                                                                weights[-(idx + 2)][jnp.newaxis, :]), axis=1),
                                           jnp.float32(1) / weights[-(idx + 1)]))
                      for idx in range(nmatrices)]]
        for nprods in range(1, nmatrices):
            prod_list = []
            for idx in range(nmatrices - nprods):
                prod_list.append(jnp.matmul(products[nprods - 1][idx], matrices[idx + nprods]))
            products.append(prod_list)
            prodnorms.append([jnp.max(jnp.multiply(jnp.sum(jnp.multiply(jnp.abs(prod_list[idx]),
                                                                        weights[-(idx + nprods + 2)][jnp.newaxis, :]),
                                                           axis=1),
                                                   jnp.float32(1) / weights[-(idx + 1)]))
                              for idx in range(nmatrices - nprods)])

        ncombs = 1 << (nmatrices - 1)
        for idx in range(ncombs):
            # To iterate over all possible ways of putting norms or products between the layers, 
            #  interpret idx as binary number of length (nmatrices - 1),
            # where the jth bit determines whether to put a norm or a product between layers j and j+1
            # We use that the (nmatrices - 1)th bit of such number is always 0, which implies that
            # each layer is taken into account for each term in the sum. 
            jprev = 0
            Lloc = jnp.float32(1)
            for jcur in range(nmatrices):
                if idx & (1 << jcur) == 0: 
                    Lloc *= prodnorms[jcur - jprev][jprev]
                    jprev = jcur + 1

            L += Lloc / ncombs

    elif (weighted and CPLip and Linfty):
        L = jnp.float32(0)
        matrices = []
        for layer in params["params"].values():
            # Collect all weight matrices of the network
            if "kernel" in layer:
                matrices.append(layer["kernel"])

        weights = [jnp.ones(jnp.shape(matrices[0])[0])]
        for mat in matrices:
            rowsums = jnp.sum(jnp.multiply(jnp.abs(mat), jnp.float32(1) / weights[-1][:, jnp.newaxis]), axis=0)
            lip = jnp.max(rowsums)
            weights.append(jnp.minimum(lip / rowsums, maxweight))
        weights.reverse()

        nmatrices = len(matrices)
        # Create a list with all products of consecutive weight matrices
        # products[i][j] is the matrix product matrices[i + j] ... matrices[j]
        products = [matrices]
        prodnorms = [[jnp.max(jnp.multiply(jnp.sum(jnp.multiply(jnp.abs(matrices[idx]),
                                                                jnp.float32(1) / weights[-(idx + 1)][:, jnp.newaxis]),
                                                   axis=0),
                                           weights[-(idx + 2)]))
                      for idx in range(nmatrices)]]
        for nprods in range(1, nmatrices):
            prod_list = []
            for idx in range(nmatrices - nprods):
                prod_list.append(jnp.matmul(products[nprods - 1][idx], matrices[idx + nprods]))
            products.append(prod_list)
            prodnorms.append([jnp.max(jnp.multiply(jnp.sum(jnp.multiply(jnp.abs(prod_list[idx]),
                                                                        jnp.float32(1) / weights[-(idx + 1)][:,
                                                                                         jnp.newaxis]), axis=0),
                                                   weights[-(idx + nprods + 2)]))
                              for idx in range(nmatrices - nprods)])

        ncombs = 1 << (nmatrices - 1)
        for idx in range(ncombs):
            # To iterate over all possible ways of putting norms or products between the layers, 
            #  interpret idx as binary number of length (nmatrices - 1),
            # where the jth bit determines whether to put a norm or a product between layers j and j+1
            # We use that the (nmatrices - 1)th bit of such number is always 0, which implies that
            # each layer is taken into account for each term in the sum. 
            jprev = 0
            Lloc = jnp.float32(1)
            for jcur in range(nmatrices):
                if idx & (1 << jcur) == 0:
                    Lloc *= prodnorms[jcur - jprev][jprev]
                    jprev = jcur + 1

            L += Lloc / ncombs

        weights.reverse()

    if weighted:
        return L, weights[-1]
    else:
        return L, None


class MLP(nn.Module):
    ''' Define multi-layer perception with JAX '''
    features: Sequence[int]
    activation_func: list

    def setup(self):
        # we automatically know what to do with lists, dicts of submodules
        self.layers = [nn.Dense(feat) for feat in self.features]
        # for single submodules, we would just write:
        # self.layer1 = nn.Dense(feat1)

    @nn.compact
    def __call__(self, x):
        for act_func, feat in zip(self.activation_func, self.features):
            if act_func is None:
                x = nn.Dense(feat)(x)
            else:
                x = act_func(nn.Dense(feat)(x))
        return x
