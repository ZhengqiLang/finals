import jax
import jax.numpy as jnp
import numpy as np


class Buffer:
    '''
    Class to store samples (from the state space) in a buffer.
    '''

    def __init__(self, dim, extra_dims=0, max_size=100_000_000):
        '''
        :param dim: integer, the length (i.e., dimension) of each sample.
        :param extra_dims: integer, the number of extra dimensions that are added to the samples, to store extra data.
        :param max_size: integer, the maximal size of the buffer.
        '''
        self.dim = dim
        self.extra_dims = extra_dims
        self.data = np.zeros(shape=(0, dim + extra_dims), dtype=np.float32)
        self.max_size = max_size

    def append(self, samples):
        '''
        Append given samples to training buffer

        :param samples: numpy array containing the samples to append.
        '''

        assert samples.shape[1] == self.dim + self.extra_dims, \
            f"Samples have wrong dimension (namely of shape {samples.shape})"

        # Check if buffer exceeds length. If not, add new samples
        if not (self.max_size is not None and len(self.data) > self.max_size):
            append_samples = np.array(samples, dtype=np.float32)
            self.data = np.vstack((self.data, append_samples), dtype=np.float32)

    def append_and_remove(self, refresh_fraction, samples, perturb=False, cell_width=False, verbose=False, weighted_sampling=False):
        '''
        Removes a given fraction of the training buffer and appends the given samples.

        :param refresh_fraction: float, fraction of the buffer to refresh.
        :param samples: numpy array containing the samples to append.
        :param perturb: boolean. If true, perturb each samples (within their cells; uniform distribution).
        :param cell_width: boolean or float. If a float, it is the size of each cell (only required if perturb is True).
        :param verbose: boolean. If true, print more information.
        :param weighted_sampling: boolean. If true, refresh buffer according to the given weights.
        '''

        assert samples.shape[1] == self.dim + self.extra_dims, \
            f"Samples have wrong dimension (namely of shape {samples.shape})"

        # Determine how many old and new samples are kept in the buffer
        nr_old = int((1 - refresh_fraction) * len(self.data))
        nr_new = int(self.max_size - nr_old)

        # Select indices to keep
        old_idxs = np.random.choice(len(self.data), nr_old, replace=False)

        if weighted_sampling:

            # Samples store three nonnegative weights (one for each type of violation)
            # The following line computes for how many samples at least one weight is positive
            nonzero_p = np.sum(np.sum(samples[:, self.dim:self.dim + 3], axis=1) > 0)
            if nr_new <= nonzero_p:
                replace = False
            else:
                replace = True

            # Weighted sampling over new counterexamples (proportional to the weights returned by the verifier)
            probabilities = np.sum(samples[:, self.dim:self.dim + 3], axis=1) / np.sum(
                samples[:, self.dim:self.dim + 3])
            new_idxs = np.random.choice(len(samples), nr_new, replace=replace, p=probabilities)
        else:

            if nr_new <= len(samples):
                replace = False
            else:
                replace = True

            print('- Number of violations to pick from:', len(samples))

            # Uniform sampling over new counterexamples
            new_idxs = np.random.choice(len(samples), nr_new, replace=replace)

        old_samples = self.data[old_idxs]
        new_samples = samples[new_idxs]

        if perturb:
            # Perturb samples within the given cell width
            new_widths = cell_width[new_idxs]

            # Generate perturbation
            perturbations = np.random.uniform(low=-0.5 * new_widths, high=0.5 * new_widths,
                                              size=new_samples[:, :self.dim].T.shape).T

            if verbose:
                print('Widths:')
                print(new_widths)

                print('Perturbation:')
                print(perturbations)

            # Add perturbation (but exclude the additional dimensions)
            new_samples[:, :self.dim] += perturbations

        self.data = np.vstack((old_samples, new_samples), dtype=np.float32)


def define_grid(low, high, size):
    '''
    Set rectangular grid over state space for neural network learning
    Specifically, given lower and upper bounds low[i] and high[i] for each dimension, 
    and the number of points size[i] for each dimension, creates the grid consisting
    of all prod_i size[i] points whose ith coordinate can be written as
      low[i] + j (high[i] - low[i])/(size[i]-1) 
    for some 0 <= j <= size[i]-1 that can depend on i.

    :param low: List of floats (lower bound grid per dimension).
    :param high: List of floats (upper bound grid per dimension).
    :param size: List of ints (entries per dimension).
    :return: Numpy array of size (prod_i size[i], len(size)), containing the points in the grid.
    '''

    points = [np.linspace(low[i], high[i], size[i]) for i in range(len(size))]
    grid = np.vstack(list(map(np.ravel, np.meshgrid(*points)))).T

    return grid


@jax.jit
def meshgrid_jax(points, size):
    '''
    Set rectangular grid over state space for neural network learning (using jax)
    Specifically, given a list of points points[i] for each dimension, 
    creates the grid consisting of all prod_i len(points[i]) points 
    whose ith coordinate is an element of points[i] for all i.

    :param points: List of len(size) lists of floats (coordinates per dimension).
    :param size: List of ints (entries per dimension).
    :return: Jax numpy array of size (prod_i size[i], len(size)), containing the points in the grid.
    '''

    meshgrid = jnp.asarray(jnp.meshgrid(*points))
    grid = jnp.reshape(meshgrid, (len(size), -1)).T

    return grid


def define_grid_jax(low, high, size, mode='linspace'):
    '''
    Set rectangular grid over state space for neural network learning (using jax)
    Specifically, given lower and upper bounds low[i] and high[i] for each dimension, 
    and the number of points size[i] for each dimension, creates the grid consisting
    of all prod_i size[i] points whose ith coordinate can be written as
      low[i] + j (high[i] - low[i])/(size[i]-1) 
    for some 0 <= j <= size[i]-1 that can depend on i.

    :param low: List of floats (lower bound grid per dimension).
    :param high: List of floats (upper bound grid per dimension).
    :param size: List of ints (entries per dimension).
    :param mode: Determines whether the numpy function linspace or arange is used.
    :return: Jax numpy array of size (prod_i size[i], len(size)), containing the points in the grid.
    '''

    if mode == 'linspace':
        points = [np.linspace(low[i], high[i], size[i]) for i in range(len(size))]
    else:
        step = (high - low) / (size - 1)
        points = [np.arange(low[i], high[i] + step[i] / 2, step[i]) for i in range(len(size))]
    grid = meshgrid_jax(points, size)

    return grid


def mesh2cell_width(mesh, dim, Linfty):
    '''
    Convert mesh size in L1 (or Linfty) norm to cell width in a rectangular gridding
    Given the L1 or Linfty norm ||.||, computes the cell width of the largest 
    axis-aligned rectangle inside a cell of the form {x : ||x-c|| <= mesh}, where c is
    some fixed arbitary center (not required to be specified).
    
    :param mesh: float, the norm bound from the center of the cell.
    :param dim: int, the dimension of the state space.
    :param Linfty: boolean, whether the Linfty norm (rather than the L1 norm) should be used.
    :return: float, the cell width of the cell.
    '''

    return mesh * 2 if Linfty else mesh * (2 / dim)


def cell_width2mesh(cell_width, dim, Linfty):
    '''
    Convert cell width in L1 norm to mesh size in a rectangular gridding
    Given the L1 or Linfty norm ||.||, computes the largest mesh such that a 
    axis-aligned rectangle with given cell_width and center c (not specified)
    contains the set {x : ||x-c|| <= mesh}.

    :param cell_width: float, cell width of the cell.
    :param dim: int, dimension of the state space.
    :param Linfty: boolean, whether the Linfty norm (rather than the L1 norm) should be used.
    :return: float, the norm bound from the center of the cell.
    '''

    return cell_width / 2 if Linfty else cell_width * (dim / 2)
