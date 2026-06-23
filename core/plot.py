from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from core.buffer import define_grid
from core.commons import MultiRectangularSet


def position_fraction(point, lb, ub):
    '''
    Calculate the relative position at which the given point is, within the box [lb, ub]. 
    For example, the point point=0.5 is at a fraction of 0.95 within [-1.4, 0.6].
    Note that the answer is between 0 and 1 if and only if the point is inside [lb, ub].

    :param point: Float, the input point. 
    :param lb: Float, the input lower bound. 
    :param ub: Float, the input upper bound. 
    :return: Float, the relative position of point in the box [lb, ub].
    '''

    return (point - lb) / (ub - lb)


def position_in_heatmap(point, lb, ub, heatmap_size):
    '''
    Calculate where the given point is located in a heatmap. 
    Note that in a heatmap, the upper left corner is (0,0), while in normal plots, the bottom left corner is (0,0).

    :param point: Float, the input point. 
    :param lb: Float, the input lower bound. 
    :param ub: Float, the input upper bound. 
    :param heatmap_size: The upper limits of the heatmap.
    :return: The position on the heatmap. 
    '''

    fraction = position_fraction(point, lb, ub)
    heatmap_x = fraction[0] * heatmap_size[0]
    heatmap_y = (1 - fraction[1]) * heatmap_size[1]

    return np.array([heatmap_x, heatmap_y])


def plot_boxes(env, ax, plot_dimensions=[0, 1], labels=False, latex=False, size=12):
    '''
    Plot the target, initial, and unsafe state sets.

    :param env: Environment
    :param ax: A matplotlib figure.
    :param plot_dimensions: List of length two, mentioning which of the dimensions of the state space to plot. 
    :param labels: boolean. If true, add labels to the axes.
    :param latex: boolean. If true, use LaTeX for the labels of the axes. 
    :param size: integer. Size of the axis labels. 
    '''

    lsize = size + 4

    # Plot target set
    if isinstance(env.target_space, MultiRectangularSet):
        for set in env.target_space.sets:
            width, height = (set.high - set.low)[plot_dimensions]
            ax.add_patch(Rectangle(set.low[plot_dimensions], width, height, fill=False, edgecolor='green'))

            MID = (set.high + set.low)[plot_dimensions] / 2
            if labels:
                if latex:
                    text = r'$\mathcal{X}_T$'
                else:
                    text = 'X_T'
                ax.annotate(text, MID, color='green', fontsize=lsize, ha='center', va='center')
    else:
        width, height = (env.target_space.high - env.target_space.low)[plot_dimensions]
        ax.add_patch(Rectangle(env.target_space.low[plot_dimensions], width, height, fill=False, edgecolor='green'))

        MID = (env.target_space.high + env.target_space.low)[plot_dimensions] / 2
        if labels:
            if latex:
                text = r'$\mathcal{X}_T$'
            else:
                text = 'X_T'
            ax.annotate(text, MID, color='green', fontsize=lsize, ha='center', va='center')

    # Plot unsafe set
    if isinstance(env.unsafe_space, MultiRectangularSet):
        for set in env.unsafe_space.sets:
            width, height = (set.high - set.low)[plot_dimensions]
            ax.add_patch(Rectangle(set.low[plot_dimensions], width, height, fill=False, edgecolor='red'))

            MID = (set.high + set.low)[plot_dimensions] / 2
            if labels:
                if latex:
                    text = r'$\mathcal{X}_U$'
                else:
                    text = 'X_U'
                ax.annotate(text, MID, color='red', fontsize=lsize, ha='center', va='center')
    else:
        width, height = (env.unsafe_space.high - env.unsafe_space.low)[plot_dimensions]
        ax.add_patch(Rectangle(env.unsafe_space.low[plot_dimensions], width, height, fill=False, edgecolor='red'))

        MID = (env.unsafe_space.high + env.unsafe_space.low)[plot_dimensions] / 2
        if labels:
            if latex:
                text = r'$\mathcal{X}_U$'
            else:
                text = 'X_U'
            ax.annotate(text, MID, color='red', fontsize=lsize, ha='center', va='center')

    # Plot initial set
    if isinstance(env.init_space, MultiRectangularSet):
        for set in env.init_space.sets:
            width, height = (set.high - set.low)[plot_dimensions]
            ax.add_patch(Rectangle(set.low[plot_dimensions], width, height, fill=False, edgecolor='black'))

            MID = (set.high + set.low)[plot_dimensions] / 2
            if labels:
                if latex:
                    text = r'$\mathcal{X}_0$'
                else:
                    text = 'X_0'
                ax.annotate(text, MID, color='black', fontsize=lsize, ha='center', va='center')
    else:
        width, height = (env.init_space.high - env.init_space.low)[plot_dimensions]
        ax.add_patch(Rectangle(env.init_space.low[plot_dimensions], width, height, fill=False, edgecolor='black'))

        MID = (env.init_space.high + env.init_space.low)[plot_dimensions] / 2
        if labels:
            if latex:
                text = r'$\mathcal{X}_0$'
            else:
                text = 'X_0'
            ax.annotate(text, MID, color='black', fontsize=lsize, ha='center', va='center')

    return


def plot_traces(env, Policy_state, key, num_traces=10, len_traces=1000, folder=False, filename=False, title=True):
    '''
    Plot simulated traces under the given policy.

    :param env: Environment. 
    :param Policy_state: Policy network.
    :param key: random number generator key. 
    :param num_traces: Number of traces to be plotted. 
    :param len_traces: Number of environment steps done per trace.
    :param folder: boolean or string. If not false, should be the pathname of the folder where to save the plot. 
    :param filename: boolean or string. If not false, should be the filename on which the plot should be saved. 
    :param title: boolean. If true, add a title. 
    :return: Numpy array containing the traces. 
    '''

    plot_dim = env.plot_dim

    # Simulate traces
    traces = np.zeros((len_traces + 1, num_traces, len(env.state_space.low)))
    actions = np.zeros((len_traces, num_traces, len(env.action_space.low)))

    if len(plot_dim) == 2:
        ax = plt.figure().add_subplot()
    else:
        ax = plt.figure().add_subplot(projection='3d')

    # Initialize traces
    for i in range(num_traces):

        key, subkey = jax.random.split(key)

        x = env.init_space.sample_single(subkey)
        traces[0, i] = x

        succes = False
        for j in range(len_traces):
            # Get state and action
            state = traces[j, i]
            action = Policy_state.apply_fn(Policy_state.params, state)
            actions[j, i] = action

            # Make step in environment
            traces[j + 1, i], key = env.step_noise_key(state, key, action)

            if env.target_space.contains(np.array([traces[j + 1, i]]), return_indices=True)[0]:
                succes = True
                if len(plot_dim) == 2:
                    plt.plot(traces[0: j + 1, i, plot_dim[0]], traces[0: j + 1, i, plot_dim[1]], 'o', color="gray", linewidth=1, markersize=1)
                    plt.plot(traces[0, i, plot_dim[0]], traces[0, i, plot_dim[1]], 'ro')
                    plt.plot(traces[j + 1, i, plot_dim[0]], traces[j + 1, i, plot_dim[1]], 'bo')
                else:
                    plt.plot(traces[0: j + 1, i, plot_dim[0]], traces[0: j + 1, i, plot_dim[1]], traces[0: j + 1, i, plot_dim[2]], 'o', color="gray", linewidth=1, markersize=1)
                    plt.plot(traces[0, i, plot_dim[0]], traces[0, i, plot_dim[1]], traces[0, i, plot_dim[2]], 'ro')
                    plt.plot(traces[j + 1, i, plot_dim[0]], traces[j + 1, i, plot_dim[1]], traces[j + 1, i, plot_dim[2]], 'bo')
                break

        if not succes:
            if len(plot_dim) == 2:
                plt.plot(traces[:, i, plot_dim[0]], traces[:, i, plot_dim[1]], 'o', color="gray", linewidth=1, markersize=1)
                plt.plot(traces[0, i, plot_dim[0]], traces[0, i, plot_dim[1]], 'ro')
                plt.plot(traces[-1, i, plot_dim[0]], traces[-1, i, plot_dim[1]], 'bo')
            else:
                plt.plot(traces[:, i, plot_dim[0]], traces[:, i, plot_dim[1]], traces[:, i, plot_dim[2]], 'o', color="gray", linewidth=1, markersize=1)
                plt.plot(traces[0, i, plot_dim[0]], traces[0, i, plot_dim[1]], traces[0, i, plot_dim[2]], 'ro')
                plt.plot(traces[-1, i, plot_dim[0]], traces[-1, i, plot_dim[1]], traces[-1, i, plot_dim[2]], 'bo')

    # Plot traces
    if len(plot_dim) == 2:

        # Plot relevant state sets
        plot_boxes(env, ax, plot_dimensions=plot_dim)

        # Goal x-y limits
        low = env.state_space.low
        high = env.state_space.high
        ax.set_xlim(low[plot_dim[0]], high[plot_dim[0]])
        ax.set_ylim(low[plot_dim[1]], high[plot_dim[1]])

        if title:
            ax.set_title(f"Simulated traces ({filename})", fontsize=10)

        if hasattr(env, 'variable_names'):
            plt.xlabel(env.variable_names[plot_dim[0]])
            plt.ylabel(env.variable_names[plot_dim[1]])

    elif len(plot_dim) == 3:

        # Goal x-y limits
        low = env.state_space.low
        high = env.state_space.high
        ax.set_xlim(low[plot_dim[0]], high[plot_dim[0]])
        ax.set_ylim(low[plot_dim[1]], high[plot_dim[1]])
        ax.set_zlim(low[plot_dim[2]], high[plot_dim[2]])

        if title:
            ax.set_title(f"Simulated traces ({filename})", fontsize=10)

        if hasattr(env, 'variable_names'):
            ax.set_xlabel(env.variable_names[plot_dim[0]])
            ax.set_ylabel(env.variable_names[plot_dim[1]])
            ax.set_zlabel(env.variable_names[plot_dim[2]])

    else:
        print('Incompatible plot dimensions')
        return

    if folder and filename:
        # Save figure
        for form in ['png']:  # ['pdf', 'png']:
            filepath = Path(folder, filename)
            filepath = filepath.parent / (filepath.name + '.' + form)
            plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)

    return traces


def plot_dataset(env, train_data=None, additional_data=None, folder=False, filename=False, title=True):
    '''
    Plot the given samples.

    :param env: Environment.
    :param train_data: Data points used in the training, to be plotted in black.
    :param additional_data: Additional data, to be plotted in blue (used for, e.g., counterexamples or hard violations). 
    :param folder: boolean or string. If not false, should be the pathname of the folder where to save the plot. 
    :param filename: boolean or string. If not false, should be the filename on which the plot should be saved. 
    :param title: boolean. If true, add a title. 
    '''

    plot_dim = env.plot_dim
    if len(plot_dim) != 2:
        print(
            f">> Cannot create dataset plot: environment has wrong state dimension (namely {len(env.state_space.low)}).")
        return

    fig, ax = plt.subplots()

    # Plot data points in buffer that are not in the stabilizing set
    if train_data is not None:
        x = train_data[:, plot_dim[0]]
        y = train_data[:, plot_dim[1]]
        plt.scatter(x, y, color='black', s=0.1)

    if additional_data is not None:
        x = additional_data[:, plot_dim[0]]
        y = additional_data[:, plot_dim[1]]
        plt.scatter(x, y, color='blue', s=0.1)

    # Plot relevant state sets
    plot_boxes(env, ax, plot_dimensions=plot_dim)

    # XY limits
    low = env.state_space.low
    high = env.state_space.high
    ax.set_xlim(low[plot_dim[0]], high[plot_dim[0]])
    ax.set_ylim(low[plot_dim[1]], high[plot_dim[1]])

    if title:
        ax.set_title(f"Sample plot ({filename})", fontsize=10)

    if hasattr(env, 'variable_names'):
        plt.xlabel(env.variable_names[plot_dim[0]])
        plt.ylabel(env.variable_names[plot_dim[1]])

    if folder and filename:
        if Path(folder).exists():
            # Save figure
            for form in ['png']:  # ['pdf', 'png']:
                filepath = Path(folder, filename)
                filepath = filepath.parent / (filepath.name + '.' + form)
                plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)
        else:
            print(f"- Cannot save figure; folder {folder} does not exists")

    return


def vector_plot(env, Pi_state, vectors_per_dim=40, seed=1, folder=False, filename=False, title=True):
    '''
    Create vector plot of the closed-loop dynamics under the given policy

    :param env: Environment. 
    :param Pi_state: Policy network.
    :param vectors_per_dim: Number of vectors in the vector plot in each dimension. 
    :param seed: Seed used for computing the next state.
    :param folder: boolean or string. If not false, should be the pathname of the folder where to save the plot. 
    :param filename: boolean or string. If not false, should be the filename on which the plot should be saved. 
    :param title: boolean. If true, add a title. 
    '''

    plot_dim = env.plot_dim
    if len(plot_dim) not in [2, 3]:
        print(
            f">> Cannot create vector plot: environment has wrong state dimension (namely {len(env.state_space.low)}).")
        return

    grid = define_grid(env.state_space.low, env.state_space.high, size=[vectors_per_dim] * env.state_dim)

    # Get actions
    action = Pi_state.apply_fn(Pi_state.params, grid)

    key = jax.random.split(jax.random.PRNGKey(seed), len(grid))

    # Make step
    next_obs, env_key, steps_since_reset, reward, terminated, truncated, infos \
        = env.vstep(jnp.array(grid, dtype=jnp.float64), key, action, jnp.zeros(len(grid), dtype=jnp.int64))

    scaling = 1
    vectors = (next_obs - grid) * scaling

    # Plot vectors
    if len(plot_dim) == 2:
        ax = plt.figure().add_subplot()
        ax.quiver(grid[:, plot_dim[0]], grid[:, plot_dim[1]], vectors[:, plot_dim[0]], vectors[:, plot_dim[1]], angles='xy')

        # Plot relevant state sets
        plot_boxes(env, ax, plot_dimensions=plot_dim)

        if title:
            ax.set_title(f"Closed-loop dynamics ({filename})", fontsize=10)

        if hasattr(env, 'variable_names'):
            plt.xlabel(env.variable_names[plot_dim[0]])
            plt.ylabel(env.variable_names[plot_dim[1]])

    elif len(plot_dim) == 3:
        ax = plt.figure().add_subplot(projection='3d')
        ax.quiver(grid[:, plot_dim[0]], grid[:, plot_dim[1]], grid[:, plot_dim[2]], vectors[:, plot_dim[0]], vectors[:, plot_dim[1]], vectors[:, plot_dim[2]],
                  length=0.5, normalize=False, arrow_length_ratio=0.5)

        ax.set_title(f"Closed-loop dynamics ({filename})", fontsize=10)

        if hasattr(env, 'variable_names'):
            ax.set_xlabel(env.variable_names[plot_dim[0]])
            ax.set_ylabel(env.variable_names[plot_dim[1]])
            ax.set_zlabel(env.variable_names[plot_dim[2]])

    if folder and filename:
        # Save figure
        for form in ['png']:  # ['pdf', 'png']:
            filepath = Path(folder, filename)
            filepath = filepath.parent / (filepath.name + '.' + form)
            plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)

    return


def plot_layout(env, folder=False, filename=False, title=True, latex=False, size=12):
    '''
    Plot the layout of the reach-avoid specification.

    :param env: Environment. 
    :param folder: boolean or string. If not false, should be the pathname of the folder where to save the plot. 
    :param filename: boolean or string. If not false, should be the filename on which the plot should be saved. 
    :param title: boolean. If true, add a title. 
    :param latex: boolean. If true, use LaTeX for the labels of the axes. 
    :param size: integer. Size of the axis labels. 
    '''

    if latex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Helvetica"
        })

    plot_dim = env.plot_dim
    if len(plot_dim) != 2:
        print(
            f">> Cannot create layout plot: environment has wrong state dimension (namely {len(env.state_space.low)}).")
        return

    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    # fig.set_facecolor('blue')

    # Plot relevant state sets
    plot_boxes(env, ax, labels=True, latex=latex, size=size, plot_dimensions=plot_dim)

    # Goal x-y limits
    low = env.state_space.low
    high = env.state_space.high
    ax.set_xlim(low[plot_dim[0]], high[plot_dim[0]])
    ax.set_ylim(low[plot_dim[1]], high[plot_dim[1]])

    if type(title) == str:
        ax.set_title(title, fontsize=size)
    elif title is True:
        ax.set_title(f"Reach-avoid layout ({filename})", fontsize=size)

    if latex:
        plt.xlabel('$x_1$', fontsize=size)
        plt.ylabel('$x_2$', fontsize=size)
    else:
        plt.xlabel('x1', fontsize=size)
        plt.ylabel('x2', fontsize=size)

    plt.xticks(fontsize=size, rotation=90)
    plt.yticks(fontsize=size)

    ax.patch.set(lw=1, ec='black')

    if folder and filename:
        # Save figure
        for form in ['pdf', 'png']:
            filepath = Path(folder, filename)
            filepath = filepath.parent / (filepath.name + '.' + form)
            plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)

    return


def plot_certificate_2D(env, cert_state, folder=False, filename=False, logscale=False, title=True, labels=True,
                        resolution=101, latex=False, size=10, contour=False):
    '''
    Plot the given RASM as a 2D heatmap.

    :param env: Environment. 
    :param cert_state: Certificate network.
    :param folder: boolean or string. If not false, should be the pathname of the folder where to save the plot. 
    :param filename: boolean or string. If not false, should be the filename on which the plot should be saved. 
    :param logscale: boolean. If true, use a logarithmic scale. 
    :param title: boolean. If true, add a title. 
    :param labels: boolean. If true, add labels to the axes.
    :param resolution: integer. Number of points used for the heatmap in each dimension. 
    :param latex: boolean. If true, use LaTeX for the labels of the axes. 
    :param size: integer. Size of the axis labels. 
    :param contour: boolean. If true, plot contour lines.
    '''

    if latex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Helvetica"
        })

    dim = env.state_dim
    plot_dim = env.plot_dim

    fig, ax = plt.subplots()

    # Visualize certificate network
    grid = define_grid(env.state_space.low, env.state_space.high,
                       size=[resolution if i in plot_dim else 1 for i in range(dim)])

    # Set dimensions that are not plot to their mean value (in the state space)
    not_plot_dim = [i for i in range(dim) if i not in plot_dim]
    grid[:, [not_plot_dim]] = (env.state_space.high[not_plot_dim] + env.state_space.low[not_plot_dim]) / 2

    # Only keep unique elements in first two dimensions
    _, idxs = np.unique(grid[:, [plot_dim]], return_index=True, axis=0)
    grid = grid[idxs]

    X = np.round(grid[:, plot_dim[0]], 3)
    Y = np.round(grid[:, plot_dim[1]], 3)
    out = cert_state.apply_fn(cert_state.params, grid).flatten()

    data = pd.DataFrame(data={'x': X, 'y': Y, 'z': out})

    data = data.pivot(index='y', columns='x', values='z')[::-1]

    ax.tick_params(axis='y', which='both', labelsize=size)
    ax.tick_params(axis='x', which='both', labelsize=size)

    if contour:
        cs = ax.contour(np.arange(len(data.index)), np.arange(len(data.columns))[::-1], data.to_numpy(), levels=10)
        plt.clabel(cs, inline=1, fontsize=9)
    else:
        if logscale:
            sns.heatmap(data, norm=LogNorm())
        else:
            sns.heatmap(data, xticklabels=20, yticklabels=20)

        # use matplotlib.colorbar.Colorbar object
        cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=size)

    ax.tick_params(axis='y', which='both', rotation=0)
    ax.tick_params(axis='x', which='both', rotation=90)

    xcells = data.shape[1]
    ycells = data.shape[0]
    xycells = np.array([xcells, ycells])
    center = 0.5 * xycells
    scale = xycells / (env.state_space.high - env.state_space.low)[plot_dim] * np.array(
        [1, -1])

    #####

    lsize = size + 4

    # Plot target set
    if isinstance(env.target_space, MultiRectangularSet):
        for set in env.target_space.sets:
            LB = position_in_heatmap(set.low[plot_dim], env.state_space.low[plot_dim],
                                     env.state_space.high[plot_dim], xycells)
            UB = position_in_heatmap(set.high[plot_dim], env.state_space.low[plot_dim],
                                     env.state_space.high[plot_dim], xycells)
            ax.add_patch(Rectangle(LB, (UB - LB)[0], (UB - LB)[1], fill=False, edgecolor='green'))

            if labels:
                if latex:
                    text = r'$\mathcal{X}_T$'
                else:
                    text = 'X_T'
                ax.annotate(text, (UB + LB) / 2, color='green', fontsize=lsize, ha='center', va='center')

    else:
        LB = position_in_heatmap(env.target_space.low[plot_dim], env.state_space.low[plot_dim],
                                 env.state_space.high[plot_dim], xycells)
        UB = position_in_heatmap(env.target_space.high[plot_dim], env.state_space.low[plot_dim],
                                 env.state_space.high[plot_dim], xycells)
        ax.add_patch(Rectangle(LB, (UB - LB)[0], (UB - LB)[1], fill=False, edgecolor='green'))

        if labels:
            if latex:
                text = r'$\mathcal{X}_T$'
            else:
                text = 'X_T'
            ax.annotate(text, (UB + LB) / 2, color='green', fontsize=lsize, ha='center', va='center')

    # Plot unsafe set
    if isinstance(env.unsafe_space, MultiRectangularSet):
        for set in env.unsafe_space.sets:
            LB = position_in_heatmap(set.low[plot_dim], env.state_space.low[plot_dim],
                                     env.state_space.high[plot_dim], xycells)
            UB = position_in_heatmap(set.high[plot_dim], env.state_space.low[plot_dim],
                                     env.state_space.high[plot_dim], xycells)
            ax.add_patch(Rectangle(LB, (UB - LB)[0], (UB - LB)[1], fill=False, edgecolor='black'))

            if labels:
                if latex:
                    text = r'$\mathcal{X}_U$'
                else:
                    text = 'X_U'
                ax.annotate(text, (UB + LB) / 2, color='black', fontsize=lsize, ha='center', va='center')

    else:
        LB = position_in_heatmap(env.unsafe_space.low[plot_dim], env.state_space.low[plot_dim],
                                 env.state_space.high[plot_dim], xycells)
        UB = position_in_heatmap(env.unsafe_space.high[plot_dim], env.state_space.low[plot_dim],
                                 env.state_space.high[plot_dim], xycells)
        ax.add_patch(Rectangle(LB, (UB - LB)[0], (UB - LB)[1], fill=False, edgecolor='black'))

        if labels:
            if latex:
                text = r'$\mathcal{X}_U$'
            else:
                text = 'X_U'
            ax.annotate(text, (UB + LB) / 2, color='black', fontsize=lsize, ha='center', va='center')

    # Plot initial set
    if isinstance(env.init_space, MultiRectangularSet):
        for set in env.init_space.sets:
            LB = position_in_heatmap(set.low[plot_dim], env.state_space.low[plot_dim],
                                     env.state_space.high[plot_dim], xycells)
            UB = position_in_heatmap(set.high[plot_dim], env.state_space.low[plot_dim],
                                     env.state_space.high[plot_dim], xycells)
            ax.add_patch(Rectangle(LB, (UB - LB)[0], (UB - LB)[1], fill=False, edgecolor='yellow'))

            if labels:
                if latex:
                    text = r'$\mathcal{X}_0$'
                else:
                    text = 'X_0'
                ax.annotate(text, (UB + LB) / 2, color='yellow', fontsize=lsize, ha='center', va='center')

    else:
        LB = position_in_heatmap(env.init_space.low[plot_dim], env.state_space.low[plot_dim],
                                 env.state_space.high[plot_dim], xycells)
        UB = position_in_heatmap(env.init_space.high[plot_dim], env.state_space.low[plot_dim],
                                 env.state_space.high[plot_dim], xycells)
        ax.add_patch(Rectangle(LB, (UB - LB)[0], (UB - LB)[1], fill=False, edgecolor='yellow'))

        if labels:
            if latex:
                text = r'$\mathcal{X}_0$'
            else:
                text = 'X_0'
            ax.annotate(text, (UB + LB) / 2, color='yellow', fontsize=lsize, ha='center', va='center')

    #####

    if type(title) == str:
        ax.set_title(title, fontsize=size)
    elif title is True:
        ax.set_title(f"Learned certificate ({filename})", fontsize=size)

    if hasattr(env, 'variable_names'):
        plt.xlabel(env.variable_names[plot_dim[0]], fontsize=size)
        plt.ylabel(env.variable_names[plot_dim[1]], fontsize=size)

    if labels:
        plt.xticks(fontsize=size)
        plt.yticks(fontsize=size)

    if folder and filename:
        # Save figure
        for form in ['pdf', 'png']:
            filepath = Path(folder, filename)
            filepath = filepath.parent / (filepath.name + '.' + form)
            plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)


def plot_heatmap(env, coordinates, values, folder=False, filename=False, title=True, size=10):
    '''
    Generitc function to plot a heatmap for the given coordinates and values.

    :param env: Environment. 
    :param coordinates: List of coordinates for which a value is given.
    :param values: List of the values corresponding to the coordinates.
    :param folder: boolean or string. If not false, should be the pathname of the folder where to save the plot. 
    :param filename: boolean or string. If not false, should be the filename on which the plot should be saved. 
    :param title: boolean. If true, add a title. 
    :param size: integer. Size of the axis labels. 
    '''

    plot_dim = env.plot_dim

    if len(plot_dim) != 2:
        print(
            f">> Cannot create heatmap: environment has wrong state dimension (namely {len(plot_dim)}).")
        return

    fig, ax = plt.subplots()

    # Only keep unique elements in the plot dimensions
    _, idxs = np.unique(coordinates[:, [plot_dim]], return_index=True, axis=0)
    coordinates = coordinates[idxs]
    values = values[idxs]

    # Create dataframe with X,Y,Z as columns and pivot
    data = pd.DataFrame(data={'x': coordinates[:, plot_dim[0]], 'y': coordinates[:, plot_dim[1]], 'z': values})
    data = data.pivot(index='y', columns='x', values='z')[::-1]
    sns.heatmap(data)
    
    # Reformat axis labels to avoid floating point issues in axis labels
    #  (based on https://github.com/mwaskom/seaborn/issues/1005#issue-175150095)
    fmt = '{:0.3f}'
    xticklabels = []
    for item in ax.get_xticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        xticklabels.append(item)
    yticklabels = []
    for item in ax.get_yticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        yticklabels.append(item)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
  
    if folder and filename:
        if Path(folder).exists():
            # Save figure
            for form in ['png']:  # ['pdf', 'png']:
                filepath = Path(folder, filename)
                filepath = filepath.parent / (filepath.name + '.' + form)
                plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)
        else:
            print(f"- Cannot save figure; folder {folder} does not exists")

    return
