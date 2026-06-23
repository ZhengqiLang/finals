from functools import partial
'''
this file is for only 1 sample in center point
'''
import jax
import numpy as np
from flax.training.train_state import TrainState
from jax import numpy as jnp

from core.commons import MultiRectangularSet
from core.jax_utils import lipschitz_coeff
from core.plot import plot_dataset


def range_from_raw(raw, eps=1e-3):
    # raw: (N,2) or (...,2)
    a = raw[..., 0]
    b = raw[..., 1]
    nmin = jax.nn.softplus(a) + eps
    nmax = nmin + jax.nn.softplus(b) + eps
    return nmin, nmax




class Learner:
    '''
    Main learner class.

    '''

    def __init__(self, env, args):
        '''
        Initialize the learner.

        :param env: Environment. 
        :param args: Command line arguments given. 
        '''

        self.env = env
        self.linfty = False  # L_infty has only experimental support (not used in experiments)

        # Copy some arguments
        self.auxiliary_loss = args.auxiliary_loss
        self.lambda_lipschitz = args.loss_lipschitz_lambda  # Lipschitz factor
        self.max_lip_certificate = args.loss_lipschitz_certificate  # Above this value, incur loss
        self.max_lip_policy = args.loss_lipschitz_policy  # Above this value, incur loss
        self.weighted = args.weighted
        self.cplip = args.cplip
        self.split_lip = args.split_lip
        self.min_lip_policy = args.min_lip_policy_loss
        self.exp_certificate = args.exp_certificate
        self.loss_decr_squared = args.loss_decr_squared
        self.loss_decr_max = args.loss_decr_max
        self.EPS_decrease = args.eps_decrease

        # Set batch sizes
        self.batch_size_total = int(args.batch_size)
        self.batch_size_base = int(args.batch_size * (1 - args.counterx_fraction))
        self.batch_size_counterx = int(args.batch_size * args.counterx_fraction)

        # Calculate the number of samples for each region type (without counterexamples)
        MIN_SAMPLES = max(int(args.min_fraction_samples_per_region * self.batch_size_base), 1)





        self.local_n = None              # 每个基础点周围采样多少个邻域点
        self.local_radius_max = 0.7      # 邻域半径
        self.local_radius_min = 0.05    #领域小值
        self.local_weight_alpha = 20.0 # 距离权重衰减强度
        self.local_use_uniform = True  # 用均匀采样还是高斯采样  


        self.local_weight_type = getattr(args, "local_weight_type", "inverse")
        self.local_gaussian_sigma_scale = getattr(args, "local_gaussian_sigma_scale", 0.1)




        self.local_distance_type = args.distance_type
        self.local_distance_weight = jnp.array(args.distance_weights,dtype=jnp.float32)




        totvol = env.state_space.volume
        if isinstance(env.init_space, MultiRectangularSet):
            rel_vols = np.array([Set.volume / totvol for Set in env.init_space.sets])
            self.num_samples_init = tuple(np.maximum(np.ceil(rel_vols * self.batch_size_base), MIN_SAMPLES).astype(int))
        else:
            self.num_samples_init = np.maximum(MIN_SAMPLES,
                                               np.ceil(env.init_space.volume / totvol * self.batch_size_base)).astype(
                int)
        if isinstance(env.unsafe_space, MultiRectangularSet):
            rel_vols = np.array([Set.volume / totvol for Set in env.unsafe_space.sets])
            self.num_samples_unsafe = tuple(
                np.maximum(MIN_SAMPLES, np.ceil(rel_vols * self.batch_size_base)).astype(int))
        else:
            self.num_samples_unsafe = np.maximum(np.ceil(env.unsafe_space.volume / totvol * self.batch_size_base),
                                                 MIN_SAMPLES).astype(int)
        if isinstance(env.target_space, MultiRectangularSet):
            rel_vols = np.array([Set.volume / totvol for Set in env.target_space.sets])
            self.num_samples_target = tuple(
                np.maximum(np.ceil(rel_vols * self.batch_size_base), MIN_SAMPLES).astype(int))
        else:
            self.num_samples_target = np.maximum(MIN_SAMPLES, np.ceil(
                env.target_space.volume / totvol * self.batch_size_base)).astype(int)

        # Infer the number of expected decrease samples based on the other batch sizes
        self.num_samples_decrease = np.maximum(self.batch_size_base
                                               - np.sum(self.num_samples_init)
                                               - np.sum(self.num_samples_unsafe)
                                               - np.sum(self.num_samples_target), 1).astype(int)

        if not args.silent:
            print(f'- Num. base train samples per batch: {self.batch_size_base}')
            print(f'-- Initial state: {self.num_samples_init}')
            print(f'-- Unsafe state: {self.num_samples_unsafe}')
            print(f'-- Target state: {self.num_samples_target}')
            print(f'-- Expected decrease: {self.num_samples_decrease}')
            print(f'- Num. counterexamples per batch: {self.batch_size_counterx}\n')

        if self.lambda_lipschitz > 0 and not args.silent:
            print('- Learner setting: Enable Lipschitz loss')
            print(f'--- For certificate up to: {self.max_lip_certificate:.3f}')
            print(f'--- For policy up to: {self.max_lip_policy:.3f}')

        self.glob_min = 0.1
        self.N_expectation = 1

        # Define vectorized functions for loss computation
        self.loss_exp_decrease_vmap = jax.vmap(self.loss_exp_decrease, in_axes=(None, None, 0, 0, 0, None), out_axes=0)

        return

    def loss_exp_decrease(self, V_state, V_params, x, u, noise_key, probability_bound):
        '''
        Compute expected certificate value in the new state for the loss related to condition 3 (expected decrease).
        
        :param V_state: Certificate neural network. 
        :param V_params: Parameters of the certificate neural network. 
        :param x: State.
        :param u: Action.
        :param noise_key: key of the random number generator.
        :param probability_bound: The probability bound of the specification that we aim to certify.
        :return: Expected certificate value in the new state.
        '''

        # For each given noise_key, compute the successor state for the pair (x,u)
        state_new, noise_key = self.env.vstep_noise_batch(x, noise_key, u)

        # Function apply_fn does a forward pass in the certificate network for all successor states in state_new,
        # which approximates the value of the certificate for the successor state (using different noise values).
        # Then, the loss term is zero if the expected decrease in certificate value is at least tau*K.
        if self.exp_certificate:
            V_expected = jnp.log(jnp.mean(jnp.exp(
                jnp.minimum(V_state.apply_fn(V_params, state_new), jnp.log(2) - jnp.log(1 - probability_bound))
            )))
            # The jnp.minimum ensures that values do not become infinite (due to the exponential).
            # Note that this retains the soundness since we may cap any valid lograsm at - jnp.log(1 - probability_bound).
        else:
            V_expected = jnp.mean(
                jnp.minimum(V_state.apply_fn(V_params, state_new), 2 / (1 - probability_bound))
            )

        return V_expected

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self,
                   key: jax.Array,
                   V_state: TrainState,
                   Policy_state: TrainState,
                   counterexamples,
                   mesh_loss,
                   probability_bound,
                   expDecr_multiplier,
                   Range_state
                   ):
        '''
        Perform one step of training the neural network.

        :param key: key of the random number generator.
        :param V_state: Certificate network.
        :param Policy_state: Policy network.
        :param counterexamples: Current list of counterexamples.
        :param mesh_loss: float, determining the largest mesh for which a loss of 0 implies that the condition is satisfied. 
        :param probability_bound: The probability bound of the specification that we aim to certify. 
        :param expDecr_multiplier: Multiplier of the expected decrease loss. 
        :return:
           - V_grads: Gradients of the certificate network. 
           - Policy_grads: Gradients of the policy network. 
           - infos: Dictionary, giving the total loss as well as each component (total, init, unsafe, expDecrease_mean, expDecrease_max (not used), Lipschitz (not used)).
           - key: key of the random number generator.
           - samples_in_batch: Dictionary, giving the samples used in the batch per category. 
        '''

        # Generate all random keys
        key, cx_key, init_key, unsafe_key, target_key, decrease_key, noise_key, perturbation_key = \
            jax.random.split(key, 8)

        # Sample from the full list of counterexamples
        if len(counterexamples) > 0:
            # Randomly sample counterexamples from the buffer
            cx = jax.random.choice(cx_key, counterexamples, shape=(self.batch_size_counterx,), replace=False)
            cx_samples = cx[:, :-3]

            # Determine which counterexamples belong to which categories
            cx_bool_init = cx[:, -2] > 0
            cx_bool_unsafe = cx[:, -1] > 0
            cx_bool_decrease = cx[:, -3] > 0
        else:
            # No counterexamples in the buffer yet (e.g., in first iteration)
            cx_samples = cx_bool_init = cx_bool_unsafe = cx_bool_decrease = False

        # Sample from each region of interest
        center_init = self.env.init_space.sample_single(init_key)
        center_unsafe = self.env.unsafe_space.sample_single(unsafe_key)
        center_target = self.env.target_space.sample_single(target_key)
        center_decrease = self.env.state_space.sample_single(decrease_key)

        center = center_decrease
        V_Center = jnp.ravel(V_state.apply_fn(V_state.params, center[None,:]))


        # print("***********V_center***************",V_Center)



        # def sample_neighbors(self, center, key,n):


        #     dirs = jax.random.normal(key, shape=(self.local_n, self.env.state_dim))
        #     dirs = dirs / (jnp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)

        #     r = jax.random.uniform(key, shape=(self.local_n, 1), minval=0.0, maxval=1.0)
        #     r = n * jnp.sqrt(r)   # 2D圆盘用 sqrt 更均匀

        #     offsets = dirs * r
        #     neighbors_only = center[None, :] + offsets                  # (local_n, state_dim)
        #     neighbors = jnp.concatenate([center[None, :], neighbors_only], axis=0)  # (local_n+1, state_dim)
        #     dists = jnp.linalg.norm(neighbors - center[None, :], axis=1)             # (local_n+1,)
        #     return neighbors, dists
        

        def sample_neighbors(self,center,key,n,local_n):
            D = self.env.state_dim

            if self.local_distance_type == "l2":
                key_dir, key_r = jax.random.split(key, 2)
                dirs = jax.random.normal(key_dir, shape=(local_n, D))
                dirs = dirs / (jnp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8) #get direction vertor in 1
                u = jax.random.uniform(key_r, shape=(local_n, 1), minval=0.0, maxval=1.0) #probability u=pi*r^2/pi*n^2
                r = n * (u ** (1.0 / D))  # r
                offsets = dirs * r  # r*direction
                dists_only = r[:, 0] 
            
            elif self.local_distance_type == "inf":
                offsets = jax.random.uniform(
                key,
                shape=(local_n, D),
                minval=-n,
                maxval=n,
            )
                dists_only = jnp.max(jnp.abs(offsets), axis=1)
            
            elif self.local_distance_type == "l1":
                oversample = local_n * 6
                cand = jax.random.uniform(
                    key,
                    shape=(oversample, D),
                    minval=-n,
                    maxval=n,
                )
                cand_d = jnp.sum(jnp.abs(cand), axis=1)
                mask = cand_d <= n

                idx = jnp.where(mask, size=local_n, fill_value=0)[0]
                offsets = cand[idx]
                dists_only = jnp.sum(jnp.abs(offsets), axis=1)




            neighbors_only = center[None, :] + offsets
            neighbors = jnp.concatenate([center[None, :], neighbors_only], axis=0)
            dists = jnp.concatenate([jnp.array([0.0], dtype=neighbors.dtype), dists_only], axis=0)
            return neighbors, dists

        

        # Exclude samples from target set
        
        def total_num_samples(num_samples):
            return int(np.sum(num_samples))
        def loss_fun(certificate_params, policy_params,range_params):



            #compute center and N by neural network
# ------------------------------------------------------------
# Compute one local radius for each center using the same Range network.
# n_region = Range(center_region, V(center_region))
# ------------------------------------------------------------
            def compute_radius(center_point):
                V_center_point = jnp.ravel(
                    V_state.apply_fn(certificate_params, center_point[None, :])
                )[0]

                radius_input = jnp.concatenate(
                    [center_point, jnp.array([V_center_point])],
                    axis=0,
                )[None, :]

                raw_n = Range_state.apply_fn(range_params, radius_input)
                raw_n = raw_n[0, 0]

                n = jax.nn.softplus(raw_n) + 1e-3
                return n, V_center_point


            n_init, V_center_init = compute_radius(center_init)
            n_unsafe, V_center_unsafe = compute_radius(center_unsafe)
            n_target, V_center_target = compute_radius(center_target)
            n_decrease, V_center_decrease = compute_radius(center_decrease)
            # n = 0.5



            # Original volume-based total numbers
            N_init = total_num_samples(self.num_samples_init)
            N_unsafe = total_num_samples(self.num_samples_unsafe)
            N_target = total_num_samples(self.num_samples_target)
            N_decrease = total_num_samples(self.num_samples_decrease)

            # One center has already been included.
            # Therefore the number of neighbors is total_N - 1.
            local_n_init = max(N_init - 1, 0)
            local_n_unsafe = max(N_unsafe - 1, 0)
            local_n_target = max(N_target - 1, 0)
            local_n_decrease = max(N_decrease - 1, 0)
            key_init_local, key_unsafe_local, key_target_local, key_decrease_local = jax.random.split(
                perturbation_key,
                4,
            )

            samples_init, dists_init = sample_neighbors(
                self,
                center=center_init,
                key=key_init_local,
                n=n_init,
                local_n=local_n_init,
            )

            samples_unsafe, dists_unsafe = sample_neighbors(
                self,
                center=center_unsafe,
                key=key_unsafe_local,
                n=n_unsafe,
                local_n=local_n_unsafe,
            )

            samples_target, dists_target = sample_neighbors(
                self,
                center=center_target,
                key=key_target_local,
                n=n_target,
                local_n=local_n_target,
            )

            samples_decrease, dists = sample_neighbors(
                self,
                center=center_decrease,
                key=key_decrease_local,
                n=n_decrease,
                local_n=local_n_decrease,
            )
            def compute_local_weights(dists_region, n_region):
                if self.local_weight_type == "inverse":
                    weights_region = jnp.where(
                        dists_region < 1e-6,
                        1.0,
                        (n_region * 2.0) / jnp.maximum(dists_region, 1e-3)
                    )
                    weights_region = jnp.clip(weights_region, a_min=0.0, a_max=100.0)

                elif self.local_weight_type == "gaussian":
                    sigma = 0.3 * n_region + 1e-6
                    weights_region = jnp.exp(-0.5 * (dists_region / sigma) ** 2)

                elif self.local_weight_type == "inverse_gaussian":
                    sigma = self.local_gaussian_sigma_scale * n_region + 1e-6

                    weights_inv = jnp.where(
                        dists_region < 1e-6,
                        1.0,
                        (n_region * 2.0) / jnp.maximum(dists_region, 1e-3)
                    )
                    weights_inv = jnp.clip(weights_inv, a_min=0.0, a_max=100.0)

                    weights_gauss = jnp.exp(-0.5 * (dists_region / sigma) ** 2)
                    weights_region = weights_inv * weights_gauss

                else:
                    raise ValueError(f"Unknown local_weight_type: {self.local_weight_type}")

                weights_region = weights_region / (jnp.sum(weights_region) + 1e-6)
                return weights_region

            weights_init = compute_local_weights(dists_init, n_init)
            weights_unsafe = compute_local_weights(dists_unsafe, n_unsafe)
            weights_target = compute_local_weights(dists_target, n_target)
            weights_decrease = compute_local_weights(dists, n_decrease)

            # ------------------------------------------------------------
            # Region masks for local-neighbor sampling.
            # We do NOT dynamically delete samples under jax.jit, because
            # that would change array shapes. Instead, invalid local samples
            # are removed from the loss by multiplying their weights by 0.
            #
            # jax_contains() is equivalent to ~jax_not_contains(), but it is
            # clearer here because these masks mean "sample is still inside
            # the region where its loss is valid".
            # ------------------------------------------------------------
            mask_init_valid = self.env.init_space.jax_contains(samples_init)
            mask_unsafe_valid = self.env.unsafe_space.jax_contains(samples_unsafe)
            mask_target_valid = self.env.target_space.jax_contains(samples_target)

            # Decrease samples should stay in the state space and should not
            # be in the target set, matching the original decrease condition.
            # If you also want to exclude init/unsafe from decrease, uncomment
            # the two extra masks below.
            mask_decrease_in_state = self.env.state_space.jax_contains(samples_decrease)
            mask_decrease_not_target = self.env.target_space.jax_not_contains(samples_decrease)
            # mask_decrease_not_init = self.env.init_space.jax_not_contains(samples_decrease)
            # mask_decrease_not_unsafe = self.env.unsafe_space.jax_not_contains(samples_decrease)
            mask_decrease_valid_region = mask_decrease_in_state * mask_decrease_not_target








            # Small epsilon used in the initial/unsafe loss terms
            EPS_init = 0.1
            EPS_unsafe = 0.1
            EPS_decrease = self.EPS_decrease

            # Compute Lipschitz coefficients.
            lip_certificate, _ = lipschitz_coeff(certificate_params, self.weighted, self.cplip, self.linfty)
            lip_policy, _ = lipschitz_coeff(policy_params, self.weighted, self.cplip, self.linfty)
            lip_policy = jnp.maximum(lip_policy, self.min_lip_policy)

            # Calculate K factor
            if self.linfty and self.split_lip:
                K = lip_certificate * (self.env.lipschitz_f_linfty_A + self.env.lipschitz_f_linfty_B * lip_policy)
            elif self.split_lip:
                K = lip_certificate * (self.env.lipschitz_f_l1_A + self.env.lipschitz_f_l1_B * lip_policy)
            elif self.linfty:
                K = lip_certificate * (self.env.lipschitz_f_linfty * (lip_policy + 1))
            else:
                K = lip_certificate * (self.env.lipschitz_f_l1 * (lip_policy + 1))

            #####

            # Compute certificate values in each of the relevant state sets
            V_init = jnp.ravel(V_state.apply_fn(certificate_params, samples_init))
            V_unsafe = jnp.ravel(V_state.apply_fn(certificate_params, samples_unsafe))
            V_target = jnp.ravel(V_state.apply_fn(certificate_params, samples_target))
            V_decrease = jnp.ravel(V_state.apply_fn(certificate_params, samples_decrease))






            #compute nmin and n max
            # raw_range = Range_state.apply_fn(range_params, samples_decrease)  # (N,2)
            # nmin, nmax = range_from_raw(raw_range, eps=getattr(self, "range_eps", 1e-3))

            # range_reg_weight = getattr(self, "range_reg_weight", 1e-3)
            # nmax_cap = getattr(self, "range_nmax_cap", 1.0)

            # # 惩罚 nmax 太大 + nmin 太大；并且鼓励 nmin 不要无限趋近 0（可选）
            # nmin0 = nmin[0]
            # nmax0 = nmax[0]
            # d = dists[1:]
            # dmin = jnp.min(d)

            # loss_range_reg =(nmax - jnp.max(dists)) ** 2+ ( dmin - nmin) ** 2




            # Loss in each initial/unsafe state
            if self.exp_certificate:
                losses_init = jnp.maximum(0, V_init + EPS_init)
                losses_unsafe = jnp.maximum(0, - jnp.log(1 - probability_bound) - V_unsafe + EPS_unsafe)
            else:
                losses_init = jnp.maximum(0, V_init - 1 + EPS_init)
                losses_unsafe = jnp.maximum(0, 1 / (1 - probability_bound) - V_unsafe + EPS_unsafe)

            # Loss for expected decrease condition
            # ????
            N = samples_decrease.shape[0]
            expDecr_keys = jax.random.split(noise_key, (N, self.N_expectation))
            #?????
            actions = Policy_state.apply_fn(policy_params, samples_decrease)
            V_expected = self.loss_exp_decrease_vmap(V_state, certificate_params, samples_decrease, actions,
                                                     expDecr_keys, probability_bound)

            # Compute E[V(x+)] - V(x), approximated over finite number of noise samples
            if self.exp_certificate:
                Vdiffs = jnp.maximum(V_expected - jnp.minimum(V_decrease, jnp.log(3) - jnp.log(1 - probability_bound)) + mesh_loss * (K + lip_certificate) + EPS_decrease, 0)
            else:
                Vdiffs = jnp.maximum(V_expected - jnp.minimum(V_decrease, 3 / (1 - probability_bound)) + mesh_loss * (K + lip_certificate) + EPS_decrease, 0)

            # Determine in which states the expected decrease condition actually applies
            if self.exp_certificate:
                V_decrease_below_thresh = (jax.lax.stop_gradient(V_decrease - mesh_loss * lip_certificate) <= jnp.log(2) - jnp.log(1 - probability_bound))
            else:
                V_decrease_below_thresh = (jax.lax.stop_gradient(V_decrease - mesh_loss * lip_certificate) <= 2 / (1 - probability_bound))

            # Restrict to the expected decrease samples only
            Vdiffs_trim = mask_decrease_valid_region * V_decrease_below_thresh * jnp.ravel(Vdiffs)

            #####

            if len(counterexamples) > 0:
                V_cx = jnp.ravel(V_state.apply_fn(certificate_params, cx_samples))

                if self.exp_certificate:
                    f_unsafe = -1 / jnp.log(1 - probability_bound)
                    losses_init_cx = jnp.maximum(0, V_cx + EPS_init)
                    losses_unsafe_cx = jnp.maximum(0, - jnp.log(1 - probability_bound) - V_cx + EPS_unsafe)
                    V_decrease_cx_below_thresh = (
                        jax.lax.stop_gradient(V_cx - mesh_loss * lip_certificate)
                        <= jnp.log(2) - jnp.log(1 - probability_bound)
                    )
                else:
                    f_unsafe = (1 - probability_bound)
                    losses_init_cx = jnp.maximum(0, V_cx - 1 + EPS_init)
                    losses_unsafe_cx = jnp.maximum(0, 1 / (1 - probability_bound) - V_cx + EPS_unsafe)
                    V_decrease_cx_below_thresh = (
                        jax.lax.stop_gradient(V_cx - mesh_loss * lip_certificate)
                        <= 2 / (1 - probability_bound)
                    )

                # mean init / unsafe ！！！！
                base_init_weights = weights_init * mask_init_valid
                base_init_mean = jnp.sum(base_init_weights * losses_init, axis=0) / (jnp.sum(base_init_weights, axis=0) + 1e-4)
                base_unsafe_weights = weights_unsafe * mask_unsafe_valid
                base_unsafe_mean = jnp.sum(base_unsafe_weights * losses_unsafe, axis=0) / (jnp.sum(base_unsafe_weights, axis=0) + 1e-4)

                num_cx_init = jnp.sum(cx_bool_init, axis=0)
                num_cx_unsafe = jnp.sum(cx_bool_unsafe, axis=0)

                cx_init_mean = jnp.sum(cx_bool_init * losses_init_cx, axis=0) / (num_cx_init + 1e-4)
                cx_unsafe_mean = jnp.sum(cx_bool_unsafe * losses_unsafe_cx, axis=0) / (num_cx_unsafe + 1e-4)

                loss_init = base_init_mean + cx_init_mean
                loss_unsafe = f_unsafe * (base_unsafe_mean + cx_unsafe_mean)

                # non-dec on counterexamples
                expDecr_keys_cx = jax.random.split(noise_key, (self.batch_size_counterx, self.N_expectation))
                actions_cx = Policy_state.apply_fn(policy_params, cx_samples)
                V_expected_cx = self.loss_exp_decrease_vmap(
                    V_state, certificate_params, cx_samples, actions_cx, expDecr_keys_cx, probability_bound
                )

                if self.exp_certificate:
                    Vdiffs_cx = jnp.maximum(
                        V_expected_cx
                        - jnp.minimum(V_cx, jnp.log(3) - jnp.log(1 - probability_bound))
                        + mesh_loss * (K + lip_certificate)
                        + EPS_decrease,
                        0
                    )
                else:
                    Vdiffs_cx = jnp.maximum(
                        V_expected_cx
                        - jnp.minimum(V_cx, 3 / (1 - probability_bound))
                        + mesh_loss * (K + lip_certificate)
                        + EPS_decrease,
                        0
                    )

                Vdiffs_cx_trim = cx_bool_decrease * V_decrease_cx_below_thresh * jnp.ravel(Vdiffs_cx)

                valid_nondec_base = mask_decrease_valid_region * V_decrease_below_thresh
                valid_nondec_cx = cx_bool_decrease * V_decrease_cx_below_thresh

                # Use the same local weights for base decrease samples as in the no-counterexample branch.
                # Counterexamples are not generated from the local sampler, so they remain unweighted.
                weights_valid_base = weights_decrease * valid_nondec_base
                denom_nondec = jnp.sum(weights_valid_base, axis=0) + jnp.sum(valid_nondec_cx, axis=0) + 1e-4

                if self.loss_decr_squared:
                    loss_nondec = expDecr_multiplier * (
                        jnp.sqrt(
                            (
                                jnp.sum(weights_valid_base * Vdiffs_trim ** 2, axis=0)
                                + jnp.sum(Vdiffs_cx_trim ** 2, axis=0)
                            )
                            / denom_nondec
                            + 1e-4
                        ) - 1e-2
                    )
                else:
                    loss_nondec = expDecr_multiplier * (
                        (
                            jnp.sum(weights_valid_base * Vdiffs_trim, axis=0)
                            + jnp.sum(Vdiffs_cx_trim, axis=0)
                        )
                        / denom_nondec
                    )

            else:
                if self.exp_certificate:
                    f_unsafe = -1 / jnp.log(1 - probability_bound)
                else:
                    f_unsafe = (1 - probability_bound)

                # Weighted base init / unsafe, consistent with the counterexample branch.
                weights_init_valid = weights_init * mask_init_valid
                loss_init = jnp.sum(weights_init_valid * losses_init, axis=0) / (jnp.sum(weights_init_valid, axis=0) + 1e-4)
                weights_unsafe_valid = weights_unsafe * mask_unsafe_valid
                loss_unsafe = f_unsafe * jnp.sum(weights_unsafe_valid * losses_unsafe, axis=0) / (jnp.sum(weights_unsafe_valid, axis=0) + 1e-4)

                valid_nondec = mask_decrease_valid_region * V_decrease_below_thresh
                weights_valid = weights_decrease * valid_nondec

                if self.loss_decr_squared:
                    loss_nondec = expDecr_multiplier * (
                        jnp.sqrt(
                            jnp.sum(weights_valid*Vdiffs_trim ** 2, axis=0) / (jnp.sum(weights_valid, axis=0)+ 1e-4) + 1e-4
                        ) - 1e-3
                    )
                else:
                    loss_nondec = expDecr_multiplier * (
                        jnp.sum(Vdiffs_trim * weights_valid, axis=0) / (jnp.sum(weights_valid, axis=0) + 1e-4)
                    )
            #####

            # Loss to promote low Lipschitz constant
            loss_lipschitz = self.lambda_lipschitz * (jnp.maximum(lip_certificate - self.max_lip_certificate, 0) +
                                                      jnp.maximum(lip_policy - self.max_lip_policy, 0))

            # Auxiliary losses
            # Auxiliary losses should also ignore local samples that left
            # their intended regions. Since each center is valid, every mask
            # should contain at least one True entry.
            big = jnp.array(1e6, dtype=V_target.dtype)
            V_target_valid_min = jnp.min(jnp.where(mask_target_valid, V_target, big), axis=0)
            V_init_valid_min = jnp.min(jnp.where(mask_init_valid, V_init, big), axis=0)
            V_unsafe_valid_min = jnp.min(jnp.where(mask_unsafe_valid, V_unsafe, big), axis=0)

            loss_min_target = jnp.maximum(0, V_target_valid_min - self.glob_min)
            loss_min_init = jnp.maximum(0, V_target_valid_min - V_init_valid_min)
            loss_min_unsafe = jnp.maximum(0, V_target_valid_min - V_unsafe_valid_min)
            loss_aux = self.auxiliary_loss * (loss_min_target + loss_min_init + loss_min_unsafe)
            # range_loss_weight = getattr(self, "range_loss_weight", 1.0)
            # loss_range = 100000*loss_range_reg
            # loss_range = range_loss_weight * loss_range_reg


            #    L_n = alpha / (n + eps) + beta * n^2
            alpha_n = 1e-3
            beta_n = 1e-3

            n_all = jnp.stack([
                n_init,
                n_unsafe,
                n_target,
                n_decrease,
            ])

            loss_n_reg = jnp.mean(
                alpha_n / (n_all + 1e-6) + beta_n * (n_all ** 2)
            )

            # 2) weights 熵正则：防止权重过度集中到少数点
            #    H(w) = - sum_i w_i log w_i
            #    在 loss 里加入 negative entropy => sum_i w_i log w_i
            lambda_entropy = 1e-4
            weights_init_entropy = weights_init * mask_init_valid
            weights_init_entropy = weights_init_entropy / (jnp.sum(weights_init_entropy) + 1e-6)
            weights_unsafe_entropy = weights_unsafe * mask_unsafe_valid
            weights_unsafe_entropy = weights_unsafe_entropy / (jnp.sum(weights_unsafe_entropy) + 1e-6)
            weights_target_entropy = weights_target * mask_target_valid
            weights_target_entropy = weights_target_entropy / (jnp.sum(weights_target_entropy) + 1e-6)
            weights_decrease_entropy = weights_decrease * mask_decrease_valid_region
            weights_decrease_entropy = weights_decrease_entropy / (jnp.sum(weights_decrease_entropy) + 1e-6)

            loss_entropy = lambda_entropy * (
                jnp.sum(weights_init_entropy * jnp.log(weights_init_entropy + 1e-8))
                + jnp.sum(weights_unsafe_entropy * jnp.log(weights_unsafe_entropy + 1e-8))
                + jnp.sum(weights_target_entropy * jnp.log(weights_target_entropy + 1e-8))
                + jnp.sum(weights_decrease_entropy * jnp.log(weights_decrease_entropy + 1e-8))
            )
            # 3) 局部 V 平滑项：鼓励邻域内 V(x_i) 不要相对中心点变化过猛
            #    L_V = mean_i w_i * (V(x_i) - V(center))^2
            # lambda_v_smooth = 1e-4
            # loss_v_smooth = lambda_v_smooth * jnp.sum(weights * (V_decrease - V_center) ** 2)
            # loss_range_reg = loss_n_reg + loss_entropy + loss_v_smooth
            loss_range_reg = loss_n_reg + loss_entropy 
            






            # Define total loss
            loss_total = loss_init + loss_unsafe + loss_nondec + loss_lipschitz + loss_aux+loss_range_reg

            infos = {
                '0. total': loss_total,
                '1. init': loss_init,
                '2. unsafe': loss_unsafe,
                '3. nondec': loss_nondec,
                '4. loss_lipschitz': loss_lipschitz,
            }

            if self.auxiliary_loss > 0:
                infos['8. loss auxiliary'] = loss_aux
            infos["9. n_de"] = n_decrease
            infos["9.1 mean_dist"] = jnp.mean(dists)
            infos["9.2 max_dist"] = jnp.max(dists)
            infos["9.6 max_weight_init"] = jnp.max(weights_init)
            infos["9.7 max_weight_unsafe"] = jnp.max(weights_unsafe)
            infos["9.8 max_weight_target"] = jnp.max(weights_target)
            infos["9.9 max_weight_decrease"] = jnp.max(weights_decrease)
            infos["9.10 valid_init_frac"] = jnp.mean(mask_init_valid)
            infos["9.11 valid_unsafe_frac"] = jnp.mean(mask_unsafe_valid)
            infos["9.12 valid_target_frac"] = jnp.mean(mask_target_valid)
            infos["9.13 valid_decrease_frac"] = jnp.mean(mask_decrease_valid_region)

            # infos["6. loss_range"] = loss_range
            # infos["6.1 range_reg"] = loss_range_reg
            # infos["6.2 nmin_mean"] = jnp.mean(nmin)
            # infos["6.3 nmax_mean"] = jnp.mean(nmax)

            return loss_total, infos

        # Compute gradients
        loss_grad_fun = jax.value_and_grad(loss_fun, argnums=(0, 1,2), has_aux=True)
        (loss_val, infos), (V_grads, Policy_grads, Range_grads) = loss_grad_fun(V_state.params, Policy_state.params, Range_state.params)
        # 重新计算当前参数下的 center / n / samples_decrease，仅用于记录
# ------------------------------------------------------------
# Recompute samples only for logging/debugging.
# These variables are outside loss_fun, so we cannot directly
# use samples_init / samples_unsafe / samples_target from loss_fun.
# ------------------------------------------------------------

        def compute_radius_log(center_log):
            V_center_log = jnp.ravel(
                V_state.apply_fn(V_state.params, center_log[None, :])
            )[0]

            range_input_log = jnp.concatenate(
                [center_log, jnp.array([V_center_log])],
                axis=0,
            )[None, :]

            raw_n_log = Range_state.apply_fn(
                Range_state.params,
                range_input_log,
            )[0, 0]

            n_log = jax.nn.softplus(raw_n_log) + 1e-3
            return n_log


        n_init_log = compute_radius_log(center_init)
        n_unsafe_log = compute_radius_log(center_unsafe)
        n_target_log = compute_radius_log(center_target)
        n_decrease_log = compute_radius_log(center_decrease)


        N_init_log = total_num_samples(self.num_samples_init)
        N_unsafe_log = total_num_samples(self.num_samples_unsafe)
        N_target_log = total_num_samples(self.num_samples_target)
        N_decrease_log_total = total_num_samples(self.num_samples_decrease)

        local_n_init_log = max(N_init_log - 1, 0)
        local_n_unsafe_log = max(N_unsafe_log - 1, 0)
        local_n_target_log = max(N_target_log - 1, 0)
        local_n_decrease_log = max(N_decrease_log_total - 1, 0)


        key_init_log, key_unsafe_log, key_target_log, key_decrease_log = jax.random.split(
            perturbation_key,
            4,
        )

        samples_init_log, dists_init_log = sample_neighbors(
            self,
            center=center_init,
            key=key_init_log,
            n=n_init_log,
            local_n=local_n_init_log,
        )

        samples_unsafe_log, dists_unsafe_log = sample_neighbors(
            self,
            center=center_unsafe,
            key=key_unsafe_log,
            n=n_unsafe_log,
            local_n=local_n_unsafe_log,
        )

        samples_target_log, dists_target_log = sample_neighbors(
            self,
            center=center_target,
            key=key_target_log,
            n=n_target_log,
            local_n=local_n_target_log,
        )

        samples_decrease_log, dists_decrease_log = sample_neighbors(
            self,
            center=center_decrease,
            key=key_decrease_log,
            n=n_decrease_log,
            local_n=local_n_decrease_log,
        )

        samples_decrease_bool_not_targetUnsafe_log = (
            self.env.state_space.jax_contains(samples_decrease_log)
            * self.env.target_space.jax_not_contains(samples_decrease_log)
        )

        samples_in_batch = {
            'init': samples_init_log,
            'target': samples_target_log,
            'unsafe': samples_unsafe_log,
            'decrease': samples_decrease_log,
            'decrease_not_in_target': samples_decrease_bool_not_targetUnsafe_log,
            'counterx': cx_samples,
            'counterx_init': cx_bool_init,
            'counterx_unsafe': cx_bool_unsafe,
            'counterx_decrease': cx_bool_decrease,
        }

        return V_grads, Policy_grads, Range_grads, infos, key, samples_in_batch

    def debug_train_step(self, args, samples_in_batch, iteration):

        '''
        Debug function for the training. 

        :param args: Command line arguments given. 
        :param samples_in_batch: Dictionary, giving the samples used in the batch per category. 
        :param iteration: Number of the CEGIS iteration.
        '''

        samples_in_batch['decrease'] = samples_in_batch['decrease'][samples_in_batch['decrease_not_in_target']]

        print('Samples used in last train steps:')
        print(f"- # init samples: {len(samples_in_batch['init'])}")
        print(f"- # unsafe samples: {len(samples_in_batch['unsafe'])}")
        print(f"- # target samples: {len(samples_in_batch['target'])}")
        print(f"- # decrease samples: {len(samples_in_batch['decrease'])}")
        print(f"- # counterexamples: {len(samples_in_batch['counterx'])}")
        print(f"-- # cx init: {len(samples_in_batch['counterx'][samples_in_batch['counterx_init']])}")
        print(f"-- # cx unsafe: {len(samples_in_batch['counterx'][samples_in_batch['counterx_unsafe']])}")
        print(f"-- # cx decrease: {len(samples_in_batch['counterx'][samples_in_batch['counterx_decrease']])}")

        # Plot samples used in batch
        for s in ['init', 'unsafe', 'target', 'decrease', 'counterx']:
            filename = f"plots/{args.start_datetime}_train_debug_{str(s)}_iteration={iteration}"
            plot_dataset(self.env, additional_data=np.array(samples_in_batch[s]), folder=args.cwd, filename=filename)

        for s in ['counterx_init', 'counterx_unsafe', 'counterx_decrease']:
            filename = f"plots/{args.start_datetime}_train_debug_{str(s)}_iteration={iteration}"
            idxs = samples_in_batch[s]
            plot_dataset(self.env, additional_data=np.array(samples_in_batch['counterx'])[idxs], folder=args.cwd,
                         filename=filename)
            

            