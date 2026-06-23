from functools import partial
'''
多点完结
'''
import jax
import numpy as np
from flax.training.train_state import TrainState
from jax import numpy as jnp

from core.commons import MultiRectangularSet
from core.jax_utils import lipschitz_coeff
from core.plot import plot_dataset


class Learner:
    '''
    Learner variant:

    - Keep the original volume-based sample allocation:
        self.num_samples_init
        self.num_samples_unsafe
        self.num_samples_target
        self.num_samples_decrease

    - But these numbers are used as the number of CENTERS, not the final number of samples.

    - Around every center, sample `local_samples_per_center` local neighbor points.
      Therefore each region has:

          final_num_region = N_region * (1 + local_samples_per_center)

    - Each center gets its own radius predicted by Range_state:

          n_i = Range(center_i, V(center_i))

    - Losses are computed with distance-based weights and masks:
        init:     weighted mean over points still in init_space
        unsafe:   weighted mean over points still in unsafe_space
        decrease: weighted mean over points in state_space and outside target_space
    '''

    def __init__(self, env, args):
        self.env = env
        self.linfty = False

        # Copy some arguments
        self.auxiliary_loss = args.auxiliary_loss
        self.lambda_lipschitz = args.loss_lipschitz_lambda
        self.max_lip_certificate = args.loss_lipschitz_certificate
        self.max_lip_policy = args.loss_lipschitz_policy
        self.weighted = args.weighted
        self.cplip = args.cplip
        self.split_lip = args.split_lip
        self.min_lip_policy = args.min_lip_policy_loss
        self.exp_certificate = args.exp_certificate
        self.loss_decr_squared = args.loss_decr_squared
        self.loss_decr_max = args.loss_decr_max
        self.EPS_decrease = args.eps_decrease

        # Local sampling settings
        self.local_samples_per_center = args.local_samples_per_center
        self.local_weight_type = args.local_weight_type
        self.local_gaussian_sigma_scale = args.local_gaussian_sigma_scale
  
        self.local_distance_type = args.distance_type


        # Radius output range. Bounded radius is much more stable than softplus-only.
        self.local_radius_min = float(getattr(args, "local_radius_min", 0.0002))
        self.local_radius_max = float(getattr(args, "local_radius_max", 0.9))

        # Optional regularization. Default 0: do not force n to a problematic value.
        self.range_reg_weight = float(getattr(args, "range_reg_weight", 0.0))
        self.range_target = float(getattr(args, "range_target", 0.1))
        self.weight_entropy_lambda = float(getattr(args, "weight_entropy_lambda", 0.0))

        # Set batch sizes
        self.batch_size_total = int(args.batch_size)
        self.batch_size_base = int(args.batch_size * (1 - args.counterx_fraction))
        self.batch_size_counterx = int(args.batch_size * args.counterx_fraction)

        # Calculate the number of samples for each region type (without counterexamples)
        # In this variant these numbers are used as CENTER counts.
        MIN_SAMPLES = max(int(args.min_fraction_samples_per_region * self.batch_size_base), 1)

        totvol = env.state_space.volume
        if isinstance(env.init_space, MultiRectangularSet):
            rel_vols = np.array([Set.volume / totvol for Set in env.init_space.sets])
            self.num_samples_init = tuple(np.maximum(np.ceil(rel_vols * self.batch_size_base), MIN_SAMPLES).astype(int))
        else:
            self.num_samples_init = np.maximum(
                MIN_SAMPLES,
                np.ceil(env.init_space.volume / totvol * self.batch_size_base),
            ).astype(int)

        if isinstance(env.unsafe_space, MultiRectangularSet):
            rel_vols = np.array([Set.volume / totvol for Set in env.unsafe_space.sets])
            self.num_samples_unsafe = tuple(
                np.maximum(MIN_SAMPLES, np.ceil(rel_vols * self.batch_size_base)).astype(int)
            )
        else:
            self.num_samples_unsafe = np.maximum(
                np.ceil(env.unsafe_space.volume / totvol * self.batch_size_base),
                MIN_SAMPLES,
            ).astype(int)

        if isinstance(env.target_space, MultiRectangularSet):
            rel_vols = np.array([Set.volume / totvol for Set in env.target_space.sets])
            self.num_samples_target = tuple(
                np.maximum(np.ceil(rel_vols * self.batch_size_base), MIN_SAMPLES).astype(int)
            )
        else:
            self.num_samples_target = np.maximum(
                MIN_SAMPLES,
                np.ceil(env.target_space.volume / totvol * self.batch_size_base),
            ).astype(int)

        self.num_samples_decrease = np.maximum(
            self.batch_size_base
            - np.sum(self.num_samples_init)
            - np.sum(self.num_samples_unsafe)
            - np.sum(self.num_samples_target),
            1,
        ).astype(int)

        if not args.silent:
            multiplier = 1 + self.local_samples_per_center
            print(f'- Num. base train samples per batch: {self.batch_size_base}')
            print(f'-- Initial centers: {self.num_samples_init}; final samples x{multiplier}')
            print(f'-- Unsafe centers: {self.num_samples_unsafe}; final samples x{multiplier}')
            print(f'-- Target centers: {self.num_samples_target}; final samples x{multiplier}')
            print(f'-- Expected decrease centers: {self.num_samples_decrease}; final samples x{multiplier}')
            print(f'-- Local samples per center: {self.local_samples_per_center}')
            print(f'- Num. counterexamples per batch: {self.batch_size_counterx}\n')

        if self.lambda_lipschitz > 0 and not args.silent:
            print('- Learner setting: Enable Lipschitz loss')
            print(f'--- For certificate up to: {self.max_lip_certificate:.3f}')
            print(f'--- For policy up to: {self.max_lip_policy:.3f}')

        self.glob_min = 0.1
        self.N_expectation = 1

        self.loss_exp_decrease_vmap = jax.vmap(
            self.loss_exp_decrease,
            in_axes=(None, None, 0, 0, 0, None),
            out_axes=0,
        )

    def loss_exp_decrease(self, V_state, V_params, x, u, noise_key, probability_bound):
        state_new, noise_key = self.env.vstep_noise_batch(x, noise_key, u)

        if self.exp_certificate:
            V_expected = jnp.log(
                jnp.mean(
                    jnp.exp(
                        jnp.minimum(
                            V_state.apply_fn(V_params, state_new),
                            jnp.log(2) - jnp.log(1 - probability_bound),
                        )
                    )
                )
            )
        else:
            V_expected = jnp.mean(
                jnp.minimum(V_state.apply_fn(V_params, state_new), 2 / (1 - probability_bound))
            )

        return V_expected

    @partial(jax.jit, static_argnums=(0,))
    def train_step(
        self,
        key: jax.Array,
        V_state: TrainState,
        Policy_state: TrainState,
        counterexamples,
        mesh_loss,
        probability_bound,
        expDecr_multiplier,
        Range_state: TrainState,
    ):
        key, cx_key, init_key, unsafe_key, target_key, decrease_key, noise_key, perturbation_key = \
            jax.random.split(key, 8)

        if len(counterexamples) > 0:
            cx = jax.random.choice(
                cx_key,
                counterexamples,
                shape=(self.batch_size_counterx,),
                replace=False,
            )
            cx_samples = cx[:, :-3]
            cx_bool_init = cx[:, -2] > 0
            cx_bool_unsafe = cx[:, -1] > 0
            cx_bool_decrease = cx[:, -3] > 0
        else:
            cx_samples = cx_bool_init = cx_bool_unsafe = cx_bool_decrease = False

        def total_num_samples(num_samples):
            return int(np.sum(num_samples))

        # N_init_centers = total_num_samples(self.num_samples_init)
        # N_unsafe_centers = total_num_samples(self.num_samples_unsafe)
        # N_target_centers = total_num_samples(self.num_samples_target)
        # N_decrease_centers = total_num_samples(self.num_samples_decrease)

        # # Original volume-based samples are used as centers.
        # centers_init = self.env.init_space.sample(rng=init_key, N=self.num_samples_init)
        # centers_unsafe = self.env.unsafe_space.sample(rng=unsafe_key, N=self.num_samples_unsafe)
        # centers_target = self.env.target_space.sample(rng=target_key, N=self.num_samples_target)
        # centers_decrease = self.env.state_space.sample(rng=decrease_key, N=self.num_samples_decrease)


        L = self.local_samples_per_center + 1

        if isinstance(self.num_samples_init, tuple):

            N_init = tuple(np.maximum(np.array(self.num_samples_init) // L, 1).astype(int))

        else:

            N_init = max(int(self.num_samples_init // L), 1)

        if isinstance(self.num_samples_unsafe, tuple):

            N_unsafe = tuple(np.maximum(np.array(self.num_samples_unsafe) // L, 1).astype(int))

        else:

            N_unsafe = max(int(self.num_samples_unsafe // L), 1)

        if isinstance(self.num_samples_target, tuple):

            N_target = tuple(np.maximum(np.array(self.num_samples_target) // L, 1).astype(int))

        else:

            N_target = max(int(self.num_samples_target // L), 1)

        N_decrease = max(int(self.num_samples_decrease // L), 1)
        centers_init = self.env.init_space.sample(rng=init_key, N=N_init)
        centers_unsafe = self.env.unsafe_space.sample(rng=unsafe_key, N=N_unsafe)
        centers_target = self.env.target_space.sample(rng=target_key, N=N_target)
        centers_decrease = self.env.state_space.sample(rng=decrease_key, N=N_decrease)
        N_init_centers = centers_init.shape[0]
        N_unsafe_centers = centers_unsafe.shape[0]
        N_target_centers = centers_target.shape[0]
        N_decrease_centers = centers_decrease.shape[0]



        def compute_radii(V_params, range_params, centers):
            V_centers = jnp.ravel(V_state.apply_fn(V_params, centers))
            radius_inputs = jnp.concatenate([centers, V_centers[:, None]], axis=1)
            raw_n = Range_state.apply_fn(range_params, radius_inputs)
            raw_n = jnp.ravel(raw_n)

            # Bounded radius: prevents n from drifting to ~0.8 and causing most samples to be masked out.
            # jax.debug.print(
            #     "raw_n mean={m}, min={mn}, max={mx}",
            #     m=jnp.mean(raw_n),
            #     mn=jnp.min(raw_n),
            #     mx=jnp.max(raw_n),

            # )
            n = self.local_radius_min + (self.local_radius_max - self.local_radius_min) * jax.nn.sigmoid(raw_n)
            return n, V_centers

        def sample_neighbors_for_centers(centers, keys, n_centers):
            """
            centers: (C, D)
            keys:    (C,) PRNG keys returned by jax.random.split.
                     Do not index keys as keys[:, 0]; with typed JAX keys this
                     produces scalar invalid key data.
            n:       (C,)

            Return flattened samples:
                samples:      (C * (1 + local_samples_per_center), D)
                dists:        (C * (1 + local_samples_per_center),)
                n_per_sample: (C * (1 + local_samples_per_center),)
            """
            C = centers.shape[0]
            D = self.env.state_dim
            L = self.local_samples_per_center
            # print("****L****",L)

            if L == 0:
                samples = centers
                dists = jnp.zeros((C,), dtype=centers.dtype)
                n_per_sample = n_centers
                return samples, dists, n_per_sample

            if self.local_distance_type == "l2":
                def sample_l2(k, n):
                    key_dir, key_r = jax.random.split(k, 2)
                    dirs_i = jax.random.normal(key_dir, shape=(L, D))
                    dirs_i = dirs_i / (jnp.linalg.norm(dirs_i, axis=1, keepdims=True) + 1e-8)
                    u_i = jax.random.uniform(key_r, shape=(L, 1), minval=0.0, maxval=1.0)
                    r_i = n * (u_i ** (1.0 / D))
                    offsets_i = dirs_i * r_i
                    dists_i = r_i[:, 0]
                    return offsets_i, dists_i

                offsets, dists_only = jax.vmap(sample_l2)(keys, n_centers)

            elif self.local_distance_type == "inf":
                def sample_inf(k, n):
                    offsets_i = jax.random.uniform(k, shape=(L, D), minval=-n, maxval=n)
                    dists_i = jnp.max(jnp.abs(offsets_i), axis=1)
                    return offsets_i, dists_i

                offsets, dists_only = jax.vmap(sample_inf)(keys, n_centers)

            elif self.local_distance_type == "l1":
                # Rejection-style l1 sampling with fixed oversampling. If not enough valid candidates,
                # fill_value=0 duplicates the first candidate; this keeps shapes static under jit.
                oversample = max(L * 6, 1)

                def sample_l1(k, n):
                    cand = jax.random.uniform(k, shape=(oversample, D), minval=-n, maxval=n)
                    cand_d = jnp.sum(jnp.abs(cand), axis=1)
                    mask = cand_d <= n
                    idx = jnp.where(mask, size=L, fill_value=0)[0]
                    offsets_i = cand[idx]
                    dists_i = jnp.sum(jnp.abs(offsets_i), axis=1)
                    return offsets_i, dists_i

                offsets, dists_only = jax.vmap(sample_l1)(keys, n_centers)

            else:
                raise ValueError(f"Unknown local_distance_type: {self.local_distance_type}")

            neighbors = centers[:, None, :] + offsets
            samples_grouped = jnp.concatenate([centers[:, None, :], neighbors], axis=1)
            dists_grouped = jnp.concatenate(
                [jnp.zeros((C, 1), dtype=centers.dtype), dists_only],
                axis=1,
            )
            n_grouped = jnp.broadcast_to(n_centers[:, None], dists_grouped.shape)

            samples = samples_grouped.reshape((C * (L + 1), D))
            dists = dists_grouped.reshape((C * (L + 1),))
            n_per_sample = n_grouped.reshape((C * (L + 1),))

            return samples, dists, n_per_sample

        def compute_local_weights(dists_region, n_per_sample_region):
            if self.local_weight_type == "inverse":
                weights_region = jnp.where(
                    dists_region < 1e-6,
                    1.0,
                    (n_per_sample_region * 2.0) / jnp.maximum(dists_region, 1e-3),
                )
                weights_region = jnp.clip(weights_region, a_min=0.0, a_max=100.0)

            elif self.local_weight_type == "gaussian":
                sigma = 0.3 * n_per_sample_region + 1e-6
                weights_region = jnp.exp(-0.5 * (dists_region / sigma) ** 2)

            elif self.local_weight_type == "inverse_gaussian":
                sigma = self.local_gaussian_sigma_scale * n_per_sample_region + 1e-6
                weights_inv = jnp.where(
                    dists_region < 1e-6,
                    1.0,
                    (n_per_sample_region * 2.0) / jnp.maximum(dists_region, 1e-3),
                )
                weights_inv = jnp.clip(weights_inv, a_min=0.0, a_max=100.0)
                weights_gauss = jnp.exp(-0.5 * (dists_region / sigma) ** 2)
                weights_region = weights_inv * weights_gauss

            elif self.local_weight_type == "uniform":
                weights_region = jnp.ones_like(dists_region)

            else:
                raise ValueError(f"Unknown local_weight_type: {self.local_weight_type}")

            return weights_region / (jnp.sum(weights_region) + 1e-6)

        def masked_weighted_mean(losses, weights, mask):
            weights_valid = weights * mask
            return jnp.sum(weights_valid * losses, axis=0) / (jnp.sum(weights_valid, axis=0) + 1e-4)

        def masked_min(values, mask):
            # Ignore invalid points by replacing them with a large value.
            return jnp.min(jnp.where(mask, values, 1e6), axis=0)

        def loss_fun(certificate_params, policy_params, range_params):
            EPS_init = 0.1
            EPS_unsafe = 0.1
            EPS_decrease = self.EPS_decrease

            lip_certificate, _ = lipschitz_coeff(certificate_params, self.weighted, self.cplip, self.linfty)
            lip_policy, _ = lipschitz_coeff(policy_params, self.weighted, self.cplip, self.linfty)
            lip_policy = jnp.maximum(lip_policy, self.min_lip_policy)

            if self.linfty and self.split_lip:
                K = lip_certificate * (self.env.lipschitz_f_linfty_A + self.env.lipschitz_f_linfty_B * lip_policy)
            elif self.split_lip:
                K = lip_certificate * (self.env.lipschitz_f_l1_A + self.env.lipschitz_f_l1_B * lip_policy)
            elif self.linfty:
                K = lip_certificate * (self.env.lipschitz_f_linfty * (lip_policy + 1))
            else:
                K = lip_certificate * (self.env.lipschitz_f_l1 * (lip_policy + 1))

            # One radius per center.
            n_init_centers, V_centers_init = compute_radii(certificate_params, range_params, centers_init)
            n_unsafe_centers, V_centers_unsafe = compute_radii(certificate_params, range_params, centers_unsafe)
            n_target_centers, V_centers_target = compute_radii(certificate_params, range_params, centers_target)
            n_decrease_centers, V_centers_decrease = compute_radii(certificate_params, range_params, centers_decrease)

            # Sample local neighbors around every center.
            key_init_local, key_unsafe_local, key_target_local, key_decrease_local = jax.random.split(
                perturbation_key, 4
            )
            keys_init = jax.random.split(key_init_local, N_init_centers)
            keys_unsafe = jax.random.split(key_unsafe_local, N_unsafe_centers)
            keys_target = jax.random.split(key_target_local, N_target_centers)
            keys_decrease = jax.random.split(key_decrease_local, N_decrease_centers)

            samples_init, dists_init, n_per_init = sample_neighbors_for_centers(
                centers_init, keys_init, n_init_centers
            )
            samples_unsafe, dists_unsafe, n_per_unsafe = sample_neighbors_for_centers(
                centers_unsafe, keys_unsafe, n_unsafe_centers
            )
            samples_target, dists_target, n_per_target = sample_neighbors_for_centers(
                centers_target, keys_target, n_target_centers
            )
            samples_decrease, dists_decrease, n_per_decrease = sample_neighbors_for_centers(
                centers_decrease, keys_decrease, n_decrease_centers
            )

            weights_init = compute_local_weights(dists_init, n_per_init)
            weights_unsafe = compute_local_weights(dists_unsafe, n_per_unsafe)
            weights_target = compute_local_weights(dists_target, n_per_target)
            weights_decrease = compute_local_weights(dists_decrease, n_per_decrease)

            # Masks: keep only local samples that still belong to the intended region.
            mask_init_valid = self.env.init_space.jax_contains(samples_init)
            mask_unsafe_valid = self.env.unsafe_space.jax_contains(samples_unsafe)
            mask_target_valid = self.env.target_space.jax_contains(samples_target)
            mask_decrease_valid = self.env.state_space.jax_contains(samples_decrease) * self.env.target_space.jax_not_contains(samples_decrease)

            V_init = jnp.ravel(V_state.apply_fn(certificate_params, samples_init))
            V_unsafe = jnp.ravel(V_state.apply_fn(certificate_params, samples_unsafe))
            V_target = jnp.ravel(V_state.apply_fn(certificate_params, samples_target))
            V_decrease = jnp.ravel(V_state.apply_fn(certificate_params, samples_decrease))

            if self.exp_certificate:
                losses_init = jnp.maximum(0, V_init + EPS_init)
                losses_unsafe = jnp.maximum(0, -jnp.log(1 - probability_bound) - V_unsafe + EPS_unsafe)
            else:
                losses_init = jnp.maximum(0, V_init - 1 + EPS_init)
                losses_unsafe = jnp.maximum(0, 1 / (1 - probability_bound) - V_unsafe + EPS_unsafe)

            N_decrease_total = samples_decrease.shape[0]
            expDecr_keys = jax.random.split(noise_key, (N_decrease_total, self.N_expectation))
            actions = Policy_state.apply_fn(policy_params, samples_decrease)
            V_expected = self.loss_exp_decrease_vmap(
                V_state,
                certificate_params,
                samples_decrease,
                actions,
                expDecr_keys,
                probability_bound,
            )

            if self.exp_certificate:
                Vdiffs = jnp.maximum(
                    V_expected
                    - jnp.minimum(V_decrease, jnp.log(3) - jnp.log(1 - probability_bound))
                    + mesh_loss * (K + lip_certificate)
                    + EPS_decrease,
                    0,
                )
                V_decrease_below_thresh = (
                    jax.lax.stop_gradient(V_decrease - mesh_loss * lip_certificate)
                    <= jnp.log(2) - jnp.log(1 - probability_bound)
                )
            else:
                Vdiffs = jnp.maximum(
                    V_expected
                    - jnp.minimum(V_decrease, 3 / (1 - probability_bound))
                    + mesh_loss * (K + lip_certificate)
                    + EPS_decrease,
                    0,
                )
                V_decrease_below_thresh = (
                    jax.lax.stop_gradient(V_decrease - mesh_loss * lip_certificate)
                    <= 2 / (1 - probability_bound)
                )

            valid_nondec_base = mask_decrease_valid * V_decrease_below_thresh
            weights_valid_decrease = weights_decrease * valid_nondec_base
            Vdiffs_trim = valid_nondec_base * jnp.ravel(Vdiffs)

            if len(counterexamples) > 0:
                V_cx = jnp.ravel(V_state.apply_fn(certificate_params, cx_samples))

                if self.exp_certificate:
                    f_unsafe = -1 / jnp.log(1 - probability_bound)
                    losses_init_cx = jnp.maximum(0, V_cx + EPS_init)
                    losses_unsafe_cx = jnp.maximum(0, -jnp.log(1 - probability_bound) - V_cx + EPS_unsafe)
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

                base_init_loss = masked_weighted_mean(losses_init, weights_init, mask_init_valid)
                base_unsafe_loss = masked_weighted_mean(losses_unsafe, weights_unsafe, mask_unsafe_valid)

                num_cx_init = jnp.sum(cx_bool_init, axis=0)
                num_cx_unsafe = jnp.sum(cx_bool_unsafe, axis=0)
                cx_init_loss = jnp.sum(cx_bool_init * losses_init_cx, axis=0) / (num_cx_init + 1e-4)
                cx_unsafe_loss = jnp.sum(cx_bool_unsafe * losses_unsafe_cx, axis=0) / (num_cx_unsafe + 1e-4)

                loss_init = base_init_loss + cx_init_loss
                loss_unsafe = f_unsafe * (base_unsafe_loss + cx_unsafe_loss)

                expDecr_keys_cx = jax.random.split(noise_key, (self.batch_size_counterx, self.N_expectation))
                actions_cx = Policy_state.apply_fn(policy_params, cx_samples)
                V_expected_cx = self.loss_exp_decrease_vmap(
                    V_state,
                    certificate_params,
                    cx_samples,
                    actions_cx,
                    expDecr_keys_cx,
                    probability_bound,
                )

                if self.exp_certificate:
                    Vdiffs_cx = jnp.maximum(
                        V_expected_cx
                        - jnp.minimum(V_cx, jnp.log(3) - jnp.log(1 - probability_bound))
                        + mesh_loss * (K + lip_certificate)
                        + EPS_decrease,
                        0,
                    )
                else:
                    Vdiffs_cx = jnp.maximum(
                        V_expected_cx
                        - jnp.minimum(V_cx, 3 / (1 - probability_bound))
                        + mesh_loss * (K + lip_certificate)
                        + EPS_decrease,
                        0,
                    )

                valid_nondec_cx = cx_bool_decrease * V_decrease_cx_below_thresh
                Vdiffs_cx_trim = valid_nondec_cx * jnp.ravel(Vdiffs_cx)

                denom_nondec = jnp.sum(weights_valid_decrease, axis=0) + jnp.sum(valid_nondec_cx, axis=0) + 1e-4
                if self.loss_decr_squared:
                    loss_exp_decrease_mean = expDecr_multiplier * (
                        jnp.sqrt(
                            (
                                jnp.sum(weights_valid_decrease * Vdiffs_trim ** 2, axis=0)
                                + jnp.sum(Vdiffs_cx_trim ** 2, axis=0)
                            )
                            / denom_nondec
                            + 1e-4
                        )
                        - 1e-2
                    )
                else:
                    loss_exp_decrease_mean = expDecr_multiplier * (
                        (
                            jnp.sum(weights_valid_decrease * Vdiffs_trim, axis=0)
                            + jnp.sum(Vdiffs_cx_trim, axis=0)
                        )
                        / denom_nondec
                    )

                if self.loss_decr_max:
                    loss_exp_decrease_max = jnp.maximum(jnp.max(Vdiffs_trim), jnp.max(Vdiffs_cx_trim))
                else:
                    loss_exp_decrease_max = 0

            else:
                if self.exp_certificate:
                    f_unsafe = -1 / jnp.log(1 - probability_bound)
                else:
                    f_unsafe = (1 - probability_bound)

                loss_init = masked_weighted_mean(losses_init, weights_init, mask_init_valid)
                loss_unsafe = f_unsafe * masked_weighted_mean(losses_unsafe, weights_unsafe, mask_unsafe_valid)

                denom_nondec = jnp.sum(weights_valid_decrease, axis=0) + 1e-4
                if self.loss_decr_squared:
                    loss_exp_decrease_mean = expDecr_multiplier * (
                        jnp.sqrt(
                            jnp.sum(weights_valid_decrease * Vdiffs_trim ** 2, axis=0)
                            / denom_nondec
                            + 1e-4
                        )
                        - 1e-3
                    )
                else:
                    loss_exp_decrease_mean = expDecr_multiplier * (
                        jnp.sum(weights_valid_decrease * Vdiffs_trim, axis=0) / denom_nondec
                    )

                if self.loss_decr_max:
                    loss_exp_decrease_max = jnp.max(Vdiffs_trim)
                else:
                    loss_exp_decrease_max = 0

            loss_lipschitz = self.lambda_lipschitz * (
                jnp.maximum(lip_certificate - self.max_lip_certificate, 0)
                + jnp.maximum(lip_policy - self.max_lip_policy, 0)
            )

            # Auxiliary losses with masks, so invalid local neighbors do not affect min values.
            V_target_min = masked_min(V_target, mask_target_valid)
            V_init_min = masked_min(V_init, mask_init_valid)
            V_unsafe_min = masked_min(V_unsafe, mask_unsafe_valid)
            loss_min_target = jnp.maximum(0, V_target_min - self.glob_min)
            loss_min_init = jnp.maximum(0, V_target_min - V_init_min)
            loss_min_unsafe = jnp.maximum(0, V_target_min - V_unsafe_min)
            loss_aux = self.auxiliary_loss * (loss_min_target + loss_min_init + loss_min_unsafe)

            # Optional range and entropy regularization.
            n_all = jnp.concatenate(
                [n_init_centers, n_unsafe_centers, n_target_centers, n_decrease_centers],
                axis=0,
            )
            loss_n_reg = self.range_reg_weight * jnp.mean((n_all - self.range_target) ** 2)
            loss_entropy = self.weight_entropy_lambda * (
                jnp.sum(weights_init * jnp.log(weights_init + 1e-8))
                + jnp.sum(weights_unsafe * jnp.log(weights_unsafe + 1e-8))
                + jnp.sum(weights_target * jnp.log(weights_target + 1e-8))
                + jnp.sum(weights_decrease * jnp.log(weights_decrease + 1e-8))
            )
            loss_n_reg = 0
            loss_range_reg = loss_n_reg + loss_entropy

            loss_total = (
                loss_init
                + loss_unsafe
                + loss_exp_decrease_mean
                + loss_exp_decrease_max
                + loss_lipschitz
                + loss_aux
                + loss_range_reg
            )

            infos = {
                '0. total': loss_total,
                '1. init': loss_init,
                '2. unsafe': loss_unsafe,
                '3. expDecrease_mean': loss_exp_decrease_mean,
                '4. expDecrease_max': loss_exp_decrease_max,
                '5. loss_lipschitz': loss_lipschitz,
                '9. n_init_mean': jnp.mean(n_init_centers),
                '9.1 n_unsafe_mean': jnp.mean(n_unsafe_centers),
                '9.2 n_target_mean': jnp.mean(n_target_centers),
                '9.3 n_decrease_mean': jnp.mean(n_decrease_centers),
                '9.4 valid_init_frac': jnp.mean(mask_init_valid),
                '9.5 valid_unsafe_frac': jnp.mean(mask_unsafe_valid),
                '9.6 valid_target_frac': jnp.mean(mask_target_valid),
                '9.7 valid_decrease_frac': jnp.mean(mask_decrease_valid),
                '9.8 max_weight_decrease': jnp.max(weights_decrease),
                '9.9 loss_range_reg': loss_range_reg,
                '9.31 n_decrease_min': jnp.min(n_decrease_centers),
                '9.32 n_decrease_max': jnp.max(n_decrease_centers),

            }

            if self.auxiliary_loss > 0:
                infos['8. loss auxiliary'] = loss_aux

            return loss_total, infos

        loss_grad_fun = jax.value_and_grad(loss_fun, argnums=(0, 1, 2), has_aux=True)
        (loss_val, infos), (V_grads, Policy_grads, Range_grads) = loss_grad_fun(
            V_state.params,
            Policy_state.params,
            Range_state.params,
        )

        # Recompute samples for debug output using current params.
        n_init_log, _ = compute_radii(V_state.params, Range_state.params, centers_init)
        n_unsafe_log, _ = compute_radii(V_state.params, Range_state.params, centers_unsafe)
        n_target_log, _ = compute_radii(V_state.params, Range_state.params, centers_target)
        n_decrease_log, _ = compute_radii(V_state.params, Range_state.params, centers_decrease)

        key_init_local, key_unsafe_local, key_target_local, key_decrease_local = jax.random.split(
            perturbation_key, 4
        )
        keys_init = jax.random.split(key_init_local, N_init_centers)
        keys_unsafe = jax.random.split(key_unsafe_local, N_unsafe_centers)
        keys_target = jax.random.split(key_target_local, N_target_centers)
        keys_decrease = jax.random.split(key_decrease_local, N_decrease_centers)

        samples_init_log, _, _ = sample_neighbors_for_centers(centers_init, keys_init, n_init_log)
        samples_unsafe_log, _, _ = sample_neighbors_for_centers(centers_unsafe, keys_unsafe, n_unsafe_log)
        samples_target_log, _, _ = sample_neighbors_for_centers(centers_target, keys_target, n_target_log)
        samples_decrease_log, _, _ = sample_neighbors_for_centers(centers_decrease, keys_decrease, n_decrease_log)

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

        for s in ['init', 'unsafe', 'target', 'decrease', 'counterx']:
            filename = f"plots/{args.start_datetime}_train_debug_{str(s)}_iteration={iteration}"
            plot_dataset(self.env, additional_data=np.array(samples_in_batch[s]), folder=args.cwd, filename=filename)

        for s in ['counterx_init', 'counterx_unsafe', 'counterx_decrease']:
            filename = f"plots/{args.start_datetime}_train_debug_{str(s)}_iteration={iteration}"
            idxs = samples_in_batch[s]
            plot_dataset(
                self.env,
                additional_data=np.array(samples_in_batch['counterx'])[idxs],
                folder=args.cwd,
                filename=filename,
            )
