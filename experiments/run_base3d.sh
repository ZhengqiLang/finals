#!/bin/bash


#!/bin/bash
# Usage:
#   bash run_local_sampling_sweep.sh
#   bash run_local_sampling_sweep.sh 300 1000 1
#
# Example with log:
#   mkdir -p output
#   bash run_local_sampling_sweep.sh 300 1000 1 | tee output/log_local_sampling_sweep.txt
run_with_timeout() {
  local timeout_seconds=$1
  shift

  "$@" &
  local cmd_pid=$!

  (
    sleep "$timeout_seconds"
    kill -TERM "$cmd_pid" 2>/dev/null
    sleep 10
    kill -KILL "$cmd_pid" 2>/dev/null
  ) &
  local watcher_pid=$!

  wait "$cmd_pid"
  local exit_code=$?

  kill "$watcher_pid" 2>/dev/null

  # 143 = killed by SIGTERM, 137 = killed by SIGKILL
  if [ "$exit_code" -eq 143 ] || [ "$exit_code" -eq 137 ]; then
    return 124
  fi

  return "$exit_code"
}
verify_batch_size=300
forward_pass_batch_size=1000
time_mul=1

if [ ! -z "$1" ]; then verify_batch_size=$1; fi
if [ ! -z "$2" ]; then forward_pass_batch_size=$2; fi
if [ ! -z "$3" ]; then time_mul=$3; fi

# Timeout setting
TO=$(((20000)*time_mul))

# Models to run
models=(
  "PlanarRobot"
)

# Parameters to sweep
distance_types=(
  "l2"
)

local_weight_types=(
  "inverse"
)

seeds=(1 2 3)

# Shared flags
common_flags="\
--probability_bound 0.8 \
--pretrain_method PPO_JAX \
--pretrain_total_steps 120000 \
--mesh_loss 0.005 \
--no-exp_certificate \
--verify_batch_size ${verify_batch_size} \
--forward_pass_batch_size ${forward_pass_batch_size} \
--max_refine_factor 4 \
--eps_decrease 0.001 \
--hidden_layers 3 \
--mesh_loss_decrease_per_iter 0.9 \
--epochs 100 \
--refine_threshold 250000000 \
--ppo_max_policy_lipschitz 10 \
--expDecr_multiplier 10 \
--logger_prefix base3d \
--mesh_verify_grid_init 0.04 \
--local_samples_per_center 10 "

mkdir -p output

echo "============================================================"
echo "Start local sampling sweep"
echo "verify_batch_size=${verify_batch_size}"
echo "forward_pass_batch_size=${forward_pass_batch_size}"
echo "timeout=${TO}"
echo "============================================================"

for model in "${models[@]}"
do
  for seed in "${seeds[@]}"
  do
    for distance_type in "${distance_types[@]}"
    do
      for local_weight_type in "${local_weight_types[@]}"
      do
        echo ""
        echo "============================================================"
        echo "Running:"
        echo "model=${model}"
        echo "seed=${seed}"
        echo "distance_type=${distance_type}"
        echo "local_weight_type=${local_weight_type}"
        echo "============================================================"
        echo ""

        run_with_timeout ${TO} python run.py \
          --model ${model} \
          --seed ${seed} \
          ${common_flags} \
          --distance_type ${distance_type} \
          --local_weight_type ${local_weight_type}

        exit_code=$?

        if [ "$exit_code" -eq 124 ]; then
          echo "TIMEOUT:"
          echo "model=${model}, seed=${seed}, distance_type=${distance_type}, local_weight_type=${local_weight_type}"
        else
          echo "Finished:"
          echo "model=${model}, seed=${seed}, distance_type=${distance_type}, local_weight_type=${local_weight_type}, exit_code=${exit_code}"
        fi

        echo ""
        echo "Finished:"
        echo "model=${model}, seed=${seed}, distance_type=${distance_type}, local_weight_type=${local_weight_type}, exit_code=${exit_code}"
        echo ""

      done
    done
  done
done

echo "============================================================"
echo "All experiments finished"
echo "============================================================" 