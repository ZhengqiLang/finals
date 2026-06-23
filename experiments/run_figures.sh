#!/bin/bash
# Expected runtime (with GPU acceleration): 10-15 minutes

verify_batch_size=30000
forward_pass_batch_size=1000000
time_mul=1
if [ ! -z "$1" ]; then verify_batch_size=$1; fi
if [ ! -z "$2" ]; then forward_pass_batch_size=$2; fi
if [ ! -z "$3" ]; then time_mul=$3; fi
collision_batch_size=$((verify_batch_size/4))

TO=$(((1800+200)*time_mul)) # Add 200 seconds to avoid that pretraining causes a timeout

models=("LinearSystem --verify_batch_size ${verify_batch_size}" 
        "LinearSystem --layout 1 --verify_batch_size ${verify_batch_size}" 
        "MyPendulum --verify_batch_size ${verify_batch_size}" 
        "CollisionAvoidance --noise_partition_cells 24 --verify_batch_size ${collision_batch_size}")
all_flags="--logger_prefix figures --eps_decrease 0.01 --ppo_max_policy_lipschitz 10 --hidden_layers 3 --expDecr_multiplier 10 --pretrain_method PPO_JAX --pretrain_total_steps 100000 --refine_threshold 250000000 --forward_pass_batch_size ${forward_pass_batch_size}"
flags_mesh1="--mesh_loss 0.001 --mesh_loss_decrease_per_iter 0.9"
flags_mesh2="--mesh_loss 0.01 --mesh_loss_decrease_per_iter 0.8"

# Generate figures of selected RASMs
timeout $TO python run.py --seed 1 --model ${models[0]} $all_flags $flags_mesh1 --probability_bound 0.999999 --exp_certificate;
timeout $TO python run.py --seed 1 --model ${models[1]} $all_flags $flags_mesh1 --probability_bound 0.999999 --exp_certificate;
timeout $TO python run.py --seed 1 --model ${models[2]} $all_flags $flags_mesh1 --probability_bound 0.999999 --exp_certificate;
timeout $TO python run.py --seed 1 --model ${models[3]} $all_flags $flags_mesh2 --probability_bound 0.999999 --exp_certificate;
