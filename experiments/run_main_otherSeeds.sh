#!/bin/bash

verify_batch_size=30000
forward_pass_batch_size=1000000
time_mul=1
if [ ! -z "$1" ]; then verify_batch_size=$1; fi
if [ ! -z "$2" ]; then forward_pass_batch_size=$2; fi
if [ ! -z "$3" ]; then time_mul=$3; fi
collision_batch_size=$((verify_batch_size/4))

models=("LinearSystem --verify_batch_size ${verify_batch_size}" "LinearSystem --layout 1 --verify_batch_size ${verify_batch_size}" "MyPendulum --verify_batch_size ${verify_batch_size}" "CollisionAvoidance --noise_partition_cells 24 --verify_batch_size ${collision_batch_size}"))
all_flags="--logger_prefix main --eps_decrease 0.01 --ppo_max_policy_lipschitz 10 --hidden_layers 3 --expDecr_multiplier 10 --pretrain_method PPO_JAX --pretrain_total_steps 100000 --refine_threshold 250000000 --forward_pass_batch_size ${forward_pass_batch_size}"
flags_mesh1="--mesh_loss 0.001 --mesh_loss_decrease_per_iter 0.8"
flags_mesh2="--mesh_loss 0.01 --mesh_loss_decrease_per_iter 0.8"

TO=$(((1800+200)*time_mul)) # Add 200 seconds to avoid that pretraining causes a timeout

for seed in {4..10};
do
  # Linsys layout=0
  # ours
  for p in 0.9 0.99
  do
    timeout $TO python run.py --seed $seed --model ${models[0]} $all_flags $flags_mesh1 --probability_bound $p --exp_certificate;
  done
  # no-lip
  for p in 0.9 0.99
  do
    timeout $TO python run.py --seed $seed --model ${models[0]} $all_flags $flags_mesh1 --probability_bound $p --exp_certificate --no-weighted --no-cplip;
  done
  # no-exp
  for p in 0.9 0.99
  do
    timeout $TO python run.py --seed $seed --model ${models[0]} $all_flags $flags_mesh1 --probability_bound $p --no-exp_certificate;
  done
  # base
  for p in 0.9 0.99
  do
    timeout $TO python run.py --seed $seed --model ${models[0]} $all_flags $flags_mesh1 --probability_bound $p --no-exp_certificate --no-weighted --no-cplip;
  done
  
  # Linsys layout=1
  # ours
  for p in 0.9 0.99
  do
    timeout $TO python run.py --seed $seed --model ${models[1]} $all_flags $flags_mesh1 --probability_bound $p --exp_certificate;
  done
  # no-lip
  for p in 0.9
  do
    timeout $TO python run.py --seed $seed --model ${models[1]} $all_flags $flags_mesh1 --probability_bound $p --exp_certificate --no-weighted --no-cplip;
  done
  # no-exp
  # base
  
  # Pendulum
  # ours
  for p in 0.9 0.99
  do
    timeout $TO python run.py --seed $seed --model ${models[2]} $all_flags $flags_mesh1 --probability_bound $p --exp_certificate;
  done
  # no-lip
  for p in 0.9 0.99
  do
    timeout $TO python run.py --seed $seed --model ${models[2]} $all_flags $flags_mesh1 --probability_bound $p --exp_certificate --no-weighted --no-cplip;
  done
  # no-exp
  for p in 0.9
  do
    timeout $TO python run.py --seed $seed --model ${models[2]} $all_flags $flags_mesh1 --probability_bound $p --no-exp_certificate;
  done
  # base
  
  # CollisionAvoidance
  # ours
  for p in 0.9 0.99
  do
    timeout $TO python run.py --seed $seed --model ${models[3]} $all_flags $flags_mesh2 --probability_bound $p --exp_certificate;
  done
  # no-lip
  for p in 0.9 0.99
  do
    timeout $TO python run.py --seed $seed --model ${models[3]} $all_flags $flags_mesh2 --probability_bound $p --exp_certificate --no-weighted --no-cplip;
  done
  # no-exp
  for p in 0.9 0.99
  do
    timeout $TO python run.py --seed $seed --model ${models[3]} $all_flags $flags_mesh2 --probability_bound $p --no-exp_certificate;
  done
  # base
  
done
