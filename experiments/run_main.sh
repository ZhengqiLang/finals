#!/bin/bash
# bash run_main.sh | tee output/log_main.txt

verify_batch_size=30000
forward_pass_batch_size=1000000
time_mul=1
if [ ! -z "$1" ]; then verify_batch_size=$1; fi
if [ ! -z "$2" ]; then forward_pass_batch_size=$2; fi
if [ ! -z "$3" ]; then time_mul=$3; fi
collision_batch_size=$((verify_batch_size/4))

# This bash script runs the main experiments presented in the paper. First, the script runs all benchmarks for 3 seeds. Then, it only runs the remaining 7 seeds for benchmarks
# that did not lead to too many timeouts on the first 3 seeds.

models=("LinearSystem --verify_batch_size ${verify_batch_size}" 
        "LinearSystem --layout 1 --verify_batch_size ${verify_batch_size}" 
        "MyPendulum --verify_batch_size ${verify_batch_size}" 
        "CollisionAvoidance --noise_partition_cells 24 --verify_batch_size ${collision_batch_size}")
all_flags="--logger_prefix main --eps_decrease 0.01 --ppo_max_policy_lipschitz 10 --hidden_layers 3 --expDecr_multiplier 10 --pretrain_method PPO_JAX --pretrain_total_steps 100000 --refine_threshold 250000000 --forward_pass_batch_size ${forward_pass_batch_size}"
flags_mesh1="--mesh_loss 0.001 --mesh_loss_decrease_per_iter 0.8"
flags_mesh2="--mesh_loss 0.01 --mesh_loss_decrease_per_iter 0.8"

prob_bounds=(0.8 0.9 0.95 0.99 0.999 0.9999 0.999999)

TO=$(((1800+200)*time_mul)) # Add 200 seconds to avoid that pretraining causes a timeout
TOtable=$((1800*time_mul))

# Run linear system, pendulum, and linear system (hard)
for x in 0 1 2
do
  for seed in 1 2 3
  do
    for p in "${prob_bounds[@]}"
    do
      timeout $TO python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh1 --probability_bound $p --exp_certificate;
      timeout $TO python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh1 --probability_bound $p --exp_certificate --no-weighted --no-cplip;
    done
    for p in "${prob_bounds[@]}"
    do
      timeout $TO python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh1 --probability_bound $p --no-exp_certificate;
      timeout $TO python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh1 --probability_bound $p --no-exp_certificate --no-weighted --no-cplip;
    done
  done
done
# Run collision avoidance
for x in 3
do
  for seed in 1 2 3
  do
    for p in "${prob_bounds[@]}"
    do
      timeout $TO python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh2 --probability_bound $p --exp_certificate;
      timeout $TO python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh2 --probability_bound $p --exp_certificate --no-weighted --no-cplip;
    done
    for p in "${prob_bounds[@]}"
    do
      timeout $TO python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh2 --probability_bound $p --no-exp_certificate;
      timeout $TO python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh2 --probability_bound $p --no-exp_certificate --no-weighted --no-cplip;
    done
  done
done

python main_genOtherSeeds.py;
bash experiments/run_main_otherSeeds.sh $verify_batch_size $forward_pass_batch_size $time_mul;

# Generate table
python table_generator.py --folders main --timeout $TOtable
