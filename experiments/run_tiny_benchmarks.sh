#!/bin/bash

verify_batch_size=30000
forward_pass_batch_size=1000000
time_mul=1
if [ ! -z "$1" ]; then verify_batch_size=$1; fi
if [ ! -z "$2" ]; then forward_pass_batch_size=$2; fi
if [ ! -z "$3" ]; then time_mul=$3; fi
collision_batch_size=$((verify_batch_size/4))


models=("LinearSystem --verify_batch_size ${verify_batch_size}" 
        "LinearSystem --layout 1 --verify_batch_size ${verify_batch_size}" 
        "MyPendulum --verify_batch_size ${verify_batch_size}" 
        "CollisionAvoidance --noise_partition_cells 24 --verify_batch_size ${collision_batch_size}")
all_flags="--logger_prefix main --eps_decrease 0.01 --ppo_max_policy_lipschitz 10 --hidden_layers 3 --expDecr_multiplier 10 --pretrain_method PPO_JAX --pretrain_total_steps 100000 --refine_threshold 250000000  --forward_pass_batch_size ${forward_pass_batch_size}"
flags_mesh1="--mesh_loss 0.001 --mesh_loss_decrease_per_iter 0.8"
flags_mesh2="--mesh_loss 0.01 --mesh_loss_decrease_per_iter 0.8"

prob_bounds=(0.9 0.99)
TO=$(((600+50)*time_mul))  # Add 50 seconds to avoid that pretraining causes a timeout
TOtable=$((600*time_mul))

############################################################
### GENERATE FIGURES
############################################################

# The following script runs four individual benchmark instances, to generate the plots presented in the paper.
bash experiments/run_figures.sh $verify_batch_size $forward_pass_batch_size $time_mul

############################################################
### MAIN BENCHMARKS
############################################################

# Run linear system, pendulum, and linear system (hard)
for x in 0
do
  for seed in 1
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

# Generate table
python table_generator.py --folders main --timeout $TOtable

############################################################
### STABLE BASELINES
############################################################

steps=(10000 100000 1000000)
algos=("TRPO" "SAC" "TQC" "A2C")
models=("LinearSystem --verify_batch_size ${verify_batch_size}" 
        "LinearSystem --layout 1 --verify_batch_size ${verify_batch_size}" 
        "MyPendulum --verify_batch_size ${verify_batch_size}" 
        "CollisionAvoidance --noise_partition_cells 24 --verify_batch_size ${collision_batch_size}")
all_flags="--epochs 100 --eps_decrease 0.01 --hidden_layers 3 --refine_threshold 100000000  --forward_pass_batch_size ${forward_pass_batch_size}"

extra_flags=("--expDecr_multiplier 10 --mesh_loss 0.0005" "--expDecr_multiplier 0.1 --mesh_loss 0.001")

TO=$((600*time_mul))

for flags in 0;
do
  for i in 2 1 0;
  do
    for j in {0..3};
    do
      for seed in 1;
      do
          checkpoint="ckpt_pretrain_sb3/LinearSystem_layout=0_alg=${algos[j]}_layers=3_neurons=128_outfn=None_seed=${seed}_steps=${steps[i]}"
          timeout $TO python run.py --load_ckpt $checkpoint --logger_prefix linsys_sb --seed $seed --model ${models[0]} ${extra_flags[flags]} $all_flags --probability_bound 0.999999 --exp_certificate;
      done
    done
  done
done

# Generate table
python table_generator.py --folders sb3 --timeout $TOtable

############################################################
### HARD EXPERIMENTS (ONLY TRIPLE INTEGRATOR)
############################################################

triple_batch_size=$((verify_batch_size*2/3))
all_flags="--eps_decrease 0.01 --ppo_max_policy_lipschitz 10 --expDecr_multiplier 10 --pretrain_method PPO_JAX --refine_threshold 250000000 --epochs 100  --forward_pass_batch_size ${forward_pass_batch_size}"

flags_triple="--model TripleIntegrator --logger_prefix TripleIntegrator --pretrain_total_steps 100000 --hidden_layers 3 --mesh_loss 0.005 --mesh_loss_decrease_per_iter 0.9 --mesh_verify_grid_init 0.04 --noise_partition_cells 6 --max_refine_factor 4 --verify_batch_size ${triple_batch_size}"

TO=$(((1800+200)*time_mul)) # Add 200 seconds to avoid that pretraining causes a timeout
TOtable=$((1800*time_mul))

# Triple integrator
for seed in 1;
do
  for p in 0.9 0.99
  do
    # Our method
    timeout $TO python run.py --seed $seed $flags_triple $all_flags --probability_bound $p --exp_certificate;
  done
done

# Generate table
python table_generator.py --folders hard --timeout $TOtable
