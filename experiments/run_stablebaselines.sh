#!/bin/bash
# bash run_stablebaselines.sh | tee output/log_stablebaselines.txt

verify_batch_size=30000
forward_pass_batch_size=1000000
time_mul=1
if [ ! -z "$1" ]; then verify_batch_size=$1; fi
if [ ! -z "$2" ]; then forward_pass_batch_size=$2; fi
if [ ! -z "$3" ]; then time_mul=$3; fi
collision_batch_size=$((verify_batch_size/4))

steps=(10000 100000 1000000)
algos=("TRPO" "SAC" "TQC" "A2C")
models=("LinearSystem --verify_batch_size ${verify_batch_size}" 
        "LinearSystem --layout 1 --verify_batch_size ${verify_batch_size}" 
        "MyPendulum --verify_batch_size ${verify_batch_size}" 
        "CollisionAvoidance --noise_partition_cells 24 --verify_batch_size ${collision_batch_size}")
all_flags="--epochs 100 --eps_decrease 0.01 --hidden_layers 3 --refine_threshold 100000000 --forward_pass_batch_size ${forward_pass_batch_size}"
extra_flags=("--expDecr_multiplier 10 --mesh_loss 0.0005" "--expDecr_multiplier 0.1 --mesh_loss 0.001")

TO=$(((1800+50)*time_mul)) # Add 50 seconds to avoid that loading the SB3 checkpoint causes a timeout
TOtable=$((1800*time_mul))

for flags in 0 1;
do
  for i in 2 1 0;
  do
    for j in {0..3};
    do
      for seed in {1..10};
      do
          checkpoint="ckpt_pretrain_sb3/LinearSystem_layout=0_alg=${algos[j]}_layers=3_neurons=128_outfn=None_seed=${seed}_steps=${steps[i]}"
          timeout $TO python run.py --load_ckpt $checkpoint --logger_prefix linsys_sb --seed $seed --model ${models[0]} ${extra_flags[flags]} $all_flags --probability_bound 0.999999 --exp_certificate;
      done
    done
  done

  for i in 2 1 0;
  do
    for j in {0..3};
    do
      for seed in {1..10};
      do
          checkpoint="ckpt_pretrain_sb3/LinearSystem_layout=1_alg=${algos[j]}_layers=3_neurons=128_outfn=None_seed=${seed}_steps=${steps[i]}"
          timeout $TO python run.py --load_ckpt $checkpoint --logger_prefix linsys1_sb --seed $seed --model ${models[1]} ${extra_flags[flags]} $all_flags --probability_bound 0.999999 --exp_certificate;
      done
    done
  done

  for i in 2 1 0;
  do
    for j in {0..3};
    do
      for seed in {1..10};
      do
          checkpoint="ckpt_pretrain_sb3/MyPendulum_layout=0_alg=${algos[j]}_layers=3_neurons=128_outfn=None_seed=${seed}_steps=${steps[i]}"
          timeout $TO python run.py --load_ckpt $checkpoint --logger_prefix pendulum_sb --seed $seed --model ${models[2]} ${extra_flags[flags]} $all_flags --probability_bound 0.999999 --exp_certificate;
      done
    done
  done

  for i in 2 1 0;
  do
    for j in {0..3};
    do
      for seed in {1..10};
      do
          checkpoint="ckpt_pretrain_sb3/CollisionAvoidance_layout=0_alg=${algos[j]}_layers=3_neurons=128_outfn=None_seed=${seed}_steps=${steps[i]}"
          timeout $TO python run.py --load_ckpt $checkpoint --logger_prefix collision_sb --seed $seed --model ${models[3]} ${extra_flags[flags]} $all_flags --probability_bound 0.999999 --exp_certificate;
      done
    done
  done
done

# Generate table
python table_generator.py --folders sb3 --timeout $TOtable
