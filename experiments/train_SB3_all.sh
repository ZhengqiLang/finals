#!/bin/bash

for s in {1..10};
do
  for model in LinearSystem MyPendulum CollisionAvoidance;
  do
    python3 train_SB3.py --model $model --layout 0 --total_steps 1000 --algorithm 'ALL_paper' --seed $s --num_envs 1;
    python3 train_SB3.py --model $model --layout 0 --total_steps 10000 --algorithm 'ALL_paper' --seed $s --num_envs 2;
    python3 train_SB3.py --model $model --layout 0 --total_steps 100000 --algorithm 'ALL_paper' --seed $s --num_envs 10;
    python3 train_SB3.py --model $model --layout 0 --total_steps 1000000 --algorithm 'ALL_paper' --seed $s --num_envs 20;
  done

  python3 train_SB3.py --model LinearSystem --layout 1 --total_steps 1000 --algorithm 'ALL_paper' --seed $s --num_envs 1;
  python3 train_SB3.py --model LinearSystem --layout 1 --total_steps 10000 --algorithm 'ALL_paper' --seed $s --num_envs 2;
  python3 train_SB3.py --model LinearSystem --layout 1 --total_steps 100000 --algorithm 'ALL_paper' --seed $s --num_envs 10;
  python3 train_SB3.py --model LinearSystem --layout 1 --total_steps 1000000 --algorithm 'ALL_paper' --seed $s --num_envs 20;
done