#!/bin/bash
# bash run_LipBaB.sh | tee output/log_LipBaB.txt

for f in "LinearSystem0" "LinearSystem1" "MyPendulum0" "CollisionAvoidance0";
do
    for p in "0.8" "0.999999" 
    do
        for s in {1..10};
        do
            echo $f $p $s;
            if [ -d ckpt_lipbab/${f}_True_${p}_${s} ]; then
                checkpoint="ckpt_lipbab/${f}_True_${p}_${s}/final_ckpt";
                python3 LipBaB_finalcheckpoint.py --checkpoint $checkpoint;
            else
                echo "skip";
            fi
        done
    done
done
