#!/bin/bash

python3 collect_checkpoints.py --model LinearSystem --layout 0 --probability_bound 0.8
python3 collect_checkpoints.py --model LinearSystem --layout 1 --probability_bound 0.8
python3 collect_checkpoints.py --model MyPendulum --probability_bound 0.8
python3 collect_checkpoints.py --model CollisionAvoidance --probability_bound 0.8
python3 collect_checkpoints.py --model LinearSystem --layout 0 --probability_bound 0.999999
python3 collect_checkpoints.py --model LinearSystem --layout 1 --probability_bound 0.999999
python3 collect_checkpoints.py --model MyPendulum --probability_bound 0.999999
python3 collect_checkpoints.py --model CollisionAvoidance --probability_bound 0.999999
