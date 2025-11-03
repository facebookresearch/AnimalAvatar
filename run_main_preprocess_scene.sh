#!/bin/bash

declare -a sequence_indexes=(
    "slow_1313807000"
    "left_right_turn_2058226999"
    "slow_turn_1771233000"
    "start_stop_785558000"
    "start_stop_1271493000"
    "turn_left_1771233000"
    "turn_right_1771233000"
    "walk_869488000"
)

for sequence_index in "${sequence_indexes[@]}"
do
    python main_preprocess_scene.py -sequence_index "$sequence_index" --visualize --recompute --custom
done
