#!/usr/bin/env bash

# Available TRAINING_SETTING:
# TRAINING_SETTING="libero_all_image_action"
# TRAINING_SETTING="libero_all_action"
# TRAINING_SETTING="robotwin_all_image_action"
# TRAINING_SETTING="robotwin_all_action"
# TRAINING_SETTING="robotwin_seen_tasks_image_action"
# TRAINING_SETTING="robotwin_cross_emb_videos_cotrain_image_action"
# TRAINING_SETTING="robotwin_same_emb_videos_cotrain_image_action"
# TRAINING_SETTING="rmbench_battery_try_image_action_language"
# TRAINING_SETTING="rmbench_blocks_ranking_try_image_action_language"
# TRAINING_SETTING="rmbench_cover_blocks_image_action_language"
# TRAINING_SETTING="rmbench_press_button_image_action_language"

TRAINING_SETTING="robotwin_all_image_action"

mkdir -p log

torchrun --nproc-per-node=8 train.py \
    --run_name "${TRAINING_SETTING}" \
    --config_file "${TRAINING_SETTING}.yaml" \
    --base_dir ./ \
    --logging_dir log \
    > "log/${TRAINING_SETTING}.log" 2>&1