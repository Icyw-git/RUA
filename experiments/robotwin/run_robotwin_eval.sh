#!/usr/bin/env bash
TASK_NAME="click_bell"
TASK_CONFIG="demo_clean"
CKPT_SETTING="WLA"
SEED="0"
GPU_ID="0"

MODEL_ID="wla_robotwin_all_image_action"
CHECKPOINTS_DIR=""
CONTROL_MODE="eef"
UNNORM_KEY="robotwin_all_eef"
MAX_STATE_DIM="16"
ORIGINAL_ACTION_DIM="16"

# MODEL_ID="wla_robotwin_cotrain_same_emb_image_action"
# CHECKPOINTS_DIR=""
# CONTROL_MODE="joint"
# UNNORM_KEY="robotwin_all"
# MAX_STATE_DIM="14"
# ORIGINAL_ACTION_DIM="14"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/_run_robotwin_eval_impl.sh"
