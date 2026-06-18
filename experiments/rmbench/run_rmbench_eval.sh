TASK_NAME="battery_try"
TASK_CONFIG="demo_clean"
CKPT_SETTING="WLA"
SEED="0"
GPU_ID="0"

MODEL_ID="SJTU-DENG-Lab/wla_rmbench_battery_try_image_action_language"
CHECKPOINTS_DIR=""
CONTROL_MODE="joint"
UNNORM_KEY="rmbench_battery_try"
MAX_STATE_DIM="14"
ORIGINAL_ACTION_DIM="14"





source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/_run_rmbench_eval_impl.sh"
