set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WLA_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"

PYTHON_BIN="${PYTHON:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi

task_name="${1:-${TASK_NAME:-}}"
task_config="${2:-${TASK_CONFIG:-}}"
ckpt_setting="${3:-${CKPT_SETTING:-}}"
seed="${4:-${SEED:-}}"
gpu_id="${5:-${GPU_ID:-}}"
model_id="${6:-${MODEL_ID:-}}"
checkpoints_dir="${7:-${CHECKPOINTS_DIR:-}}"
control_mode="${8:-${CONTROL_MODE:-eef}}"
unnorm_key="${9:-${UNNORM_KEY:-}}"
max_state_dim="${10:-${MAX_STATE_DIM:-}}"
original_action_dim="${11:-${ORIGINAL_ACTION_DIM:-}}"

norm_file_path="${NORM_FILE_PATH:-configs/norm_stats.json}"
instruction_type="${INSTRUCTION_TYPE:-unseen}"
robotwin_policy_name="${ROBOTWIN_POLICY_NAME:-WLA}"

required_values=(
    "task_name:${task_name}"
    "task_config:${task_config}"
    "ckpt_setting:${ckpt_setting}"
    "seed:${seed}"
    "gpu_id:${gpu_id}"
    "model_id:${model_id}"
    "control_mode:${control_mode}"
    "unnorm_key:${unnorm_key}"
)

for item in "${required_values[@]}"; do
    key="${item%%:*}"
    value="${item#*:}"
    if [[ -z "${value}" ]]; then
        echo "ERROR: ${key} is required. Set it in run_robotwin_eval.sh or pass it as an argument." >&2
        exit 1
    fi
done

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[36mtask: ${task_name} | config: ${task_config} | ckpt: ${ckpt_setting} | seed: ${seed} | mode: ${control_mode} | unnorm_key: ${unnorm_key} | state_dim: ${max_state_dim} | action_dim: ${original_action_dim}\033[0m"

cmd=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/run_robotwin_eval.py"
    --task_name "${task_name}"
    --task_config "${task_config}"
    --ckpt_setting "${ckpt_setting}"
    --seed "${seed}"
    --control_mode "${control_mode}"
)

if [[ -n "${model_id}" ]]; then
    cmd+=(--model_id "${model_id}")
fi

if [[ -n "${checkpoints_dir}" ]]; then
    cmd+=(--checkpoints_dir "${checkpoints_dir}")
fi

if [[ -n "${ROBOTWIN_ROOT:-}" ]]; then
    cmd+=(--robotwin_root "${ROBOTWIN_ROOT}")
fi

if [[ -n "${norm_file_path}" ]]; then
    cmd+=(--norm_file_path "${norm_file_path}")
fi

if [[ -n "${unnorm_key}" ]]; then
    cmd+=(--unnorm_key "${unnorm_key}")
fi

if [[ -n "${max_state_dim}" ]]; then
    cmd+=(--max_state_dim "${max_state_dim}")
fi

if [[ -n "${original_action_dim}" ]]; then
    cmd+=(--original_action_dim "${original_action_dim}")
fi

if [[ -n "${instruction_type}" ]]; then
    cmd+=(--instruction_type "${instruction_type}")
fi

if [[ -n "${robotwin_policy_name}" ]]; then
    cmd+=(--policy_name "${robotwin_policy_name}")
fi

cd "${WLA_ROOT}"
"${cmd[@]}"
