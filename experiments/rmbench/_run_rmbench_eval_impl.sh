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
control_mode="${8:-${CONTROL_MODE:-joint}}"
unnorm_key="${9:-${UNNORM_KEY:-${MANTIS_NORM_KEY:-}}}"
max_state_dim="${10:-${MAX_STATE_DIM:-}}"
original_action_dim="${11:-${ORIGINAL_ACTION_DIM:-}}"

norm_file_path="${NORM_FILE_PATH:-${MANTIS_NORM_STATS_PATH:-configs/norm_stats.json}}"
instruction_type="${INSTRUCTION_TYPE:-unseen}"
rmbench_policy_name="${RMBENCH_POLICY_NAME:-WLA}"

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
        echo "ERROR: ${key} is required. Set it in run_rmbench_eval.sh or pass it as an argument." >&2
        exit 1
    fi
done

if [[ "${ROBOTWIN_FILTER_DRIVER_LIBS:-0}" == "1" && -n "${LD_LIBRARY_PATH:-}" ]]; then
    filtered_ld_library_path=""
    IFS=':' read -ra ld_library_paths <<< "${LD_LIBRARY_PATH}"
    for ld_library_path in "${ld_library_paths[@]}"; do
        if [[ "${ld_library_path}" == *nvidia550extract* || "${ld_library_path}" == */cuda/compat/lib* ]]; then
            continue
        fi

        if [[ -z "${filtered_ld_library_path}" ]]; then
            filtered_ld_library_path="${ld_library_path}"
        else
            filtered_ld_library_path="${filtered_ld_library_path}:${ld_library_path}"
        fi
    done
    export LD_LIBRARY_PATH="${filtered_ld_library_path}"
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[36mtask: ${task_name} | config: ${task_config} | ckpt: ${ckpt_setting} | seed: ${seed} | mode: ${control_mode} | unnorm_key: ${unnorm_key} | state_dim: ${max_state_dim} | action_dim: ${original_action_dim}\033[0m"

cmd=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/run_rmbench_eval.py"
    --task_name "${task_name}"
    --task_config "${task_config}"
    --ckpt_setting "${ckpt_setting}"
    --seed "${seed}"
    --control_mode "${control_mode}"
)

add_arg() {
    local name="$1"
    local value="$2"
    if [[ -n "${value}" ]]; then
        cmd+=(--"${name}" "${value}")
    fi
}

add_arg "model_id" "${model_id}"
add_arg "checkpoints_dir" "${checkpoints_dir}"
add_arg "rmbench_root" "${RMBENCH_ROOT:-}"
add_arg "norm_file_path" "${norm_file_path}"
add_arg "unnorm_key" "${unnorm_key}"
add_arg "max_state_dim" "${max_state_dim}"
add_arg "original_action_dim" "${original_action_dim}"
add_arg "instruction_type" "${instruction_type}"
add_arg "policy_name" "${rmbench_policy_name}"

cd "${WLA_ROOT}"
PYTHONWARNINGS=ignore::UserWarning "${cmd[@]}"
