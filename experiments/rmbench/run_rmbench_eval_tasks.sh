#!/usr/bin/env bash
set -euo pipefail

# If command-line task names are provided, they override this list.
TASK_NAMES=(
    battery_try
    blocks_ranking_try
    cover_blocks
    press_button
)

TASK_CONFIGS=(
    demo_clean
)

SCRIPT_DIR=$(cd -- $(dirname -- ${BASH_SOURCE[0]}) >/dev/null 2>&1 && pwd)

if [[ $# -gt 0 ]]; then
    TASK_NAMES=($@)
fi

default_model_id_for_task() {
    case "$1" in
        battery_try)
            echo "wla_rmbench_battery_try_image_language_action"
            ;;
        blocks_ranking_try)
            echo "wla_rmbench_blocks_ranking_try_image_language_action"
            ;;
        cover_blocks)
            echo "wla_rmbench_cover_blocks_image_language_action"
            ;;
        press_button)
            echo "wla_rmbench_press_button_image_language_action"
            ;;
        *)
            echo "${MODEL_ID:-}"
            ;;
    esac
}

for task_name in ${TASK_NAMES[@]}; do
    for task_config in ${TASK_CONFIGS[@]}; do
        model_id="${MODEL_ID:-$(default_model_id_for_task "${task_name}")}"
        echo -e \033[35mRunning ${task_name} / ${task_config}\033[0m
        bash ${SCRIPT_DIR}/run_rmbench_eval.sh \
            ${task_name} \
            ${task_config} \
            "${CKPT_SETTING:-WLA}" \
            "${SEED:-0}" \
            "${GPU_ID:-0}" \
            "${model_id}"
    done
done
