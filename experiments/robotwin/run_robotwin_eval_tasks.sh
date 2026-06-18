#!/usr/bin/env bash
set -euo pipefail

# If command-line task names are provided, they override this list.
TASK_NAMES=(
    stack_blocks_three
    stack_bowls_three
    place_a2b_left
)

TASK_CONFIGS=(
    demo_clean
    demo_randomized
)

SCRIPT_DIR=$(cd -- $(dirname -- ${BASH_SOURCE[0]}) >/dev/null 2>&1 && pwd)

if [[ $# -gt 0 ]]; then
    TASK_NAMES=($@)
fi

for task_name in ${TASK_NAMES[@]}; do
    for task_config in ${TASK_CONFIGS[@]}; do
        echo -e \033[35mRunning ${task_name} / ${task_config}\033[0m
        bash ${SCRIPT_DIR}/run_robotwin_eval.sh \
            ${task_name} \
            ${task_config}
    done
done
