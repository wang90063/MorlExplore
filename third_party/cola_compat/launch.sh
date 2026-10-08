#!/bin/zsh
# launch.sh <task: hopper|ant|walker|cheetah> <seed> <run_dir> [num_steps] [eval_interval]
# Flags are copied verbatim from COLA's run.sh (2-objective v2 tasks).
cd "$(dirname "$0")/../.."
COMMON=(--old_Q_update_freq 1 --regular_bar 0.25 --consider_other --prefer 0 --buf_num 0 --q_freq 1000 --EA_policy_num 0 --RL_policy_num 0 --iso_sigma 0.005 --line_sigma 0.05 --Use_Critic_Preference --Use_Policy_Preference --train_with_fixed_preference --reward_coef 1.0 --dynamic_coef 1.0 --value_coef 1.0 --encoder_update_freq 1 --Policy_use_latent --Policy_use_s --Policy_use_w --Critic_use_both --Critic_use_s --Critic_use_a)
case $1 in
  hopper)  T=(--env_id MO-Hopper-v2 --regular_alpha 1.0 --latent_dim 10) ;;
  walker)  T=(--env_id MO-Walker2d-v2 --regular_alpha 1.0 --latent_dim 10) ;;
  ant)     T=(--env_id MO-Ant-v2 --regular_alpha 0.001 --latent_dim 50 --use_avg) ;;
  cheetah) T=(--env_id MO-New-HalfCheetah-v2 --regular_alpha 0.01 --latent_dim 50 --use_avg) ;;
  *) echo "unknown task $1"; exit 1 ;;
esac
mkdir -p "$3"
# resumes from $3/ckpt.pt if present (run_cola.py); stdout is appended so a resumed run keeps its history
RUN_DIR="$3" COLA_EXIT_AFTER=${COLA_EXIT_AFTER:-0} COLA_NUM_STEPS=${4:-1000000} COLA_EVAL_INTERVAL=${5:-50000} COLA_NO_ANOMALY=1 COLA_FAST_ELU=1 \
OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 exec .venv/bin/python third_party/cola_compat/run_cola.py "${T[@]}" --seed $2 "${COMMON[@]}" >> "$3/stdout.txt" 2>&1
