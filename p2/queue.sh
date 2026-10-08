#!/bin/zsh
# P2 reduced kill test: 4 arms x {Hopper, Ant} x seeds 1-3, 1M steps, at most $P jobs at once.
# COLA jobs go first (slowest). A job whose done.txt says exit=0 is skipped; any other job is (re)started and
# continues from its ckpt.pt (saved after every eval), so after p2/stop.sh / a crash just run this again.
# Usage: caffeinate -i p2/queue.sh [P=8] [steps=1000000]
cd "$(dirname "$0")/.."
P=${1:-8}; STEPS=${2:-1000000}; OUT=${OUT:-runs/p2/kill}
mkdir -p $OUT
{
  for task in ant hopper; do for s in 1 2 3; do echo "cola $task $s"; done; done
  for arm in base momba perobj; do for task in ant hopper; do for s in 1 2 3; do echo "$arm $task $s"; done; done; done
} > $OUT/jobs.txt

job() {
  local arm=$1 task=$2 seed=$3 d=$OUT/$1_$2_s$3
  grep -qs '^exit=0' $d/done.txt && return
  mkdir -p $d
  if [[ $arm == cola ]]; then
    third_party/cola_compat/launch.sh $task $seed $d $STEPS 50000
  else
    local env=MO-Hopper-v2; [[ $task == ant ]] && env=MO-Ant-v2
    OMP_NUM_THREADS=1 .venv/bin/python p2/esac.py --arm $arm --env $env --seed $seed --out $d --steps $STEPS \
      >> $d/stdout.txt 2>&1
  fi
  echo "exit=$? $(date '+%F %T')" > $d/done.txt
}
export OUT STEPS
# zsh has no export -f: each xargs worker is a fresh zsh that evals job() from JOB_FN
JOB_FN=$(functions job) xargs -P $P -L 1 zsh -c 'eval "$JOB_FN"; job "$@"' _ < $OUT/jobs.txt
echo "all done $(date '+%F %T')" > $OUT/ALL_DONE.txt
