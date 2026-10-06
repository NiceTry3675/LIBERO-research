set -x
ENV=/content/mamba/envs/rt; export PATH=$ENV/bin:$PATH
cd /content/RoboTwin
sed -e 's/^episode_num: .*/episode_num: 3/' env_cfg/task_config/demo_clean.yml > env_cfg/task_config/demo_test.yml
sed -e 's/^episode_num: .*/episode_num: 2/' env_cfg/task_config/demo_randomized.yml > env_cfg/task_config/demo_test_rand.yml
for task in beat_block_hammer place_empty_cup; do
  start=$(date +%s)
  PYTHONWARNINGS=ignore::UserWarning timeout 1500 python scripts/collect_data.py $task demo_test > /content/expert_${task}.log 2>&1
  echo "RESULT $task demo_test exit=$? seconds=$(( $(date +%s) - start ))"
  tail -5 /content/expert_${task}.log
done
if [ -d assets/background_texture ]; then
  start=$(date +%s)
  PYTHONWARNINGS=ignore::UserWarning timeout 1500 python scripts/collect_data.py place_empty_cup demo_test_rand > /content/expert_rand.log 2>&1
  echo "RESULT place_empty_cup demo_test_rand exit=$? seconds=$(( $(date +%s) - start ))"
  tail -5 /content/expert_rand.log
else
  echo "SKIP randomized run: no background textures (bootstrap without WITH_TEXTURES=1)"
fi
find data -name "*.hdf5" | head; du -sh data
echo EXPERT_DONE
