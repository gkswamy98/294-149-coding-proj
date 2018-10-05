#NUM_AGENTS = 10
for ddpg in {1..10}
do
   maddpg="$((10-$ddpg))"
   python3 run.py --num-agents=10 --num-ddpg=$ddpg --num-maddpg=$maddpg --max-episode-len=10 --num-episodes=5000 --save-suffix="ddpg_$ddpg"
done