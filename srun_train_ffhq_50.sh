source ~/glow-pytorch/venv/bin/activate
for d in `cat ./parameters/ffhq_50.deltas`
do
    srun --partition=common  --qos=16gpu3d --gres=gpu:1 --time 3-0 python train.py --delta="$d" `cat ./parameters/ffhq_50.params`  --device cuda:0   &
done
deactivate
