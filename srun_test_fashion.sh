source ~/virtualenvs/glow-pytorch/bin/activate
for f in `ls ~/glow-pytorch/checkpoint/model_batch#64\;n_channels#1\;epochs#200\;n_flow#32\;n_block#4\;no_lu#False\;affine#True\;n_bits#8\;lr#5e-05\;img_size#32\;temp#0.7\;n_sample#20\;dataset#fashion_mnist\;device#cuda\:0\;delta#*_40*`
do
    echo $f
    srun --partition=common --qos=16gpu3d --gres=gpu:1 bash run_test_fashion.sh $f &
    srun --partition=common --qos=16gpu3d --gres=gpu:1 bash run_test_fashion_on_mnist.sh $f &
done
deactivate
