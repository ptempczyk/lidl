for d in `cat ./parameters/mnist.deltas`
do
    python train.py --delta="$d" `cat ./parameters/mnist.params` --device cuda:0
done
