from math import log
import numpy as np


def net_args(parser):
    parser.add_argument("--batch", default=16, type=int, help="batch size")
    parser.add_argument(
        "--n_channels", default=1, type=int, help="number of image channels"
    )
    parser.add_argument("--epochs", default=200000, type=int, help="maximum epochs")
    parser.add_argument(
        "--n_flow", default=32, type=int, help="number of flows in each block"
    )
    parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
    parser.add_argument(
        "--no_lu",
        action="store_true",
        help="use plain convolution instead of LU decomposed version",
    )
    parser.add_argument(
        "--affine", action="store_true", help="use affine coupling instead of additive"
    )
    parser.add_argument(
        "--tr_dq",
        action="store_true",
        help="use de-quantization during training",
    )
    parser.add_argument(
        "--te_dq",
        action="store_true",
        help="use de-quantization during testing",
    )
    parser.add_argument(
        "--te_noise", action="store_true", help="use noise during testing"
    )
    parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--img_size", default=64, type=int, help="image size")
    parser.add_argument(
        "--temp", default=0.7, type=float, help="temperature of sampling"
    )
    parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
    parser.add_argument(
        "--dataset",
        default="mnist",
        type=str,
        choices=["mnist", "fashion_mnist", "point_2d", "celeba", "ffhq_5", "cifar_horses_40", "ffhq_50", "cifar_horses_20", "cifar_horses_80", "mnist_30", "mnist_gan_all", "mnist_pad", "cifar_horses_20_top", "cifar_horses_40_top", "cifar_horses_20_top_small_lr", "cifar_horses_40_top_small_lr", "arrows_big", "arrows_small", "cifar_20_picked_inds_2", "cifar_40_picked_inds_2", "cifar_20_picked_inds_3", "cifar_40_picked_inds_3"],
        help="name of the dataset",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        choices=["cpu", "cuda:0", "cuda:1", "cuda"],
        help="device used to run the neural network",
    )
    parser.add_argument(
        "--delta",
        default=0.01,
        type=float,
        help="standard deviation of the de-quantizing noise",
    )
    return parser


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins, n_dim):
    n_pixel = image_size * image_size * n_dim

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def string_args(args):
    return ";".join(
        [f"{k}#{v}" for k, v in args.__dict__.items() if "model_path" not in k]
    )


def expspace(start: float, end: float, n_steps: int):
    space = np.linspace(np.log(start), np.log(end), n_steps)
    space = np.exp(space)
    return space


def create_deltas_sequence(start: float, stop: float):
    deltas = [start] * 50
    steps = expspace(start, stop, 16)
    for s in steps[1:]:
        deltas += [s] * 10
    return deltas
