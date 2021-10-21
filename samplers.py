import numpy as np
import torch
import torchvision
from scipy.ndimage import gaussian_filter, shift
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms, datasets


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[index]

        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self):
        return self.tensors.size(0)


def generate_2D_point_image(N, noise=1.0, seed=7):
    image = torch.zeros((8, 8))
    image[4, 4] = 1.0
    image = gaussian_filter(image, 0.5)
    np.random.seed(seed)
    dataset = torch.stack(
        [
            torch.FloatTensor(
                shift(image, noise * np.random.normal(size=2) - np.array([0.5, 0.5]))
            ).reshape(1, 8, 8)
            for _ in range(N)
        ]
    )
    return dataset


def sample_data(path, batch_size, image_size, n_channels):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * n_channels, (1,) * n_channels),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=8)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)


def memory_mnist(batch_size, image_size, n_channels, return_y=False):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * n_channels, (1,) * n_channels),
        ]
    )
    data = torchvision.datasets.MNIST(
        "~/datasets/mnist/", train=True, download=True, transform=transform
    )

    train_data = CustomTensorDataset(data.data[:55000].clone(), transform=transform)
    train_val_data = CustomTensorDataset(
        data.data[50000:55000].clone(), transform=transform
    )
    val_data = CustomTensorDataset(data.data[55000:].clone(), transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
    )
    train_val_loader = torch.utils.data.DataLoader(
        train_val_data,
        batch_size=batch_size,
        shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
    )
    if not return_y:
        return train_loader, val_loader, train_val_loader
    else:
        return (
            train_loader,
            val_loader,
            train_val_loader,
            data.targets[:55000],
            data.targets[55000:],
            data.targets[50000:55000],
        )


def memory_fashion(batch_size, image_size, n_channels, return_y=False):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * n_channels, (1,) * n_channels),
        ]
    )

    data = torchvision.datasets.FashionMNIST(
        "~/datasets/fashion_mnist/", train=True, download=True, transform=transform
    )

    train_data = CustomTensorDataset(data.data[:55000].clone(), transform=transform)
    train_val_data = CustomTensorDataset(
        data.data[50000:55000].clone(), transform=transform
    )
    val_data = CustomTensorDataset(data.data[55000:].clone(), transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
    )
    train_val_loader = torch.utils.data.DataLoader(
        train_val_data,
        batch_size=batch_size,
        shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
    )
    if not return_y:
        return train_loader, val_loader, train_val_loader
    else:
        return (
            train_loader,
            val_loader,
            train_val_loader,
            data.targets[:55000],
            data.targets[55000:],
            data.targets[50000:55000],
        )


def celeba(batch_size, image_size, n_channels, return_y=False):
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root="~/datasets/celeba/train/",
            transform=transforms.Compose(
                [
                    transforms.Resize(64),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    train_val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root="~/datasets/celeba/train_val/",
            transform=transforms.Compose(
                [
                    transforms.Resize(64),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root="~/datasets/celeba/val/",
            transform=transforms.Compose(
                [
                    transforms.Resize(64),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    if not return_y:
        return train_loader, val_loader, train_val_loader
    else:
        raise ValueError("CelebA does not contain y labels")


def ffhq_5(batch_size, image_size, n_channels, return_y=False):
    def get_loader(root):
        loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=root,
                transform=transforms.Compose(
                    [
                        #transforms.Resize(64),
                        #transforms.CenterCrop(64),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        return loader
    train_loader = get_loader("~/image-generator/ffhq_5/train/")
    train_val_loader = get_loader("~/image-generator/ffhq_5/train_val/")
    val_loader = get_loader("~/image-generator/ffhq_5/val/")
    if not return_y:
        return train_loader, val_loader, train_val_loader
    else:
        raise ValueError("ffhq_5 does not contain y labels")


def cifar_horses_40(batch_size, image_size, n_channels, return_y=False):
    def get_loader(root):
        loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=root,
                transform=transforms.Compose(
                    [
                        #transforms.Resize(64),
                        #transforms.CenterCrop(64),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        return loader
    train_loader = get_loader("~/image-generator/cifar_horses_40/train/")
    train_val_loader = get_loader("~/image-generator/cifar_horses_40/train_val/")
    val_loader = get_loader("~/image-generator/cifar_horses_40/val/")
    if not return_y:
        return train_loader, val_loader, train_val_loader
    else:
        raise ValueError("cifar_horses does not contain y labels")

def ffhq_50(batch_size, image_size, n_channels, return_y=False):
    def get_loader(root):
        loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=root,
                transform=transforms.Compose(
                    [
                        #transforms.Resize(64),
                        #transforms.CenterCrop(64),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        return loader
    train_loader = get_loader("~/image-generator/ffhq_50/train/")
    train_val_loader = get_loader("~/image-generator/ffhq_50/train_val/")
    val_loader = get_loader("~/image-generator/ffhq_50/val/")
    if not return_y:
        return train_loader, val_loader, train_val_loader
    else:
        raise ValueError("ffhq_50 does not contain y labels")


def cifar_horses_20(batch_size, image_size, n_channels, return_y=False):
    def get_loader(root):
        loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=root,
                transform=transforms.Compose(
                    [
                        #transforms.Resize(64),
                        #transforms.CenterCrop(64),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        return loader
    train_loader = get_loader("~/image-generator/cifar_horses_20/train/")
    train_val_loader = get_loader("~/image-generator/cifar_horses_20/train_val/")
    val_loader = get_loader("~/image-generator/cifar_horses_20/val/")
    if not return_y:
        return train_loader, val_loader, train_val_loader
    else:
        raise ValueError("cifar_horses does not contain y labels")



def cifar_horses_80(batch_size, image_size, n_channels, return_y=False):
    def get_loader(root):
        loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=root,
                transform=transforms.Compose(
                    [
                        #transforms.Resize(64),
                        #transforms.CenterCrop(64),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        return loader
    train_loader = get_loader("~/image-generator/cifar_horses_80/train/")
    train_val_loader = get_loader("~/image-generator/cifar_horses_80/train_val/")
    val_loader = get_loader("~/image-generator/cifar_horses_80/val/")
    if not return_y:
        return train_loader, val_loader, train_val_loader
    else:
        raise ValueError("cifar_horses does not contain y labels")



def mnist_30(batch_size, image_size, n_channels, return_y=False):
    def get_loader(root):
        loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=root,
                transform=transforms.Compose(
                    [
                        #transforms.ToPILImage(),
                        #transforms.Resize(64),
                        transforms.Pad(2),
                        #transforms.CenterCrop(64),
                        transforms.Grayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        return loader
    train_loader = get_loader("~/datasets/mnist_30/train/")
    train_val_loader = get_loader("~/datasets/mnist_30/train_val/")
    val_loader = get_loader("~/datasets/mnist_30/val/")
    if not return_y:
        return train_loader, val_loader, train_val_loader
    else:
        raise ValueError("cifar_horses does not contain y labels")



def mnist_gan_all(batch_size, image_size, n_channels, return_y=False):
    def get_loader(root):
        loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=root,
                transform=transforms.Compose(
                    [
                        #transforms.ToPILImage(),
                        #transforms.Resize(64),
                        transforms.Pad(2),
                        #transforms.CenterCrop(64),
                        transforms.Grayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        return loader
    train_loader = get_loader("~/datasets/mnist_gan_all/train/")
    train_val_loader = get_loader("~/datasets/mnist_gan_all/train_val/")
    val_loader = get_loader("~/datasets/mnist_gan_all/val/")
    if not return_y:
        return train_loader, val_loader, train_val_loader
    else:
        raise ValueError("cifar_horses does not contain y labels")



def mnist_pad(batch_size, image_size, n_channels, return_y=False):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * n_channels, (1,) * n_channels),
        ]
    )
    data = torchvision.datasets.MNIST(
        "~/datasets/mnist/", train=True, download=True, transform=transform
    )

    train_data = CustomTensorDataset(data.data[:55000].clone(), transform=transform)
    train_val_data = CustomTensorDataset(
        data.data[50000:55000].clone(), transform=transform
    )
    val_data = CustomTensorDataset(data.data[55000:].clone(), transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
    )
    train_val_loader = torch.utils.data.DataLoader(
        train_val_data,
        batch_size=batch_size,
        shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
    )
    if not return_y:
        return train_loader, val_loader, train_val_loader
    else:
        return (
            train_loader,
            val_loader,
            train_val_loader,
            data.targets[:55000],
            data.targets[55000:],
            data.targets[50000:55000],
        )


def cifar_horses_20_top(batch_size, image_size, n_channels, return_y=False):
    def get_loader(root):
        loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=root,
                transform=transforms.Compose(
                    [
                        #transforms.Resize(64),
                        #transforms.CenterCrop(64),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        return loader
    dataset_path = "~/datasets/cifar_horses_20_top"
    train_loader = get_loader(f"{dataset_path}/train/")
    train_val_loader = get_loader(f"{dataset_path}/train_val/")
    val_loader = get_loader(f"{dataset_path}/val/")
    if not return_y:
        return train_loader, val_loader, train_val_loader
    else:
        raise ValueError("cifar_horses does not contain y labels")



def cifar_horses_40_top(batch_size, image_size, n_channels, return_y=False):
    def get_loader(root):
        loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=root,
                transform=transforms.Compose(
                    [
                        #transforms.Resize(64),
                        #transforms.CenterCrop(64),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        return loader
    dataset_path = "~/datasets/cifar_horses_40_top"
    train_loader = get_loader(f"{dataset_path}/train/")
    train_val_loader = get_loader(f"{dataset_path}/train_val/")
    val_loader = get_loader(f"{dataset_path}/val/")
    if not return_y:
        return train_loader, val_loader, train_val_loader
    else:
        raise ValueError("cifar_horses does not contain y labels")


def cifar_horses_20_top_small_lr(batch_size, image_size, n_channels, return_y=False):
    def get_loader(root):
        loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=root,
                transform=transforms.Compose(
                    [
                        #transforms.Resize(64),
                        #transforms.CenterCrop(64),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        return loader
    dataset_path = "~/datasets/cifar_horses_20_top"
    train_loader = get_loader(f"{dataset_path}/train/")
    train_val_loader = get_loader(f"{dataset_path}/train_val/")
    val_loader = get_loader(f"{dataset_path}/val/")
    if not return_y:
        return train_loader, val_loader, train_val_loader
    else:
        raise ValueError("cifar_horses does not contain y labels")



def cifar_horses_40_top_small_lr(batch_size, image_size, n_channels, return_y=False):
    def get_loader(root):
        loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=root,
                transform=transforms.Compose(
                    [
                        #transforms.Resize(64),
                        #transforms.CenterCrop(64),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        return loader
    dataset_path = "~/datasets/cifar_horses_40_top"
    train_loader = get_loader(f"{dataset_path}/train/")
    train_val_loader = get_loader(f"{dataset_path}/train_val/")
    val_loader = get_loader(f"{dataset_path}/val/")
    if not return_y:
        return train_loader, val_loader, train_val_loader
    else:
        raise ValueError("cifar_horses does not contain y labels")


def arrows_big(batch_size, image_size, n_channels, return_y=False):
    def get_loader(root):
        loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=root,
                transform=transforms.Compose(
                    [
                        #transforms.ToPILImage(),
                        #transforms.Resize(64),
                        #transforms.Pad(2),
                        transforms.Grayscale(),
                        #transforms.CenterCrop(64),
                        #transforms.Grayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        return loader
    dataset_path = '~/datasets/arrows_big'
    train_loader = get_loader(f'{dataset_path}/train/')
    train_val_loader = get_loader(f'{dataset_path}/train_val/')
    val_loader = get_loader(f'{dataset_path}/val/')

    if not return_y:
        return train_loader, val_loader, train_val_loader
    else:
        raise ValueError("arrows big does not contain y labels")



def arrows_small(batch_size, image_size, n_channels, return_y=False):
    def get_loader(root):
        loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=root,
                transform=transforms.Compose(
                    [
                        #transforms.ToPILImage(),
                        #transforms.Resize(64),
                        transforms.Pad((2, 2, 3, 3)),
                        transforms.Grayscale(),
                        #transforms.CenterCrop(64),
                        #transforms.Grayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        return loader
    dataset_path = '~/datasets/arrows_small'
    train_loader = get_loader(f'{dataset_path}/train/')
    train_val_loader = get_loader(f'{dataset_path}/train_val/')
    val_loader = get_loader(f'{dataset_path}/val/')

    if not return_y:
        return train_loader, val_loader, train_val_loader
    else:
        raise ValueError("arrows big does not contain y labels")



def cifar_20_picked_inds_2(batch_size, image_size, n_channels, return_y=False):
    def get_loader(root):
        loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=root,
                transform=transforms.Compose(
                    [
                        #transforms.Resize(64),
                        #transforms.CenterCrop(64),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        return loader
    dataset_path = "~/datasets/cifar_20_picked_inds_2"
    train_loader = get_loader(f"{dataset_path}/train/")
    train_val_loader = get_loader(f"{dataset_path}/train_val/")
    val_loader = get_loader(f"{dataset_path}/val/")
    if not return_y:
        return train_loader, val_loader, train_val_loader
    else:
        raise ValueError("cifar_horses does not contain y labels")


def cifar_40_picked_inds_2(batch_size, image_size, n_channels, return_y=False):
    def get_loader(root):
        loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=root,
                transform=transforms.Compose(
                    [
                        #transforms.Resize(64),
                        #transforms.CenterCrop(64),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        return loader
    dataset_path = "~/datasets/cifar_40_picked_inds_2"
    train_loader = get_loader(f"{dataset_path}/train/")
    train_val_loader = get_loader(f"{dataset_path}/train_val/")
    val_loader = get_loader(f"{dataset_path}/val/")
    if not return_y:
        return train_loader, val_loader, train_val_loader
    else:
        raise ValueError("cifar_horses does not contain y labels")


cifar_20_picked_inds_3 = cifar_20_picked_inds_2
cifar_40_picked_inds_3 = cifar_40_picked_inds_2
