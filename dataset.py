import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


def getdata(batch_size=16):
    transform_trainset = transforms.Compose(
        [transforms.RandomRotation(15, expand=True),
         transforms.RandomCrop(24),
         transforms.Pad(4),
         transforms.Grayscale(1),
         transforms.ToTensor(),  # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.
                                 # FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ]                                 # Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels,
                                           # this transform will normalize each channel of the input
                                           # input[channel] = (input[channel] - mean[channel]) / std[channel]
    )
    transform_testset = transforms.Compose(
        [transforms.Pad(2),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ]
    )
    trainset = datasets.MNIST(root='./data', train=True,
                              download=True, transform=transform_trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = datasets.MNIST(root='/data', train=False,
                             download=True, transform=transform_testset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    return trainset, trainloader, testset, testloader


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# a, b, c, d = getdata()
# print(len(a))
# print(len(b))
# inputs, labels = next(iter(d))
# print('input.shape is {}'.format(inputs.shape))
# print('labels is {}'.format(labels.shape))
# out = torchvision.utils.make_grid(inputs)  # put a batch of images together
# imshow(out)


