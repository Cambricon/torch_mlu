import time
import torch
import torch.nn as nn  # pylint: disable=R0402
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms  # pylint: disable=R0402
import torchvision.models as models  # pylint: disable=R0402
from torch.autograd import Variable
from torchvision.datasets import FakeData

import torch_mlu
import unittest
import sys
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../test")
from common_utils import testinfo, TestCase


def train(batch_size, lr, momentum, weight_decay):
    # set MLU device number
    torch.mlu.set_device(0)

    # init dataloader
    train_dataset = FakeData(size=batch_size, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=None, sampler=None, num_workers=4
    )

    # init model
    model = models.resnet50()
    # copy model weights to MLU device
    model.mlu()

    # set model into training mode
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr, momentum=momentum, weight_decay=weight_decay
    )

    # copy the parameters of Loss layer to MLU device
    criterion.mlu()

    for _, (images, target) in enumerate(train_loader):
        images = Variable(images.float(), requires_grad=False)
        # copy input images to MLU device
        images = images.to("mlu", non_blocking=True)
        target = target.to("mlu", non_blocking=True)
        # forward propagation
        output = model(images)
        loss = criterion(output, target)
        optimizer.zero_grad()
        # backward propagation
        loss.backward()
        optimizer.step()


class TestSingleCardDemo(TestCase):
    @testinfo()
    def test_single_card_demo(self):
        start_time = time.time()
        # enter the arguments in the order: batch size, learning rate, momentum, weight decay
        train(16, 0.9, 0.1, 0.1)
        use_time = time.time() - start_time
        print("use time", use_time)


if __name__ == "__main__":
    unittest.main()
