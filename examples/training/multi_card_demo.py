import os
import sys
import torch.multiprocessing as mp
import torch
import torch_mlu  # pylint: disable=C0411, W0611
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_utils import find_free_port
from torchvision.datasets import FakeData
from torchvision import transforms
from torchvision import models
import unittest

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../test")
from common_utils import testinfo, TestCase


def spawn_processes(world_size, func):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    mp.spawn(func, args=(world_size,), nprocs=world_size, join=True)


def do_broadcast(rank, world_size):
    torch.mlu.set_device(rank)

    # initialize the process group
    dist.init_process_group(backend="cncl", rank=rank, world_size=world_size)

    # broadcast operation
    t = torch.randn(10).to("mlu")
    src = 0  # broadcast from card 0 to other cards
    dist.broadcast(t, src)

    # destroy the process group
    dist.destroy_process_group()
    sys.exit(0)


def do_allreduce(rank, world_size):
    torch.mlu.set_device(rank)

    # initialize the process group
    dist.init_process_group(backend="cncl", rank=rank, world_size=world_size)

    t = torch.randn(10).to("mlu")
    # allreduce operation, support sum/product/min/max operation
    dist.all_reduce(t, dist.ReduceOp.SUM)

    # destroy the process group
    dist.destroy_process_group()
    sys.exit(0)


def do_send_recv(rank, world_size):
    torch.mlu.set_device(rank)

    os.environ["CNCL_SEND_RECV_ENABLE"] = str(1)

    # initialize the process group
    dist.init_process_group(backend="cncl", rank=rank, world_size=world_size)

    p2p_op_list = []
    if rank == 0:
        send_tensor = torch.randn(10).to("mlu:{}".format(rank))
        send_op = dist.P2POp(dist.isend, send_tensor, 1)
        p2p_op_list.append(send_op)
    elif rank == 1:
        recv_tensor = torch.randn(10).to("mlu:{}".format(rank))
        recv_op = dist.P2POp(dist.irecv, recv_tensor, 0)
        p2p_op_list.append(recv_op)

    reqs = dist.batch_isend_irecv(p2p_op_list)
    for req in reqs:
        req.wait()
    torch.mlu.synchronize()

    # destroy the process group
    dist.destroy_process_group()
    sys.exit(0)


def do_reduce(rank, world_size):
    torch.mlu.set_device(rank)

    # initialize the process group
    dist.init_process_group(backend="cncl", rank=rank, world_size=world_size)

    t = torch.randn(10).to("mlu")
    # allreduce operation, support sum/product/min/max operation
    dist.reduce(t, 0, dist.ReduceOp.SUM)

    # destroy the process group
    dist.destroy_process_group()
    sys.exit(0)


def do_allgather(rank, world_size):
    torch.mlu.set_device(rank)

    # initialize the process group
    dist.init_process_group(backend="cncl", rank=rank, world_size=world_size)

    t = torch.randn(10).to("mlu")
    ts = [torch.randn(10).to("mlu") for i in range(2)]
    # allgather operation
    dist.all_gather(ts, t)

    # destroy the process group
    dist.destroy_process_group()
    sys.exit(0)


def do_allgather_object(rank, world_size):
    torch.mlu.set_device(rank)

    # initialize the process group
    dist.init_process_group(backend="cncl", rank=rank, world_size=world_size)

    gather_objects = ["foo", 12]
    output = [None for _ in gather_objects]

    # allgather operation
    dist.all_gather_object(output, gather_objects[rank])

    # destroy the process group
    dist.destroy_process_group()
    sys.exit(0)


def do_barrier(rank, world_size):
    torch.mlu.set_device(rank)

    # initialize the process group
    dist.init_process_group(backend="cncl", rank=rank, world_size=world_size)

    # barrier operation
    dist.barrier()

    # destroy the process group
    dist.destroy_process_group()
    sys.exit(0)


BATCH_SIZE = 16
LR = 0.01
MOMENTUM = 0.1
WEIGHT_DECAY = 0.1


def multi_card_train(rank, world_size):
    torch.mlu.set_device(rank)

    # initialize the process group
    dist.init_process_group(backend="cncl", rank=rank, world_size=world_size)

    # init dataloader
    train_dataset = FakeData(size=BATCH_SIZE * 6, transform=transforms.ToTensor())
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2
    )

    # init DDP model
    model = models.resnet50().to(rank)
    model = DDP(model, device_ids=[rank])
    model.train()
    criterion = nn.CrossEntropyLoss()
    criterion.mlu()
    optimizer = torch.optim.SGD(
        model.parameters(), LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    for _, (images, target) in enumerate(train_loader):
        images = images.to(rank, non_blocking=True)
        target = target.to(rank, non_blocking=True)
        output = model(images)  # forward propagation
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()  # backward propagation
        optimizer.step()

    # destroy the process group
    dist.destroy_process_group()
    sys.exit(0)


class TestMultiCardDemo(TestCase):
    @testinfo()
    def test_broadcast(self):
        world_size = 2  # set card number
        # collective operations based on MLU cards
        spawn_processes(world_size, do_broadcast)

    @testinfo()
    def test_allreduce(self):
        world_size = 2
        spawn_processes(world_size, do_allreduce)

    @testinfo()
    def test_do_reduce(self):
        world_size = 2
        spawn_processes(world_size, do_reduce)

    @testinfo()
    def test_do_allgather(self):
        world_size = 2
        spawn_processes(world_size, do_allgather)

    @testinfo()
    def test_do_allgather_object(self):
        world_size = 2
        spawn_processes(world_size, do_allgather_object)

    @testinfo()
    def test_do_barrier(self):
        world_size = 2
        spawn_processes(world_size, do_barrier)

    @testinfo()
    def test_do_send_recv(self):
        world_size = 2
        # point-to-point communication
        spawn_processes(world_size, do_send_recv)

    @testinfo()
    def test_multicard_train(self):
        world_size = 2
        # distributed train based on MLU cards
        spawn_processes(world_size, multi_card_train)


if __name__ == "__main__":
    unittest.main()
