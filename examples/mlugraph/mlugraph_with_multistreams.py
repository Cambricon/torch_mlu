import os

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_mlu
import torch_mlu.utils.gpu_migration

torch.manual_seed(0)


class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def test(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    x_static = torch.randn(32, 20, device="cuda")
    model1 = SimpleModel(20, 10, 5).cuda()
    model2 = SimpleModel(20, 10, 5).cuda()

    # warm up
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    s1.wait_stream(torch.cuda.current_stream())
    s2.wait_stream(torch.cuda.current_stream())
    for i in range(3):
        # compute model1 first, model2 and all_reduce are parallel
        with torch.cuda.stream(s1):
            y1 = model1(x_static)
        s2.wait_stream(s1)
        with torch.cuda.stream(s2):
            y2 = model2(x_static)
        with torch.cuda.stream(s1):
            dist.all_reduce(y1)
        s1.wait_stream(s2)
    torch.cuda.current_stream().wait_stream(s2)
    torch.cuda.current_stream().wait_stream(s1)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        capture_stream = torch.cuda.current_stream()
        for i in range(10):
            y1 = model1(x_static)
            # fork stream
            s1.wait_stream(capture_stream)
            with torch.cuda.stream(s1):
                y2 = model2(x_static)

            # fork stream
            s2.wait_stream(capture_stream)
            with torch.cuda.stream(s2):
                dist.all_reduce(y1)
            # stream rejoin capture stream
            capture_stream.wait_stream(s1)
            capture_stream.wait_stream(s2)

    torch.cuda.synchronize()
    x = torch.ones(32, 20, device="cuda")
    x_static.copy_(x)
    g.replay()
    torch.cuda.synchronize()
    cleanup()


if __name__ == "__main__":
    world_size = 2
    mp.spawn(test, args=(world_size,), nprocs=world_size, join=True)
