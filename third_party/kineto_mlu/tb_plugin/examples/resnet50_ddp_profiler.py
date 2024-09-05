import os

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision
import torchvision.transforms as T
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models

try:
    import torch_mlu
    USE_GPU = False
    device_activitity = torch.profiler.ProfilerActivity.MLU
except:
    USE_GPU = True
    device_activitity = torch.profiler.ProfilerActivity.CUDA

def example(rank, use_gpu=True):
    if use_gpu:
        torch.cuda.set_device(rank)
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.cuda()
        cudnn.benchmark = True
        model = DDP(model, device_ids=[rank])
    else:
        torch.mlu.set_device(rank)
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.mlu()
        model = DDP(model, device_ids=[rank])

    # Use gradient compression to reduce communication
    # model.register_comm_hook(None, default.fp16_compress_hook)
    # or
    # state = powerSGD_hook.PowerSGDState(process_group=None,matrix_approximation_rank=1,start_powerSGD_iter=2)
    # model.register_comm_hook(state, powerSGD_hook.powerSGD_hook)

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, sampler=train_sampler,
                                              shuffle=False, num_workers=4)

    if use_gpu:
        criterion = nn.CrossEntropyLoss()
        criterion.cuda()
    else:
        criterion = nn.CrossEntropyLoss()
        criterion.mlu()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            device_activitity],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=2),
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./result'),
        record_shapes=True
    ) as p:
        for step, data in enumerate(trainloader, 0):
            print("step:{}".format(step))
            if use_gpu:
                inputs, labels = data[0].to('cuda'), data[1].to('cuda')
            else:
                inputs, labels = data[0].to('mlu'), data[1].to('mlu')
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            p.step()
            if step + 1 >= 8:
                break


def init_process(rank, size, fn):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    backend = 'nccl' if USE_GPU else 'cncl'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, USE_GPU)


if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, example))
        p.start()
        processes.append(p)

    timeout = 100
    for rank, p in enumerate(processes):
        p.join(timeout)
        if p.is_alive():
            print("Timeout waiting for rank %d to terminate" % rank)
            p.terminate()
