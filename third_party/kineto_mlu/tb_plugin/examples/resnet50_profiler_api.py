import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as T

import torch.profiler
from torchvision import models

try:
    import torch_mlu
    device = "mlu"
    device_activitity = torch.profiler.ProfilerActivity.MLU
except:
    device = "cuda"
    device_activitity = torch.profiler.ProfilerActivity.CUDA
    cudnn.benchmark = True

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.to(device=device)

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
# There is a known bug where with_stack=True, setting num_workers=0 or repeat=1 are workarounds.
# See https://github.com/pytorch/pytorch/issues/109969
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)

criterion = nn.CrossEntropyLoss().to(device=device)
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
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./result', worker_name='worker0'),
    record_shapes=True,
    profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
    with_stack=True
) as p:
    for step, data in enumerate(trainloader, 0):
        print("step:{}".format(step))
        inputs, labels = data[0].to(device=device), data[1].to(device=device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step + 1 >= 8:
            break
        p.step()
