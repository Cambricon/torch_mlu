import torch
import torch_mlu

model = torch.nn.Linear(20, 30).mlu()
inputs = torch.randn(128, 20).mlu()
m = torch.nn.Conv2d(16, 33, 3, stride=2).mlu()
input = torch.randn(20, 16, 50, 100).mlu()
output = m(input)
with torch.mlu.profiler.profile() as prof:
    model(inputs)
