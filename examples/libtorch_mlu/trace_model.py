import torch


class ConvModel(torch.nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 2, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class AddModel(torch.nn.Module):
    def __init__(self):
        super(AddModel, self).__init__()
        self.add = torch.add
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.add(x, -0.5))


class ConvOnlyModel(torch.nn.Module):
    def __init__(self):
        super(ConvOnlyModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 2, 3)

    def forward(self, x):
        return self.conv(x)


def trace_model():
    inputs = torch.rand(4, 3, 4, 4)

    conv_model = ConvOnlyModel()
    traced_model = torch.jit.trace(conv_model, inputs)
    torch.jit.save(traced_model, "./conv_trace_model.pt")

    add_model = AddModel()
    traced_model = torch.jit.trace(add_model, inputs)
    torch.jit.save(traced_model, "./add_model.pt")
    print("Generated TorchSctipt model.")


if __name__ == "__main__":
    trace_model()
