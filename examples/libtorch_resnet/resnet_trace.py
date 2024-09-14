import os
import sys
import torch
import torchvision

import unittest
import torch.testing


class TestJitTrace(unittest.TestCase):
    def test_jit_trace_load_script(self):
        model = torchvision.models.resnet18().eval()
        example_input = torch.rand(1, 3, 244, 244)

        output = model(example_input)

        resnet_model = torch.jit.trace(model, example_input)
        torch.jit.save(resnet_model, "resnet_model.pt")
        assert os.path.isfile("./resnet_model.pt")

        trace_model = torch.jit.load("./resnet_model.pt")
        trace_output = trace_model(example_input)
        torch.testing.assert_close(trace_output, output)

        script_model = torch.jit.script(model)
        script_output = script_model(example_input)
        torch.testing.assert_close(script_output, output)


if __name__ == "__main__":
    unittest.main()
