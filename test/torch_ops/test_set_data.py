import sys
import os
import copy
import unittest
import logging

import torch
import torch_mlu
from torch import nn

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class Net(nn.Module):  # pylint: disable=W0223
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Linear(512, 2048, bias=True)

    def forward(self, x):
        output = self.features(x)
        return output


class TestSetDataOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_set_data(self):
        a = torch.randn((1, 2)).to("mlu")
        b = torch.randn((1, 2))
        b_id_saved = id(b)
        b.data = a
        self.assertTrue(b_id_saved == id(b))

        a = torch.randn((1, 2), requires_grad=True).to("mlu")
        b = torch.randn((1, 2), requires_grad=True)
        b_id_saved = id(b)
        b.data = a
        self.assertTrue(b_id_saved == id(b))

        a = torch.randn((1, 2)).to("mlu")
        b = torch.randn((1, 2))
        b = torch.abs(b)
        b_id_saved = id(b)
        b.data = a
        self.assertTrue(b_id_saved == id(b))

    # @unittest.skip("not test")
    @testinfo()
    def test_set_data_model_to(self):
        model_cpu = Net()
        id_weight_value = id(model_cpu.features.weight)
        id_bias_value = id(model_cpu.features.bias)
        model_mlu = model_cpu.to("mlu")
        self.assertEqual(id_weight_value, id(model_mlu.features.weight))
        self.assertEqual(id_bias_value, id(model_mlu.features.bias))

    # @unittest.skip("not test")
    @testinfo()
    def test_set_data_model_training(self):
        input_shape = (512, 512)
        precision = 0.003
        convert_func = [self.to_non_dense, self.convert_to_channel_last, lambda x: x]
        model_cpu = Net()
        model_mlu = copy.deepcopy(model_cpu).to("mlu")
        for convert in convert_func:
            x = torch.randn(input_shape, dtype=torch.float, requires_grad=True)
            mlu_x = torch.randn(input_shape, dtype=torch.float, requires_grad=True)
            mlu_x.data = copy.deepcopy(x).to("mlu")
            out_cpu = model_cpu(convert(x))
            out_mlu = model_mlu(convert(mlu_x))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), precision, use_MSE=True)
            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            grad_mlu = torch.randn(out_cpu.shape, dtype=torch.float)
            grad_mlu.data = copy.deepcopy(grad).to("mlu")
            out_cpu.backward(grad)
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(x.grad, mlu_x.grad.cpu(), precision, use_MSE=True)
            self.assertTensorsEqual(
                model_cpu.features.bias.grad,
                model_mlu.features.bias.grad.cpu(),
                precision,
                use_MSE=True,
            )
            self.assertTensorsEqual(
                model_cpu.features.weight.grad,
                model_mlu.features.weight.grad.cpu(),
                precision,
                use_MSE=True,
            )


if __name__ == "__main__":
    unittest.main()
