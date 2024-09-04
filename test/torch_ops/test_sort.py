from __future__ import print_function
import sys
import os
import unittest
import logging
import random
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
)  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_sort(self):
        shape_list = [
            (64,),
            (76, 102),
            (32, 43, 54),
            (2, 3, 4, 5),
            (32, 1, 51, 43),
            (2, 1, 4, 5, 6),
            (32, 16, 51, 43, 52),
        ]
        type_list = [
            torch.long,
            torch.int,
            torch.short,
            torch.float,
            torch.half,
            torch.double,
            torch.int8,
            torch.uint8,
        ]
        order = [True, False]
        channel_first = [True, False]
        for i in shape_list:
            for typeId in type_list:
                x = torch.randn(i, dtype=torch.float).to(typeId)
                random_i = random.randint(0, 1)
                is_descending = order[random_i]
                channel = channel_first[random_i]
                dim = random.randint(-len(i), len(i) - 1)
                out_cpu = torch.sort(x, dim, descending=is_descending)
                if channel is False:
                    x = self.convert_to_channel_last(x)
                out_mlu = torch.sort(x.to("mlu"), dim, descending=is_descending)
                self.assertTensorsEqual(
                    out_cpu[0].float(),
                    out_mlu[0].cpu().float().contiguous(),
                    0.0,
                    use_MSE=True,
                )
                self.assertTrue(out_cpu[1].dtype, out_mlu[1].cpu().dtype)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_sort_bfloat16(self):
        shape_list = [
            (64,),
            (76, 102),
            (32, 43, 54),
            (2, 3, 4, 5),
            (32, 1, 51, 43),
            (2, 1, 4, 5, 6),
            (32, 16, 51, 43, 52),
        ]
        type_list = [torch.bfloat16]
        order = [True, False]
        channel_first = [True, False]
        for i in shape_list:
            for typeId in type_list:
                x = torch.randn(i, dtype=torch.float).to(typeId)
                random_i = random.randint(0, 1)
                is_descending = order[random_i]
                channel = channel_first[random_i]
                dim = random.randint(-len(i), len(i) - 1)
                out_cpu = torch.sort(x, dim, descending=is_descending)
                if channel is False:
                    x = self.convert_to_channel_last(x)
                out_mlu = torch.sort(x.to("mlu"), dim, descending=is_descending)
                self.assertTensorsEqual(
                    out_cpu[0].float(),
                    out_mlu[0].cpu().float().contiguous(),
                    0.0,
                    use_MSE=True,
                )
                self.assertTrue(out_cpu[1].dtype, out_mlu[1].cpu().dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_sort_out(self):
        shape_list = [(64,), (76, 124), (32, 43, 54), (32, 16, 51, 43)]
        type_list = [
            torch.long,
            torch.int,
            torch.short,
            torch.float,
            torch.half,
            torch.double,
            torch.int8,
            torch.uint8,
        ]
        for i in shape_list:
            local_value = list(range(-len(i), len(i)))
            for typeId in type_list:
                x = torch.randn(i, dtype=torch.float).to(typeId)
                for dim in local_value:
                    for descending_true in [False, True]:
                        values = torch.randn(i, dtype=torch.float).to(typeId)
                        indices = torch.randint(-10, 10, i)
                        torch.sort(
                            x, dim, descending=descending_true, out=(values, indices)
                        )
                        values_mlu = values.to("mlu")
                        indices_mlu = indices.to("mlu")
                        torch.sort(
                            x.to("mlu"),
                            dim,
                            descending=descending_true,
                            out=(values_mlu, indices_mlu),
                        )
                        self.assertTensorsEqual(
                            values.float(), values_mlu.cpu().float(), 0.0, use_MSE=True
                        )
                        self.assertTrue(indices.dtype, indices_mlu.cpu().dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_sort_zero_dim(self):
        shape_list = [()]
        type_list = [
            torch.long,
            torch.int,
            torch.short,
            torch.float,
            torch.half,
            torch.double,
            torch.int8,
            torch.uint8,
        ]
        order = [True, False]
        for i in shape_list:
            local_value = [-1, 0]
            for typeId in type_list:
                x = torch.randn(i, dtype=torch.float).to(typeId)
                for dim in local_value:
                    for is_descending in order:
                        out_cpu = torch.sort(x, dim, descending=is_descending)
                        out_mlu = torch.sort(x.to("mlu"), dim, descending=is_descending)
                        self.assertTensorsEqual(
                            out_cpu[0].float(),
                            out_mlu[0].cpu().float(),
                            0.0,
                            use_MSE=True,
                        )
                        self.assertTrue(out_cpu[1].dtype, out_mlu[1].cpu().dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_sort_no_dense(self):
        shape_list = [
            (2, 3, 4, 2, 1, 7, 8 * 2),
            (20, 30, 40, 50 * 2),
            (20, 30, 40, 50, 10 * 2),
            (2, 3, 4 * 2),
            (7, 300 * 2),
            (20, 26258 * 2),
        ]
        dim_list = [6, 1, 1, 1, -1, 1]
        for i in range(len(shape_list)):  # pylint: disable=C0200
            x = torch.randn(shape_list[i], dtype=torch.float)[
                ..., : int(shape_list[i][-1] / 2)
            ]
            out_cpu = torch.sort(x, dim_list[i])
            out_mlu = torch.sort(self.to_mlu(x), dim_list[i])
            self.assertTensorsEqual(
                out_cpu[0].float(),
                out_mlu[0].cpu().float().contiguous(),
                0.0,
                use_MSE=True,
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_sort_stable(self):
        shape_list = [
            (64,),
            (76, 102),
            (32, 43, 54),
            (2, 3, 4, 5),
            (32, 1, 51, 43),
            (2, 1, 4, 5, 6),
            (32, 16, 51, 43, 52),
        ]
        type_list = [
            torch.long,
            torch.int,
            torch.short,
            torch.float,
            torch.half,
            torch.double,
            torch.int8,
            torch.uint8,
        ]
        order = [True, False]
        channel_first = [True, False]
        stable = [True, False]
        for i in shape_list:
            for typeId in type_list:
                x = torch.randn(i, dtype=torch.float).to(typeId)
                random_i = random.randint(0, 1)
                is_descending = order[random_i]
                channel = channel_first[random_i]
                is_stable = stable[random_i]
                dim = random.randint(-len(i), len(i) - 1)
                out_cpu = torch.sort(
                    x, stable=is_stable, dim=dim, descending=is_descending
                )
                if channel is False:
                    x = self.convert_to_channel_last(x)
                out_mlu = torch.sort(
                    x.to("mlu"), stable=is_stable, dim=dim, descending=is_descending
                )
                self.assertTensorsEqual(
                    out_cpu[0].float(),
                    out_mlu[0].cpu().float().contiguous(),
                    0.0,
                    use_MSE=True,
                )
                self.assertTrue(out_cpu[1].dtype, out_mlu[1].cpu().dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_sort_out_stable(self):
        shape_list = [
            (64,),
            (76, 102),
            (32, 43, 54),
            (2, 3, 4, 5),
            (32, 1, 51, 43),
            (2, 1, 4, 5, 6),
            (32, 16, 51, 43, 52),
        ]
        type_list = [
            torch.long,
            torch.int,
            torch.short,
            torch.float,
            torch.half,
            torch.double,
            torch.int8,
            torch.uint8,
        ]
        order = [True, False]
        channel_first = [True, False]
        stable = [True, False]
        for i in shape_list:
            for typeId in type_list:
                x = torch.randn(i, dtype=torch.float).to(typeId)
                random_i = random.randint(0, 1)
                is_descending = order[random_i]
                channel = channel_first[random_i]
                is_stable = stable[random_i]
                dim = random.randint(-len(i), len(i) - 1)
                sorted_tensor = torch.zeros(i).to(typeId)
                indices = torch.zeros(i).long()
                out_cpu = torch.sort(
                    x,
                    stable=is_stable,
                    dim=dim,
                    descending=is_descending,
                    out=(sorted_tensor, indices),
                )
                if channel is False:
                    x = self.convert_to_channel_last(x)
                out_mlu = torch.sort(
                    x.to("mlu"), stable=is_stable, dim=dim, descending=is_descending
                )
                self.assertTensorsEqual(
                    out_cpu[0].float(),
                    out_mlu[0].cpu().float().contiguous(),
                    0.0,
                    use_MSE=True,
                )
                self.assertTrue(out_cpu[1].dtype, out_mlu[1].cpu().dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_exception(self):
        x = torch.tensor([1.0, 2.0], dtype=torch.complex64).mlu()
        ref_msg = r"Sort currently does not support ComplexFloat dtypes on MLU."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.sort(x)


if __name__ == "__main__":
    unittest.main()
