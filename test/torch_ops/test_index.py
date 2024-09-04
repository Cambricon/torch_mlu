from __future__ import print_function

import sys
import os
import unittest
import logging

import copy
import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    read_card_info,
)

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestIndexOps(TestCase):
    def _generate_tensor(self, shape, dtype, value=16777216):  # pylint: disable=R0201
        if dtype == torch.bool:
            out = torch.randint(2, shape).type(dtype)
        else:
            # TODO(liuyuxin): test negative number in future.  # pylint: disable=W0511
            out = torch.randint(value, shape).type(dtype)
        return out

    # @unittest.skip("not test")
    @testinfo()
    def test_index_bool(self):
        dtype_lst = [
            (torch.long),
            (torch.float),
            (torch.int),
            (torch.short),
            (torch.int8),
            (torch.uint8),
            (torch.double),
        ]
        for dtype1 in dtype_lst:
            for shape1, shape2 in [
                ((4, 1), (4, 1)),
                ((2, 2), (2, 2)),
                ((1, 0), (1, 0)),
                ((0,), (0,)),
                ((4, 1, 1), (4, 1)),
            ]:
                input = self._generate_tensor(shape1, dtype1)
                indice_1 = self._generate_tensor(shape2, torch.bool, value=shape1[0])
                input_mlu = self.to_device(input)
                result_cpu = input[indice_1]
                result_mlu = input_mlu[self.to_device(indice_1)]
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

            for shape1, shape2 in [((4, 1), (4,))]:
                input = self._generate_tensor(shape1, dtype1)
                indice_1 = self._generate_tensor(shape2, torch.bool, value=shape1[0])
                input_mlu = self.to_device(input)
                result_cpu = input[indice_1, :]
                result_mlu = input_mlu[self.to_device(indice_1)]
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

            for shape1, shape2 in [((5, 5), (5,))]:
                input = self._generate_tensor(shape1, dtype1)
                input_mlu = self.to_device(input)
                indice_1 = torch.ones(shape2).bool()
                indice_2 = torch.ones(shape2).bool()
                result_cpu = input[indice_1, indice_2]
                result_mlu = input_mlu[
                    self.to_device(indice_1), self.to_device(indice_2)
                ]
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

            for shape1, shape2 in [((13230, 85), (32, 13230))]:
                input = self._generate_tensor(shape1, dtype1)
                indice_1 = self._generate_tensor(shape2, torch.bool, value=shape1[0])
                input_mlu = self.to_device(input)
                result_cpu = input[indice_1[3]]
                result_mlu = input_mlu[self.to_device(indice_1)[3]]
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_index_bfloat16(self):
        dtype_lst = [(torch.bfloat16)]
        for dtype1 in dtype_lst:
            for shape1, shape2 in [
                ((4, 1), (4, 1)),
                ((2, 2), (2, 2)),
                ((1, 0), (1, 0)),
                ((0,), (0,)),
                ((4, 1, 1), (4, 1)),
            ]:
                input = self._generate_tensor(shape1, dtype1)
                indice_1 = self._generate_tensor(shape2, torch.bool, value=shape1[0])
                input_mlu = self.to_device(input)
                result_cpu = input[indice_1]
                result_mlu = input_mlu[self.to_device(indice_1)]
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

            for shape1, shape2 in [((4, 1), (4,))]:
                input = self._generate_tensor(shape1, dtype1)
                indice_1 = self._generate_tensor(shape2, torch.bool, value=shape1[0])
                input_mlu = self.to_device(input)
                result_cpu = input[indice_1, :]
                result_mlu = input_mlu[self.to_device(indice_1)]
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

            for shape1, shape2 in [((5, 5), (5,))]:
                input = self._generate_tensor(shape1, dtype1)
                input_mlu = self.to_device(input)
                indice_1 = torch.ones(shape2).bool()
                indice_2 = torch.ones(shape2).bool()
                result_cpu = input[indice_1, indice_2]
                result_mlu = input_mlu[
                    self.to_device(indice_1), self.to_device(indice_2)
                ]
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

            for shape1, shape2 in [((13230, 85), (32, 13230))]:
                input = self._generate_tensor(shape1, dtype1)
                indice_1 = self._generate_tensor(shape2, torch.bool, value=shape1[0])
                input_mlu = self.to_device(input)
                result_cpu = input[indice_1[3]]
                result_mlu = input_mlu[self.to_device(indice_1)[3]]
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_bool_indices(self):
        v = torch.randn(5, 7, 3).to("mlu")
        boolIndices = torch.tensor(
            [True, False, True, True, False], dtype=torch.bool
        ).to("mlu")
        self.assertEqual(v[boolIndices].shape, (3, 7, 3))

    # @unittest.skip("not test")
    @testinfo()
    def test_multiple_bool_indices(self):
        v = torch.randn(5, 7, 3).to("mlu")
        # note: these broadcast together and are transposed to the first dim
        mask1 = torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool).to("mlu")
        mask2 = torch.tensor([1, 1, 1], dtype=torch.bool).to("mlu")
        self.assertEqual(v[mask1, :, mask2].shape, (3, 7))

    # @unittest.skip("not test")
    @testinfo()
    def test_combile_bool_int_indices(self):
        a = torch.randn(5, 7).to("mlu")
        mask = torch.tensor([False, True, True, True, False], dtype=torch.bool).to(
            "mlu"
        )
        indices = torch.tensor([1, 1, 1], dtype=torch.long).to("mlu")
        self.assertEqual(a[mask, indices].shape, [3])

    # @unittest.skip("not test")
    @testinfo()
    def test_index_long(self):
        dtype_lst = [
            (torch.long),
            (torch.float),
            (torch.int),
            (torch.short),
            (torch.int8),
            (torch.uint8),
            (torch.double),
        ]
        for dtype1 in dtype_lst:
            for shape1, shape2 in [
                ((4,), (4,)),
                ((217413,), (0,)),
                ((1, 0), (1, 0)),
                ((4, 4), (0,)),
            ]:
                # TODO(liuyuxin): support indices != input.dim() in future.  # pylint: disable=W0511
                input = self._generate_tensor(shape1, dtype1)
                indice_1 = self._generate_tensor(shape2, torch.long, value=shape1[0])
                input_mlu = self.to_device(input)
                result_cpu = input[indice_1]
                result_mlu = input_mlu[self.to_device(indice_1)]
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

            for shape1, shape2 in [((546, 300), (546,)), ((1, 300), (10,))]:
                input = self._generate_tensor(shape1, dtype1)
                input_mlu = self.to_device(input)
                indice_1 = self._generate_tensor(shape2, torch.long, value=shape1[0])
                indice_2 = self._generate_tensor(shape2, torch.long, value=shape1[1])
                result_cpu = input[indice_1, indice_2]
                result_mlu = input_mlu[
                    self.to_device(indice_1), self.to_device(indice_2)
                ]
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

                input_mlu = self.to_device(input)
                indice_1 = self._generate_tensor(shape2, torch.long, value=shape1[0])
                result_cpu = input[indice_1, :]
                result_mlu = input_mlu[self.to_device(indice_1), :]
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

                input_mlu = self.to_device(input)
                indice_1 = self._generate_tensor(shape2, torch.long, value=shape1[1])
                result_cpu = input[:, indice_1]
                result_mlu = input_mlu[:, self.to_device(indice_1)]
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

            for shape1, shape2, shape3 in [
                ((2, 182400, 4), (2, 1), (2, 2000)),
                ((1, 163200, 4), (1, 1), (1, 2000)),
            ]:
                input = self._generate_tensor(shape1, dtype1)
                input_mlu = self.to_device(input)
                indice_1 = self._generate_tensor(shape2, torch.long, value=shape1[0])
                indice_2 = self._generate_tensor(shape3, torch.long, value=shape1[1])
                result_cpu = input[indice_1, indice_2]
                result_mlu = input_mlu[
                    self.to_device(indice_1), self.to_device(indice_2)
                ]
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

            for shape1, shape2 in [((16, 3, 13, 13, 85), (132,))]:
                input = torch.randn(shape1, dtype=torch.float, requires_grad=True)
                input_mlu = torch.nn.Parameter(self.to_device(copy.deepcopy(input)))
                input_mlu_with_cpu_indices = torch.nn.Parameter(
                    self.to_device(copy.deepcopy(input))
                )
                indice_1 = self._generate_tensor(shape2, torch.long, value=shape1[0])
                indice_2 = self._generate_tensor(shape2, torch.long, value=shape1[1])
                indice_3 = self._generate_tensor(shape2, torch.long, value=shape1[2])
                indice_4 = self._generate_tensor(shape2, torch.long, value=shape1[3])
                result_cpu = input[indice_1, indice_2, indice_3, indice_4]
                grad = torch.randn(result_cpu.shape, dtype=torch.float)
                grad_mlu = self.to_device(grad)
                result_cpu.backward(grad)
                grad_out = copy.deepcopy(input.grad)
                result_mlu = input_mlu[
                    self.to_device(indice_1),
                    self.to_device(indice_2),
                    self.to_device(indice_3),
                    self.to_device(indice_4),
                ]
                result_mlu_with_cpu_indices = input_mlu_with_cpu_indices[
                    indice_1, indice_2, indice_3, indice_4
                ]
                result_mlu.backward(grad_mlu)
                result_mlu_with_cpu_indices.backward(grad_mlu)
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)
                self.assertTensorsEqual(
                    result_cpu, result_mlu_with_cpu_indices.cpu(), 0
                )
                self.assertTensorsEqual(grad_out, input_mlu.grad.cpu(), 1e-6)
                self.assertTensorsEqual(
                    grad_out, input_mlu_with_cpu_indices.grad.cpu(), 1e-6
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_index_channels_last(self):
        # test for bool indices
        for shape1, shape2 in [((2, 2, 2, 2), (2, 2, 2, 2)), ((4, 4, 4, 4), (4, 4))]:
            input = self._generate_tensor(shape1, torch.float)
            input = input.to(memory_format=torch.channels_last)
            indice_1 = self._generate_tensor(shape2, torch.bool, value=shape1[0])
            if indice_1.dim() == 4:
                indice_1 = indice_1.to(memory_format=torch.channels_last)
            input_mlu = self.to_device(input)
            result_cpu = input[indice_1]
            result_mlu = input_mlu[self.to_device(indice_1)]
            self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

        # test for long indices
        for shape1, shape2 in [
            ((16, 3, 13, 13), (132,)),
            ((16, 2, 13, 13), (2, 2, 2, 2)),
        ]:
            input = self._generate_tensor(shape1, torch.float)
            input = input.to(memory_format=torch.channels_last)
            indice_1 = self._generate_tensor(shape2, torch.long, value=shape1[0])
            if indice_1.dim() == 4:
                indice_1 = indice_1.to(memory_format=torch.channels_last)
            input_mlu = self.to_device(input)
            result_cpu = input[indice_1]
            result_mlu = input_mlu[self.to_device(indice_1)]
            self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

        # test for not dense bool input & indices
        for shape1, shape2 in [((4, 4), (4, 4))]:
            input = self._generate_tensor(shape1, torch.float)
            indice_1 = self._generate_tensor(shape2, torch.bool, value=shape1[0])
            input_mlu = self.to_device(input)[:2, :2]
            result_cpu = input[:2, :2][indice_1[:2, :2]]
            result_mlu = input_mlu[self.to_device(indice_1)[:2, :2]]
            self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

        # test for not dense long input & indices
        for shape1, shape2 in [((16, 3, 13, 13), (132,))]:
            input = self._generate_tensor(shape1, torch.float)
            input = input.to(memory_format=torch.channels_last)
            indice_1 = self._generate_tensor(shape2, torch.long, value=shape1[0])
            if indice_1.dim() == 4:
                indice_1 = indice_1.to(memory_format=torch.channels_last)
            # use slice not index for not dense tensor
            input_mlu = self.to_device(input)[:, :, :, :10]
            result_cpu = input[:, :, :, :10][indice_1[:10]]
            result_mlu = input_mlu[self.to_device(indice_1)[:10]]
            self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)
            # use cpu slice not index for not dense tensor
            input_mlu = self.to_device(input)[:, :, :, :10]
            result_cpu = input[:, :, :, :10][indice_1[:10]]
            result_mlu = input_mlu[indice_1[:10]]
            self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_exception(self):
        # ref_msg_1
        input = self._generate_tensor((4, 1), torch.long)
        indice_1 = self._generate_tensor((4,), torch.long, value=4)
        indice_2 = self._generate_tensor((2,), torch.long, value=4)
        # MLU error message is different with pytorch.
        ref_msg = r"CNNL error: CNNL_STATUS_BAD_PARAM"
        input_mlu = self.to_device(input)
        indice_mlu_1 = self.to_device(indice_1)
        indice_mlu_2 = self.to_device(indice_2)
        with self.assertRaisesRegex(Exception, ref_msg):
            result = input_mlu[indice_mlu_1, indice_mlu_2]  # pylint: disable=W0612

        # ref_msg_2
        input = self._generate_tensor((4, 1), torch.long)
        indice_1 = self._generate_tensor((1, 1), torch.long, value=4)
        indice_2 = self._generate_tensor((1, 1), torch.long, value=4)
        indice_3 = self._generate_tensor((1, 1), torch.long, value=4)
        # TODO(liuyuxin): Long indice will be supported in future.  # pylint: disable=W0511
        ref_msg = r"too many indices for tensor of dimension 2"
        input_mlu = self.to_device(input)
        indice_1_mlu = self.to_device(indice_1)
        indice_2_mlu = self.to_device(indice_2)
        indice_3_mlu = self.to_device(indice_3)
        with self.assertRaisesRegex(IndexError, ref_msg):
            result = input_mlu[indice_1_mlu, indice_2_mlu, indice_3_mlu]

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("46GB", device="mlu")
    def test_index_large(self):
        dtype_lst = [(torch.long)]
        for dtype in dtype_lst:
            for shape1, shape2, shape3 in [((3000, 100, 14840), (3000, 1), (3000, 20))]:
                input = self._generate_tensor(shape1, dtype)
                indice_1 = self._generate_tensor(shape2, torch.long, value=shape1[0])
                indice_2 = self._generate_tensor(shape3, torch.long, value=shape1[1])
                input_mlu = self.to_device(input)
                result_cpu = input[indice_1, indice_2]
                result_mlu = input_mlu[
                    self.to_device(indice_1), self.to_device(indice_2)
                ]
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)


if __name__ == "__main__":
    run_tests()
