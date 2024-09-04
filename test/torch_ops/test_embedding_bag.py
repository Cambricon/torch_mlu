from __future__ import print_function

import sys
import os
from itertools import product
import copy
import unittest
import logging

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413, C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestEmbeddingOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_embedding_bag_empty(self):
        es = torch.nn.EmbeddingBag(5, 2, mode="sum", include_last_offset=False).to(
            dtype=torch.float32, device="mlu"
        )
        input = torch.tensor([], device="mlu", dtype=torch.long)
        offsets = torch.tensor([0, 0, 0, 0, 0], device="mlu", dtype=torch.long)
        per_sample_weights = torch.randn_like(input, dtype=torch.float32)
        mlu_res = es(input, offsets, per_sample_weights)
        cpu_res = torch.zeros(offsets.shape[0], 2)
        self.assertEqual(mlu_res.cpu().float(), cpu_res)

    # @unittest.skip("not test")
    @testinfo()
    def test_embedding_bag_1D(self):
        x_shape_lists = [(32,), (16,), (12,), (23,), (7200,)]
        offset_lists = [(0, 2), (0, 3), (0, 4), (0, 0, 2), (0,)]
        c_lists = [9796, 1234]
        dtype_list = [(torch.float, 1e-4), (torch.half, 3e-3)]
        for c, x_shape in product(c_lists, x_shape_lists):
            for data_type, err in dtype_list:
                for offset_ in offset_lists:
                    offset = torch.tensor(offset_, dtype=torch.int32)
                    x_0 = torch.randint(0, c, x_shape, dtype=torch.int32)
                    x = self.get_not_contiguous_tensor(copy.deepcopy(x_0))
                    matrix = torch.rand(c, 511, dtype=torch.half)
                    out_cpu = torch.nn.functional.embedding_bag(
                        x, matrix.float(), offset
                    )
                    out_mlu = torch.nn.functional.embedding_bag(
                        x.to("mlu"),
                        self.to_mlu_dtype(matrix, data_type),
                        offset.to("mlu"),
                    )
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_embedding_bag_2D(self):
        x_shape_lists = [(2, 5), (5, 8), (21, 53)]
        c_lists = [9796, 1234]
        dtype_list = [(torch.float, 1e-4), (torch.half, 3e-3)]
        for c, x_shape in product(c_lists, x_shape_lists):
            for data_type, err in dtype_list:
                x_0 = torch.randint(0, c, x_shape, dtype=torch.int32)
                x = self.get_not_contiguous_tensor(copy.deepcopy(x_0))
                matrix = torch.rand(c, 511, dtype=torch.half)
                out_cpu = torch.nn.functional.embedding_bag(x, matrix.float())
                out_mlu = torch.nn.functional.embedding_bag(
                    x.to("mlu"), self.to_mlu_dtype(matrix, data_type)
                )
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_embedding_bag_mode(self):
        x_shape_lists = [(2, 5), (5, 8), (21, 53)]
        mode_lists = ["sum", "mean", "max"]
        dtype_list = [(torch.float, 1e-4), (torch.half, 3e-2)]
        for mode in mode_lists:
            for x_shape in x_shape_lists:
                for data_type, err in dtype_list:
                    x_0 = torch.randint(0, 7200, x_shape, dtype=torch.int32)
                    x = self.get_not_contiguous_tensor(copy.deepcopy(x_0))
                    matrix = torch.rand(7200, 511, dtype=torch.half)
                    out_cpu = torch.nn.functional.embedding_bag(
                        x, matrix.float(), mode=mode
                    )
                    out_mlu = torch.nn.functional.embedding_bag(
                        x.to("mlu"), self.to_mlu_dtype(matrix, data_type), mode=mode
                    )
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_EmbeddingBag(self):
        a = torch.tensor([[1, 3, 2], [0, 2, 1]], dtype=torch.int32)
        embeddings = torch.rand(4, 3, requires_grad=True)

        embed_old = torch.nn.EmbeddingBag(4, 3)
        embed_old.weight = torch.nn.Parameter(embeddings)
        res_old = embed_old(a)

        res_F = torch.nn.functional.embedding_bag(a, embeddings)
        self.assertEqual(res_old, res_F)

        embed_old = torch.nn.EmbeddingBag(4, 3)
        embed_old = embed_old.from_pretrained(embeddings)
        res_old = embed_old(a)
        res_F = torch.nn.functional.embedding_bag(a, embeddings)

        self.assertEqual(res_old, res_F)

    # @unittest.skip("not test")
    @testinfo()
    def test_embedding_bag_backward(self):
        x_shape_lists = [(2, 4), (5, 8), (3, 16), (21, 53)]
        c_lists = [9796, 1234]
        dtype_list = [(torch.float, 1e-4), (torch.half, 3e-3), (torch.double, 1e-4)]
        idx_type_list = [torch.int, torch.long]
        for c, x_shape in product(c_lists, x_shape_lists):
            for data_type, err in dtype_list:
                for idx_type in idx_type_list:
                    x = torch.randint(0, c, x_shape).to(idx_type)
                    matrix = torch.rand(c, 3, dtype=torch.float, requires_grad=True)
                    out_cpu = torch.nn.functional.embedding_bag(x, matrix, mode="sum")
                    grad = torch.randn(out_cpu.shape, dtype=torch.float)
                    out_cpu.backward(grad)
                    g1 = copy.deepcopy(matrix.grad)
                    matrix.grad.zero_()
                    out_mlu = torch.nn.functional.embedding_bag(
                        self.to_device(x),
                        self.to_mlu_dtype(matrix, data_type),
                        mode="sum",
                    )
                    out_mlu.backward(self.to_mlu_dtype(grad, data_type))
                    g2 = copy.deepcopy(matrix.grad)
                    self.assertTensorsEqual(
                        out_cpu, out_mlu.cpu().float(), err, use_MSE=True
                    )
                    self.assertTensorsEqual(g1, g2.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_embedding_bag_exception(self):
        x = torch.randint(0, 9796, [51, 77])
        matrix = torch.rand(9796, 511).int()
        ref_msg = "\"embedding_bag_mlu\" not implemented for 'Int'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.nn.functional.embedding_bag(self.to_mlu(x), self.to_device(matrix))

        x = torch.randint(0, 9796, [51, 77]).float()
        matrix = torch.rand(9796, 511, dtype=torch.float)
        ref_msg_2 = (
            "Expected tensor for argument #1 'indices' to have one of the following"
        )
        ref_msg_2 = ref_msg_2 + " scalar types: Long, Int; but got mluFloatType instead"
        ref_msg_2 = ref_msg_2 + r" \(while checking arguments for cnnl__embedding_bag\)"
        with self.assertRaisesRegex(RuntimeError, ref_msg_2):
            torch.nn.functional.embedding_bag(self.to_device(x), self.to_mlu(matrix))

        x = torch.randint(0, 9796, (51,))
        offsets = torch.LongTensor([0, 2]).float()
        matrix = torch.rand(9796, 511, dtype=torch.float)
        ref_msg_3 = (
            "Expected tensor for argument #1 'indices' to have one of the following"
        )
        ref_msg_3 = ref_msg_3 + " scalar types: Long, Int; but got mluFloatType instead"
        ref_msg_3 = ref_msg_3 + r" \(while checking arguments for cnnl__embedding_bag\)"
        with self.assertRaisesRegex(RuntimeError, ref_msg_3):
            torch.nn.functional.embedding_bag(
                self.to_device(x), self.to_mlu(matrix), self.to_mlu(offsets)
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_embedding_bag_bfloat16(self):
        c = 9796
        x_shape = [2, 4]
        x = torch.randint(0, c, x_shape)
        matrix = torch.rand(c, 3, dtype=torch.bfloat16, requires_grad=True)
        out_cpu = torch.nn.functional.embedding_bag(x, matrix.float(), mode="sum")
        out_mlu = torch.nn.functional.embedding_bag(
            self.to_device(x), self.to_mlu(matrix), mode="sum"
        )
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_embedding_bag_backward_bfloat16(self):
        c = 9796
        x_shape = [2, 4]
        x = torch.randint(0, c, x_shape)
        matrix = torch.rand(c, 3, dtype=torch.bfloat16, requires_grad=True)
        out_cpu = torch.nn.functional.embedding_bag(x, matrix.float(), mode="sum")
        grad = torch.randn(out_cpu.shape, dtype=torch.bfloat16)
        out_cpu.backward(grad.float())
        g1 = copy.deepcopy(matrix.grad)
        matrix.grad.zero_()
        out_mlu = torch.nn.functional.embedding_bag(
            self.to_device(x), self.to_mlu(matrix), mode="sum"
        )
        out_mlu.backward(self.to_mlu(grad))
        g2 = copy.deepcopy(matrix.grad)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)
        self.assertTensorsEqual(g1, g2.cpu(), 3e-3, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
