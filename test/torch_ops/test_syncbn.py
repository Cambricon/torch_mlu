import unittest
import logging
from itertools import product
import sys
import os
import torch


cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


# torch.nn.SyncBatchNorm consists of torch.batch_norm_stats, torch.batch_norm_elemt,
# torch.batch_norm_gather_stats_with_counts, torch.batch_norm_backward_reduce and
# torch.batch_norm_backward_elemt. The realization of these ops do not have CPU version,
# so we compare the result with MLU itself mostly.
# More test cases are in test/distributions/test_distributed.py.
class TestSyncBatchNormRelatedOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_batch_norm_stats(self):
        shape_list = [
            (4, 32, 257, 257),
            (4, 256, 1, 1),
            (3, 4),
            (10, 3, 12),
            (2, 6, 8, 11, 5),
            (3, 1, 4, 5, 8, 2),
        ]
        input_dtype_list = [torch.double, torch.float, torch.half]
        input_dtype_list += [torch.bfloat16] if TEST_BFLOAT16 else []
        for shape in shape_list:
            for dtype in input_dtype_list:
                input_cpu = torch.randn(shape, dtype=dtype)
                input = input_cpu.to("mlu")
                # baseline contiguous
                mean, invstd = torch.batch_norm_stats(input, 1e-5)
                # test the accuracy of mean
                l = list(range(len(shape)))
                l.remove(1)
                mean_ref = torch.mean(input_cpu.float(), tuple(l), keepdim=False)
                self.assertTensorsEqual(
                    mean_ref.float(), mean.cpu().float(), 0.003, use_MSE=True
                )
                # test channel last input
                mean0, invstd0 = torch.batch_norm_stats(
                    self.convert_to_channel_last(input), 1e-5
                )
                self.assertTensorsEqual(mean.cpu(), mean0.cpu(), 0)
                self.assertTensorsEqual(invstd.cpu(), invstd0.cpu(), 0)
                # test not dense input
                mean0, invstd0 = torch.batch_norm_stats(self.to_non_dense(input), 1e-5)
                self.assertTensorsEqual(mean.cpu(), mean0.cpu(), 0)
                self.assertTensorsEqual(invstd.cpu(), invstd0.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_batch_norm_stats_exception(self):
        input = torch.randn(1).to("mlu")
        ref_msg = r"input.dim\(\) must be not less than 2"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = torch.batch_norm_stats(input, 0.001)  # pylint: disable=W0612
        input = torch.randint(3, 5, (2, 4)).to("mlu")
        ref_msg = "\"batch_norm_stats_mlu\" not implemented for 'Long'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = torch.batch_norm_stats(input, 0.001)  # pylint: disable=W0612
        input = torch.randn(0, 4).to("mlu")
        ref_msg = "currently do not support empty input as GPU"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = torch.batch_norm_stats(input, 0.001)  # pylint: disable=W0612

    # @unittest.skip("not test")
    @testinfo()
    def test_batch_norm_gather_stats(self):
        input = torch.randn(1, 3, 3, 3, device="mlu")
        mean, invstd = torch.batch_norm_gather_stats(
            input,
            mean=torch.ones(2, 3, device="mlu"),
            invstd=torch.ones(2, 3, device="mlu"),
            running_mean=None,
            running_var=None,
            momentum=0.1,
            eps=1e-5,
            count=2,
        )
        self.assertEqual(mean.cpu(), torch.ones(3))
        self.assertEqual(invstd.cpu(), torch.ones(3))

    # @unittest.skip("not test")
    @testinfo()
    def test_batch_norm_gather_stats_with_counts(self):
        torch.manual_seed(1)
        shape_list = [
            (4, 32, 257, 257),
            (4, 256, 1, 1),
            (3, 4),
            (10, 3, 12),
            (2, 6, 8, 11, 5),
            (3, 1, 4, 5, 8, 2),
        ]
        world_size_list = [1, 4, 0]
        # because the max dim of inputs is less than 3, we do not need to test channel last
        func_list = [lambda x: x, self.to_non_dense]
        dtype_list = [torch.float]
        dtype_list += [torch.bfloat16] if TEST_BFLOAT16 else []
        list_list = [shape_list, world_size_list, dtype_list]
        for shape, world_size, dtype in product(*list_list):
            input = torch.randn(shape, dtype=dtype).to("mlu")
            mean_all = torch.randn((world_size, shape[1])).to("mlu")
            invstd_all = torch.randn((world_size, shape[1])).to("mlu")
            running_mean = torch.randn((shape[1],), dtype=dtype).to("mlu")
            running_var = torch.randn((shape[1],), dtype=dtype).to("mlu")
            running_mean_ = running_mean.clone()
            running_var_ = running_var.clone()
            count_all = (
                torch.randn((world_size,), dtype=dtype)
                .fill_(input.numel() // input.size(1))
                .to("mlu")
            )
            # baseline contiguous
            mean, invstd = torch.batch_norm_gather_stats_with_counts(
                input,
                mean_all,
                invstd_all,
                running_mean_,
                running_var_,
                0.1,
                0.0001,
                count_all,
            )
            mean_base = mean.cpu()
            invstd_base = invstd.cpu()
            running_mean_base = running_mean_.cpu()
            running_var_base = running_var_.cpu()
            # test not dense (input is not used in caculation, so we do not need wrap input)
            for f0, f1, f2, f3, f4 in product(*[func_list for i in range(5)]):
                running_mean_ = f2(running_mean.clone())
                running_var_ = f3(running_var.clone())
                raw_ptr0 = running_mean_.data_ptr()
                raw_ptr1 = running_var_.data_ptr()
                mean, invstd = torch.batch_norm_gather_stats_with_counts(
                    input,
                    f0(mean_all),
                    f1(invstd_all),
                    running_mean_,
                    running_var_,
                    0.1,
                    0.0001,
                    f4(count_all),
                )
                self.assertTensorsEqual(mean_base, mean.cpu(), 0)
                self.assertTensorsEqual(invstd_base, invstd.cpu(), 0)
                self.assertEqual(raw_ptr0, running_mean_.data_ptr())
                self.assertEqual(raw_ptr1, running_var_.data_ptr())
                self.assertTensorsEqual(running_mean_base, running_mean_.cpu(), 0)
                self.assertTensorsEqual(running_var_base, running_var_.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_batch_norm_gather_stats_with_counts_exception(self):
        shape = (3, 4)
        world_size = 1
        input = torch.randn(shape).to("mlu")
        mean_all = torch.randn((world_size, shape[1])).to("mlu")
        invstd_all = torch.randn((world_size, shape[1])).to("mlu")
        running_mean = torch.randn((shape[1],)).to("mlu")
        running_var = torch.randn((shape[1],)).to("mlu")
        count_all = (
            torch.randn((world_size,)).fill_(input.numel() // input.size(1)).to("mlu")
        )
        ref_msg = r"mean.dim\(\) and invstd.dim\(\) must equal 2, but got 1 and 2"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            (
                mean,
                invstd,
            ) = torch.batch_norm_gather_stats_with_counts(  # pylint: disable=W0612
                input,
                mean_all.view(-1),
                invstd_all,
                running_mean,
                running_var,
                0.1,
                0.0001,
                count_all,
            )
        ref_msg = (
            r"running_mean.dim\(\), running_var.dim\(\) and counts.dim\(\) must equal 1"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            (
                mean,
                invstd,
            ) = torch.batch_norm_gather_stats_with_counts(  # pylint: disable=W0612
                input,
                mean_all,
                invstd_all,
                running_mean.unsqueeze(0),
                running_var,
                0.1,
                0.0001,
                count_all,
            )
        ref_msg = "data type of mean must equal data type of var"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            (
                mean,
                invstd,
            ) = torch.batch_norm_gather_stats_with_counts(  # pylint: disable=W0612
                input,
                mean_all.half(),
                invstd_all,
                running_mean,
                running_var,
                0.1,
                0.0001,
                count_all,
            )
        ref_msg = "data type of counts must equal data type of running_mean, but got Float and Int"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            (
                mean,
                invstd,
            ) = torch.batch_norm_gather_stats_with_counts(  # pylint: disable=W0612
                input,
                mean_all,
                invstd_all,
                running_mean,
                running_var,
                0.1,
                0.0001,
                count_all.int(),
            )
        ref_msg = "mean and invstd currently only support float data type, but got Half"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            (
                mean,
                invstd,
            ) = torch.batch_norm_gather_stats_with_counts(  # pylint: disable=W0612
                input,
                mean_all.half(),
                invstd_all.half(),
                running_mean,
                running_var,
                0.1,
                0.0001,
                count_all,
            )
        ref_msg = "\"batch_norm_update_stats_mlu\" not implemented for 'Int'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            (
                mean,
                invstd,
            ) = torch.batch_norm_gather_stats_with_counts(  # pylint: disable=W0612
                input,
                mean_all,
                invstd_all,
                running_mean.int(),
                running_var.int(),
                0.1,
                0.0001,
                count_all.int(),
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_batch_norm_elemt(self):
        shape_list = [
            (4, 32, 257, 257),
            (4, 256, 1, 1),
            (3, 4),
            (10, 3, 12),
            (2, 6, 8, 11, 5),
            (3, 1, 4, 5, 8, 2),
        ]
        func_list0 = [lambda x: x, self.to_non_dense]
        func_list1 = [lambda x: x, self.convert_to_channel_last, self.to_non_dense]
        dtype_list = [torch.half, torch.float, torch.double]
        dtype_list += [torch.bfloat16] if TEST_BFLOAT16 else []
        for shape in shape_list:
            for dtype in dtype_list:
                input = torch.randn(shape, dtype=dtype).to("mlu")
                weight = torch.randn(shape[1]).to("mlu")
                bias = torch.randn(shape[1]).to("mlu")
                mean = torch.randn(shape[1]).to("mlu")
                invstd = torch.randn(shape[1]).to("mlu")
                # baseline contiguous
                out = torch.batch_norm_elemt(input, weight, bias, mean, invstd, 0.001)
                out_base = out.cpu()
                # test not contiguous
                for f0, f1, f2, f3, f4 in product(
                    func_list1, func_list0, func_list0, func_list0, func_list0
                ):
                    inputs = f0(input), f1(weight), f2(bias), f3(mean), f4(invstd)
                    out = torch.batch_norm_elemt(*inputs, 0.001)
                    self.assertTensorsEqual(out_base, out.cpu(), 0)
                    # test out
                    out_in = torch.randn((3, 3, 3, 3), dtype=dtype).to("mlu")
                    need_cmp_ptr = False
                    if out_in.numel() >= out.numel():
                        need_cmp_ptr = True
                        raw_ptr = out_in.data_ptr()
                    torch.batch_norm_elemt(*inputs, 0.001, out=out_in)
                    if need_cmp_ptr:
                        self.assertEqual(raw_ptr, out_in.data_ptr())
                    self.assertTensorsEqual(out_base, out_in.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_batch_norm_elemt_exception(self):
        shape = (3, 4)
        input = torch.randn(shape).to("mlu")
        weight = torch.randn(shape[1]).to("mlu")
        bias = torch.randn(shape[1]).to("mlu")
        mean = torch.randn(shape[1]).to("mlu")
        invstd = torch.randn(shape[1]).to("mlu")
        ref_msg = "data type of mean and invstd must be float, but got Half and Float"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out = torch.batch_norm_elemt(  # pylint: disable=W0612
                input, weight, bias, mean.half(), invstd, 0.001
            )
        ref_msg = "data type of weight and bias must be float, but got Half and Float"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out = torch.batch_norm_elemt(  # pylint: disable=W0612
                input, weight.half(), bias, mean, invstd, 0.001
            )
        ref_msg = r"input.dim\(\) must be greater than 2"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out = torch.batch_norm_elemt(  # pylint: disable=W0612
                input.view(-1), weight, bias, mean, invstd, 0.001
            )
        ref_msg = "\"batch_norm_elementwise_mlu\" not implemented for 'Int'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out = torch.batch_norm_elemt(  # pylint: disable=W0612
                input.int(), weight, bias, mean, invstd, 0.001
            )
        out_in = torch.randn(1, dtype=torch.half).to("mlu")
        ref_msg = "data type of self and output must be equal, but got Float and Half"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out = torch.batch_norm_elemt(  # pylint: disable=W0612
                input, weight, bias, mean, invstd, 0.001, out=out_in
            )
        ref_msg = r"mean.dim\(\) and invstd.dim\(\) must equal 1, but got 2 and 1"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out = torch.batch_norm_elemt(  # pylint: disable=W0612
                input, weight, bias, mean.unsqueeze(0), invstd, 0.001
            )
        ref_msg = r"weight.dim\(\) and bias.dim\(\) must equal 1, but got 2 and 1"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out = torch.batch_norm_elemt(  # pylint: disable=W0612
                input, weight.unsqueeze(0), bias, mean, invstd, 0.001
            )
        ref_msg = "currently do not support empty input as GPU"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out = torch.batch_norm_elemt(  # pylint: disable=W0612
                torch.randn((0, 4)).to("mlu"), weight, bias, mean, invstd, 0.001
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_batch_norm_backward_reduce(self):
        shape_list = [
            (4, 32, 257, 257),
            (4, 256, 1, 1),
            (3, 4),
            (10, 3, 12),
            (2, 6, 8, 11, 5),
            (3, 1, 4, 5, 8, 2),
        ]
        func_list0 = [lambda x: x, self.to_non_dense]
        func_list1 = [lambda x: x, self.convert_to_channel_last, self.to_non_dense]
        dtype_list = [torch.half, torch.float, torch.double]
        dtype_list += [torch.bfloat16] if TEST_BFLOAT16 else []
        for shape in shape_list:
            for dtype in dtype_list:
                input = torch.randn(shape, dtype=dtype).to("mlu")
                doutput = torch.randn(shape, dtype=dtype).to("mlu")
                weight = torch.randn(shape[1]).to("mlu")
                mean = torch.randn(shape[1]).to("mlu")
                invstd = torch.randn(shape[1]).to("mlu")
                # baseline contiguous
                o0, o1, dweight, dbias = torch.batch_norm_backward_reduce(
                    doutput, input, mean, invstd, weight, True, True, True
                )
                o0_, o1_, dweight_, dbias_ = (
                    o0.cpu(),
                    o1.cpu(),
                    dweight.cpu(),
                    dbias.cpu(),
                )
                # test not contiguous (weight is not used in caculation, so we do not need wrap weight)
                for f0, f1, f2, f3 in product(
                    func_list1, func_list1, func_list0, func_list0
                ):
                    o0, o1, dweight, dbias = torch.batch_norm_backward_reduce(
                        f0(doutput),
                        f1(input),
                        f2(mean),
                        f3(invstd),
                        weight,
                        True,
                        True,
                        True,
                    )
                    self.assertTensorsEqual(o0_, o0.cpu(), 0)
                    self.assertTensorsEqual(o1_, o1.cpu(), 0)
                    self.assertTensorsEqual(dweight_, dweight.cpu(), 0)
                    self.assertTensorsEqual(dbias_, dbias.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_batch_norm_backward_reduce_exception(self):
        shape = (3, 4)
        input = torch.randn(shape).to("mlu")
        doutput = torch.randn(shape).to("mlu")
        weight = torch.randn(shape[1]).to("mlu")
        mean = torch.randn(shape[1]).to("mlu")
        invstd = torch.randn(shape[1]).to("mlu")
        ref_msg = "data type of mean and invstd must be float, but got Half and Float"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out = torch.batch_norm_backward_reduce(  # pylint: disable=W0612
                doutput, input, mean.half(), invstd, weight, True, True, True
            )
        ref_msg = "data type of weight must be float, but got Half"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out = torch.batch_norm_backward_reduce(  # pylint: disable=W0612
                doutput, input, mean, invstd, weight.half(), True, True, True
            )
        ref_msg = r"input.dim\(\) must be greater than 2"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out = torch.batch_norm_backward_reduce(  # pylint: disable=W0612
                doutput, input.view(-1), mean, invstd, weight, True, True, True
            )
        ref_msg = "\"batch_norm_backward_reduce_mlu\" not implemented for 'Int'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out = torch.batch_norm_backward_reduce(  # pylint: disable=W0612
                doutput.int(), input.int(), mean, invstd, weight, True, True, True
            )
        ref_msg = r"input.sizes\(\) and grad_out.sizes\(\) must be equal, "
        ref_msg += r"but got \[3, 4\] and \[12\]"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out = torch.batch_norm_backward_reduce(  # pylint: disable=W0612
                doutput.view(-1), input, mean, invstd, weight, True, True, True
            )
        ref_msg = "data type of input and grad_out must be equal, but got Float and Int"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out = torch.batch_norm_backward_reduce(  # pylint: disable=W0612
                doutput.int(), input, mean, invstd, weight, True, True, True
            )
        ref_msg = "currently do not support empty input as GPU"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out = torch.batch_norm_backward_reduce(  # pylint: disable=W0612
                doutput,
                torch.randn((0, 4)).to("mlu"),
                mean,
                invstd,
                weight,
                True,
                True,
                True,
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_batch_norm_backward_elemt(self):
        shape_list = [
            (4, 32, 257, 257),
            (4, 256, 1, 1),
            (3, 4),
            (10, 3, 12),
            (2, 6, 8, 11, 5),
            (3, 1, 4, 5, 8, 2),
        ]
        func_list0 = [lambda x: x, self.to_non_dense]
        func_list1 = [lambda x: x, self.convert_to_channel_last, self.to_non_dense]
        dtype_list = [torch.half, torch.float, torch.double]
        dtype_list += [torch.bfloat16] if TEST_BFLOAT16 else []
        for shape in shape_list:
            for dtype in dtype_list:
                input = torch.randn(shape, dtype=dtype).to("mlu")
                doutput = torch.randn(shape, dtype=dtype).to("mlu")
                mean = torch.randn(shape[1]).to("mlu")
                invstd = torch.randn(shape[1]).to("mlu")
                weight = torch.randn(shape[1]).to("mlu")
                sum_dy = torch.randn(shape[1]).to("mlu")
                sum_dy_xmu = torch.randn(shape[1]).to("mlu")
                count = torch.full(
                    (4,), input.numel() // input.size(1), dtype=torch.int32
                ).to("mlu")
                # baseline contiguous
                dinput = torch.batch_norm_backward_elemt(
                    doutput, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count
                )
                dinput_base = dinput.cpu()
                for f0, f1, f2, f3, f4, f5, f6, f7 in product(
                    func_list1,
                    func_list1,
                    func_list0,
                    func_list0,
                    func_list0,
                    func_list0,
                    func_list0,
                    func_list1,
                ):
                    dinput = torch.batch_norm_backward_elemt(
                        f0(doutput),
                        f1(input),
                        f2(mean),
                        f3(invstd),
                        f4(weight),
                        f5(sum_dy),
                        f6(sum_dy_xmu),
                        f7(count),
                    )
                    self.assertTensorsEqual(dinput_base, dinput.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_batch_norm_backward_elemt_exception(self):
        shape = (3, 4)
        input = torch.randn(shape).to("mlu")
        doutput = torch.randn(shape).to("mlu")
        mean = torch.randn(shape[1]).to("mlu")
        invstd = torch.randn(shape[1]).to("mlu")
        weight = torch.randn(shape[1]).to("mlu")
        sum_dy = torch.randn(shape[1]).to("mlu")
        sum_dy_xmu = torch.randn(shape[1]).to("mlu")
        count = torch.full((2,), input.numel() // input.size(1), dtype=torch.int32).to(
            "mlu"
        )
        ref_msg = (
            "data type of sum_dy and sum_dy_xmu must be float, but got Half and Float"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out = torch.batch_norm_backward_elemt(  # pylint: disable=W0612
                doutput, input, mean, invstd, weight, sum_dy.half(), sum_dy_xmu, count
            )
        ref_msg = "currently do not support empty input as GPU"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out = torch.batch_norm_backward_elemt(  # pylint: disable=W0612
                doutput,
                torch.randn((0, 4)).to("mlu"),
                mean,
                invstd,
                weight,
                sum_dy,
                sum_dy_xmu,
                count,
            )


if __name__ == "__main__":
    unittest.main()
