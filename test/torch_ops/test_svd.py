import sys
import os
import unittest
import logging
import copy
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)


class TestSvdOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_tensor_empty(self):
        shape_list = [(0, 0), (7, 0), (5, 3, 0), (7, 5, 0, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            cpu_u, cpu_s, cpu_v = torch.svd(x)
            mlu_u, mlu_s, mlu_v = torch.svd(self.to_mlu(x))
            self.assertEqual(cpu_u.shape, mlu_u.shape)
            self.assertEqual(cpu_s.shape, mlu_s.shape)
            self.assertEqual(cpu_v.shape, mlu_v.shape)

    def _test_svd_helper(self, shape, some, device, dtype):
        cpu_tensor = torch.randn(shape, device="cpu").to(dtype)
        device_tensor = cpu_tensor.to(device=device)
        cpu_result = torch.svd(cpu_tensor, some=some)
        device_result = torch.svd(device_tensor, some=some)
        m = min(cpu_tensor.shape[-2:])
        # torch.svd returns torch.return_types.svd which is a tuple of (U, V, S).
        # - When some==False, U[..., m:] can be arbitrary.
        # - When some==True, U shape: [..., m], V shape: [m, m]
        # - Signs are not deterministic. If the sign of a column of U is changed
        #   then the corresponding column of the V has to be changed.
        # Thus here we only compare result[..., :m].abs() from CPU and device.
        for x, y in zip(cpu_result, device_result):
            # Because the diff of svd is greatly affected by the scale and data distribution,
            # MLU cannot meet the diff requirements of 1e-5, so it is temporarily set to 3e-3.
            # http://jira.cambricon.com/browse/CNNLCORE-1225
            # http://jira.cambricon.com/browse/CNNLCORE-1248
            # self.assertEqual(x[..., :m].abs(), y[..., :m].abs(), atol=3e-3, rtol=0)
            self.assertTensorsEqual(
                x[..., :m].abs(), y[..., :m].abs().cpu(), 0.003, use_MSE=True
            )

    def _test_svd_helper_not_dense(self, shape, some, device, dtype):
        cpu_tensor = torch.empty(0)
        if len(shape) == 2:
            cpu_tensor = torch.randn(shape, device="cpu").to(dtype)[
                :, : int(shape[-1] / 2)
            ]
        elif len(shape) == 3:
            cpu_tensor = torch.randn(shape, device="cpu").to(dtype)[
                :, :, : int(shape[-1] / 2)
            ]
        elif len(shape) == 4:
            cpu_tensor = torch.randn(shape, device="cpu").to(dtype)[
                :, :, :, : int(shape[-1] / 2)
            ]
        device_tensor = cpu_tensor.to(device=device)
        cpu_result = torch.svd(cpu_tensor, some=some)
        device_result = torch.svd(device_tensor, some=some)
        m = min(cpu_tensor.shape[-2:])
        for x, y in zip(cpu_result, device_result):
            self.assertEqual(x[..., :m].abs(), y[..., :m].abs(), atol=3e-3, rtol=0)

    # @unittest.skip("not test")
    @testinfo()
    def test_svd_square(self, device="mlu", dtype=torch.float):
        self._test_svd_helper((10, 10), True, device, dtype)
        self._test_svd_helper((10, 10), False, device, dtype)
        self._test_svd_helper_not_dense((10, 20), True, device, dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_svd_tall_some(self, device="mlu", dtype=torch.float):
        self._test_svd_helper((20, 5), True, device, dtype)
        self._test_svd_helper_not_dense((20, 10), True, device, dtype)
        self._test_svd_helper((5, 20), True, device, dtype)
        self._test_svd_helper_not_dense((5, 40), True, device, dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_svd_tall_all(self, device="mlu", dtype=torch.float):
        self._test_svd_helper((20, 5), False, device, dtype)
        self._test_svd_helper_not_dense((20, 10), False, device, dtype)
        self._test_svd_helper((5, 20), False, device, dtype)
        self._test_svd_helper_not_dense((5, 40), False, device, dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_svd_some_3d(self, device="mlu", dtype=torch.float):
        self._test_svd_helper((5, 7, 3), True, device, dtype)
        self._test_svd_helper_not_dense((5, 7, 6), True, device, dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_svd_tall_all_4d(self, device="mlu", dtype=torch.float):
        self._test_svd_helper((7, 5, 3, 7), True, device, dtype)
        self._test_svd_helper_not_dense((7, 5, 3, 7 * 2), True, device, dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_svd_out(self):
        shape_list = [(5, 7, 3), (7, 5, 3, 3)]
        dtype = torch.float
        channel_first = [True, False]
        for shape in shape_list:
            for channel in channel_first:
                x = torch.randn(shape, dtype=dtype)
                resu, ress, resv = torch.svd(x)
                if channel is False:
                    x = self.convert_to_channel_last(x)
                outu = torch.tensor((), dtype=dtype).to("mlu")
                outs = torch.tensor((), dtype=dtype).to("mlu")
                outv = torch.tensor((), dtype=dtype).to("mlu")
                torch.svd(self.to_mlu(x), out=(outu, outs, outv))
                m = min(x.shape[-2:])
                self.assertEqual(
                    resu[..., :m].abs(), outu[..., :m].abs(), atol=3e-3, rtol=0
                )
                self.assertEqual(
                    ress[..., :m].abs(), outs[..., :m].abs(), atol=3e-3, rtol=0
                )
                self.assertEqual(
                    resv[..., :m].abs(), outv[..., :m].abs(), atol=3e-3, rtol=0
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_svd_backward(self):
        # To test the functionality of the svd op backward, we set a specific random seed
        # to ensure that the U and V do not appear randomness even include symbols.
        # See http://jira.cambricon.com/browse/PYTORCH-8235 for more details.
        torch.manual_seed(123456)
        shape_list = [(0, 0), (7, 0), (0, 3, 5), (3, 3), (2, 4), (4, 2)]
        for shape in shape_list:
            input = torch.randn(shape, dtype=torch.float, requires_grad=True)
            m = min(input.shape[-2:])
            U, S, V = torch.svd(input)
            U_mlu, S_mlu, V_mlu = torch.svd(input.mlu())
            self.assertTensorsEqual(
                U[..., :m].abs(), U_mlu.cpu()[..., :m].abs(), 0.003, use_MSE=True
            )
            self.assertTensorsEqual(
                S[..., :m].abs(), S_mlu.cpu()[..., :m].abs(), 0.003, use_MSE=True
            )
            self.assertTensorsEqual(
                V[..., :m].abs(), V_mlu.cpu()[..., :m].abs(), 0.003, use_MSE=True
            )
            # [PYTORCH-11396] cpu output sign is unstable, only if the output sign of cpu and mlu are consistent,
            # the backward can be tested. And S(Singular) is always non-negative, so U and V, only one
            # of them needs to be tried.
            try:
                self.assertTensorsEqual(
                    U[..., :m], U_mlu.cpu()[..., :m], 0.003, use_MSE=True
                )
            except Exception:
                continue
            U_grad = torch.randn(U.shape, dtype=torch.float)
            S_grad = torch.randn(S.shape, dtype=torch.float)
            V_grad = torch.randn(V.shape, dtype=torch.float)
            U_grad_mlu = U_grad.mlu()
            S_grad_mlu = S_grad.mlu()
            V_grad_mlu = V_grad.mlu()
            U.backward(U_grad, retain_graph=True)
            input_U_grad = copy.deepcopy(input.grad)
            input.grad.zero_()
            S.backward(S_grad, retain_graph=True)
            input_S_grad = copy.deepcopy(input.grad)
            input.grad.zero_()
            V.backward(V_grad)
            input_V_grad = copy.deepcopy(input.grad)
            input.grad.zero_()
            U_mlu.backward(U_grad_mlu, retain_graph=True)
            input_U_grad_mlu = copy.deepcopy(input.grad)
            input.grad.zero_()
            S_mlu.backward(S_grad_mlu, retain_graph=True)
            input_S_grad_mlu = copy.deepcopy(input.grad)
            input.grad.zero_()
            V_mlu.backward(V_grad_mlu)
            input_V_grad_mlu = copy.deepcopy(input.grad)
            self.assertTensorsEqual(
                input_U_grad[..., :m],
                input_U_grad_mlu.cpu()[..., :m],
                0.003,
                use_MSE=True,
            )
            self.assertTensorsEqual(
                input_S_grad[..., :m],
                input_S_grad_mlu.cpu()[..., :m],
                0.003,
                use_MSE=True,
            )
            self.assertTensorsEqual(
                input_V_grad[..., :m],
                input_V_grad_mlu.cpu()[..., :m],
                0.003,
                use_MSE=True,
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_svd_bfloat16(self, device="mlu", dtype=torch.bfloat16):
        # To test the functionality of the svd op backward, we set a specific random seed
        # to ensure that the U and V do not appear randomness even include symbols.
        # See http://jira.cambricon.com/browse/PYTORCH-8235 for more details.
        torch.manual_seed(123456)
        shape_list = [(0, 0), (7, 0), (0, 3, 5), (3, 3), (2, 4), (4, 2)]
        for shape in shape_list:
            input = torch.randn(shape, dtype=torch.float, requires_grad=True)
            m = min(input.shape[-2:])
            U, S, V = torch.svd(input)
            U_mlu, S_mlu, V_mlu = torch.svd(input.to(torch.bfloat16).mlu())
            self.assertTensorsEqual(
                U[..., :m].abs(),
                U_mlu.cpu().float()[..., :m].abs(),
                0.003,
                use_MSE=True,
            )
            self.assertTensorsEqual(
                S[..., :m].abs(),
                S_mlu.cpu().float()[..., :m].abs(),
                0.003,
                use_MSE=True,
            )
            self.assertTensorsEqual(
                V[..., :m].abs(),
                V_mlu.cpu().float()[..., :m].abs(),
                0.003,
                use_MSE=True,
            )
            # [PYTORCH-11396] cpu output sign is unstable, only if the output sign of cpu and mlu are consistent,
            # the backward can be tested. And S(Singular) is always non-negative, so U and V, only one
            # of them needs to be tried.
            try:
                self.assertTensorsEqual(
                    U[..., :m], U_mlu.cpu()[..., :m], 0.003, use_MSE=True
                )
            except Exception:
                continue
            U_grad = torch.randn(U.shape, dtype=torch.float)
            S_grad = torch.randn(S.shape, dtype=torch.float)
            V_grad = torch.randn(V.shape, dtype=torch.float)
            U_grad_mlu = U_grad.to(torch.bfloat16).mlu()
            S_grad_mlu = S_grad.to(torch.bfloat16).mlu()
            V_grad_mlu = V_grad.to(torch.bfloat16).mlu()
            U.backward(U_grad, retain_graph=True)
            input_U_grad = copy.deepcopy(input.grad)
            input.grad.zero_()
            S.backward(S_grad, retain_graph=True)
            input_S_grad = copy.deepcopy(input.grad)
            input.grad.zero_()
            V.backward(V_grad)
            input_V_grad = copy.deepcopy(input.grad)
            input.grad.zero_()
            U_mlu.backward(U_grad_mlu, retain_graph=True)
            input_U_grad_mlu = copy.deepcopy(input.grad)
            input.grad.zero_()
            S_mlu.backward(S_grad_mlu, retain_graph=True)
            input_S_grad_mlu = copy.deepcopy(input.grad)
            input.grad.zero_()
            V_mlu.backward(V_grad_mlu)
            input_V_grad_mlu = copy.deepcopy(input.grad)
            self.assertTensorsEqual(
                input_U_grad[..., :m],
                input_U_grad_mlu.cpu().float()[..., :m],
                0.003,
                use_MSE=True,
            )
            self.assertTensorsEqual(
                input_S_grad[..., :m],
                input_S_grad_mlu.cpu().float()[..., :m],
                0.003,
                use_MSE=True,
            )
            self.assertTensorsEqual(
                input_V_grad[..., :m],
                input_V_grad_mlu.cpu().float()[..., :m],
                0.003,
                use_MSE=True,
            )


if __name__ == "__main__":
    unittest.main()
