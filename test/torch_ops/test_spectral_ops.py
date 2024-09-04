import unittest
import logging
import itertools
from itertools import product
from contextlib import contextmanager
from packaging import version
from typing import Optional, List
import sys
import os
import librosa
import numpy as np
import torch


cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

has_scipy_fft = False
try:
    import scipy.fft

    has_scipy_fft = True
except ModuleNotFoundError:
    pass

REFERENCE_NORM_MODES = (
    (None, "forward", "backward", "ortho")
    if version.parse(np.__version__) >= version.parse("1.20.0")
    and (
        not has_scipy_fft or version.parse(scipy.__version__) >= version.parse("1.6.0")
    )
    else (None, "ortho")
)


class TestSpectralOps(TestCase):
    def _test_stft(self):
        def librosa_stft(x, n_fft, hop_len, center, pad_mode, normalized):
            input_1d = x.dim() == 1
            if input_1d:
                x = x.view(1, -1)
            result = []
            for xi in x:
                ri = librosa.stft(
                    xi.cpu().numpy(),
                    n_fft=n_fft,
                    hop_length=hop_len,
                    window=np.ones(n_fft),
                    center=center,
                    pad_mode=pad_mode,
                )
                result.append(torch.from_numpy(np.stack([ri.real, ri.imag], -1)))
            result = torch.stack(result, 0)
            if input_1d:
                result = result[0]
            if normalized:
                result /= n_fft**0.5
            return result

        in_sizes_list = [(10,), (10, 4000), (15, 28800), (16, 18720)]
        dtype_list = [torch.half, torch.float, torch.double]
        n_fft_list = [7, 1024, 31]
        hop_length_list = [None, 160]
        center_list = [True, False]
        pad_mode_list = ["constant", "reflect"]  # currently only support constant pad
        normalized_list = [False, True]
        onesided_list = [True]  # currently only support onesided test
        return_complex_list = [False, True]
        list_list = [
            in_sizes_list,
            dtype_list,
            n_fft_list,
            hop_length_list,
            center_list,
            pad_mode_list,
            normalized_list,
            onesided_list,
            return_complex_list,
        ]
        for (
            in_sizes,
            dtype,
            n_fft,
            hop_len,
            center,
            pad_mode,
            normalized,
            onesided,
            return_complex,
        ) in product(*list_list):
            # CNNLFFT only supports dimensions whose sizes are powers of two
            # when computing in half precision
            if dtype == torch.half:
                in_sizes = (14,)
                n_fft = 8
            if n_fft > in_sizes[-1]:
                continue

            x = torch.randn(*in_sizes, dtype=dtype).to("mlu")

            ref_result = librosa_stft(x, n_fft, hop_len, center, pad_mode, normalized)

            result0 = x.stft(
                n_fft,
                hop_len,
                center=center,
                pad_mode=pad_mode,
                normalized=normalized,
                onesided=onesided,
                return_complex=return_complex,
            )
            if result0.is_complex():
                result0 = torch.view_as_real(result0)
            self.assertTensorsEqual(
                result0.float().cpu(), ref_result.float(), 0.003, use_MSE=True
            )

            # test not contiguous
            result1 = self.to_non_dense(x).stft(
                n_fft,
                hop_len,
                center=center,
                pad_mode=pad_mode,
                normalized=normalized,
                onesided=onesided,
                return_complex=return_complex,
            )
            if result1.is_complex():
                result1 = torch.view_as_real(result1)
            self.assertTensorsEqual(result0.float().cpu(), result1.float().cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_stft(self):
        self._test_stft()

    # @unittest.skip("not test")
    @testinfo()
    def test_stft_exception(self):
        x = torch.randn((10,)).to("mlu")
        ref_msg = "CNNL FFT currently only support onesided"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            o = torch.stft(
                x, 7, pad_mode="constant", onesided=False, return_complex=False
            )  # pylint: disable=W0612
        x = torch.randn(16384).to("mlu")
        ref_msg = r"when the length of FFT \> 4096, the length must can be factorized "
        ref_msg += r"into 2 \^ m \* l, and l \<\= 4096"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            o = torch.stft(
                x, 4097, pad_mode="constant", return_complex=False
            )  # pylint: disable=W0612
        x = torch.randn(0, 16).to("mlu")
        ref_msg = "currently do not support empty Tensor input"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            o = torch.stft(
                x, 8, center=False, return_complex=False
            )  # pylint: disable=W0612
        x = torch.randn(10).to("mlu").half()
        ref_msg = (
            r"CNNL FFT only supports dimensions whose sizes are powers of "
            + r"two when computing in half precision, but got a signal size of\[7\]"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.stft(x, 7, return_complex=False)

    # @unittest.skip("not test")
    @testinfo()
    def test_cnfft_plan_cache(self):
        @contextmanager
        def plan_cache_max_size(device, n):
            if device is None:
                plan_cache = torch.backends.mlu.cnfft_plan_cache
            else:
                plan_cache = torch.backends.mlu.cnfft_plan_cache[device]
            original = plan_cache.max_size
            plan_cache.max_size = n
            yield
            plan_cache.max_size = original

        with plan_cache_max_size(
            0, max(1, torch.backends.mlu.cnfft_plan_cache.size - 10)
        ):
            self._test_stft()

        with plan_cache_max_size(0, 0):
            self._test_stft()

        torch.backends.mlu.cnfft_plan_cache.clear()

        # check that stll works after clearing cache
        with plan_cache_max_size(0, 10):
            self._test_stft()

        with self.assertRaisesRegex(RuntimeError, r"must be non-negative"):
            torch.backends.mlu.cnfft_plan_cache.max_size = -1

        with self.assertRaisesRegex(RuntimeError, r"read-only property"):
            torch.backends.mlu.cnfft_plan_cache.size = -1

        with self.assertRaisesRegex(RuntimeError, r"but got device with index"):
            torch.backends.mlu.cnfft_plan_cache[
                torch.mlu.device_count() + 10
            ]  # pylint: disable=W0106

        # Multimlu tests
        devices = ["mlu:" + str(d) for d in range(torch.mlu.device_count())]
        if len(devices) > 1:
            # Test that different GPU has different cache
            x0 = torch.randn(10, device=devices[0])
            x1 = x0.to(devices[1])
            self.assertTensorsEqual(
                torch.stft(x0, 7, pad_mode="constant", return_complex=False).cpu(),
                torch.stft(x1, 7, pad_mode="constant", return_complex=False).cpu(),
                0,
            )
            x0.copy_(x1)

            # Test that un-indexed `torch.backends.mlu.cnfft_plan_cache` uses current device
            with plan_cache_max_size(devices[0], 10):
                with plan_cache_max_size(devices[1], 11):
                    self.assertEqual(
                        torch.backends.mlu.cnfft_plan_cache[0].max_size, 10
                    )
                    self.assertEqual(
                        torch.backends.mlu.cnfft_plan_cache[1].max_size, 11
                    )

                    self.assertEqual(
                        torch.backends.mlu.cnfft_plan_cache.max_size, 10
                    )  # default is mlu:0
                    with torch.mlu.device(devices[1]):
                        self.assertEqual(
                            torch.backends.mlu.cnfft_plan_cache.max_size, 11
                        )  # default is mlu:1
                        with torch.mlu.device(devices[0]):
                            self.assertEqual(
                                torch.backends.mlu.cnfft_plan_cache.max_size, 10
                            )  # default is mlu:0

                self.assertEqual(torch.backends.mlu.cnfft_plan_cache[0].max_size, 10)
                with torch.mlu.device(devices[1]):
                    with plan_cache_max_size(None, 11):  # default is mlu:1
                        self.assertEqual(
                            torch.backends.mlu.cnfft_plan_cache[0].max_size, 10
                        )
                        self.assertEqual(
                            torch.backends.mlu.cnfft_plan_cache[1].max_size, 11
                        )

                        self.assertEqual(
                            torch.backends.mlu.cnfft_plan_cache.max_size, 11
                        )  # default is mlu:1
                        with torch.mlu.device(devices[0]):
                            self.assertEqual(
                                torch.backends.mlu.cnfft_plan_cache.max_size, 10
                            )  # default is mlu:0
                        self.assertEqual(
                            torch.backends.mlu.cnfft_plan_cache.max_size, 11
                        )  # default is mlu:1

    # @unittest.skip("not test")
    @testinfo()
    def test_fft_round_trip(self):
        # Test that round trip through ifft(fft(x)) or irfft(rfft(x)) is the identity
        for dtype in [torch.float, torch.double, torch.complex64]:
            test_args = list(
                product(
                    # input
                    (
                        torch.randn(67, dtype=dtype),
                        torch.randn(80, dtype=dtype),
                        torch.randn(12, 14, dtype=dtype),
                        torch.randn(9, 6, 3, dtype=dtype),
                        torch.randn(4, 7, 16, 48000, dtype=dtype),
                    ),
                    # dim
                    (-1, 0),
                    # norm
                    (None, "forward", "backward", "ortho"),
                    # layout
                    (lambda x: x, self.convert_to_channel_last, self.to_non_dense),
                )
            )

            # Currently don't support real dtype
            if dtype.is_complex:
                fft_functions = [("fft", "ifft")]
            else:  # Real-only functions
                fft_functions = [("rfft", "irfft")]

            for fwd, bwd in fft_functions:
                for x, dim, norm, func in test_args:
                    o = getattr(torch.fft, fwd)(
                        func(x.mlu()), n=x.size(dim), dim=dim, norm=norm
                    )
                    # The “backward”, “forward” values were added after numpy 1.20.0
                    if norm is None or norm == "ortho":
                        o_np = torch.from_numpy(
                            getattr(np.fft, fwd)(
                                func(x).numpy(), n=x.size(dim), axis=dim, norm=norm
                            )
                        )
                        self.assertTensorsEqual(
                            torch.view_as_real(o).cpu(),
                            torch.view_as_real(o_np),
                            0.003,
                            use_MSE=True,
                        )
                    y = getattr(torch.fft, bwd)(o, n=x.size(dim), dim=dim, norm=norm)
                    # The “backward”, “forward” values were added after numpy 1.20.0
                    if norm is None or norm == "ortho":
                        o_np = torch.from_numpy(
                            getattr(np.fft, bwd)(
                                o.cpu().numpy(), n=x.size(dim), axis=dim, norm=norm
                            )
                        )
                        y_real = torch.view_as_real(y) if y.is_complex() else y
                        o_np_real = (
                            torch.view_as_real(o_np) if o_np.is_complex() else o_np
                        )
                        self.assertTensorsEqual(
                            y_real.cpu(), o_np_real, 0.003, use_MSE=True
                        )
                    if x.dtype == torch.double:
                        self.assertEqual(x.float(), y.float())
                    else:
                        self.assertEqual(x, y)

    # @unittest.skip("not test")
    @testinfo()
    def test_complex_stft_definition(self):
        def _stft_reference(x, hop_length, n_fft):
            r"""Reference stft implementation

            This doesn't implement all of torch.stft, only the STFT definition:

            .. math:: X(m, \omega) = \sum_n x[n]w[n - m] e^{-jn\omega}

            """
            X = torch.empty(
                (n_fft, (x.numel() - n_fft + hop_length) // hop_length),
                device=x.device,
                dtype=torch.complex64,
            )
            for m in range(X.size(1)):
                start = m * hop_length
                if start + n_fft > x.numel():
                    slc = torch.empty(n_fft, device=x.device, dtype=x.dtype)
                    tmp = x[start:]
                    slc[: tmp.numel()] = tmp
                else:
                    slc = x[start : start + n_fft]
                X[:, m] = torch.fft.fft(slc)
            return X

        test_args = list(
            product(
                # input
                (
                    torch.randn(600, device="mlu", dtype=torch.complex64),
                    torch.randn(807, device="mlu", dtype=torch.complex64),
                ),
                # n_fft
                (50, 27),
                # hop_length
                (10, 15),
            )
        )

        for args in test_args:
            expected = _stft_reference(args[0], args[2], args[1])
            actual = torch.stft(*args, center=False)
            self.assertEqual(actual.cpu(), expected.cpu())

    # TODO(CNNLCORE-18269): constant_pad_nd doesn't support complexfloat
    @unittest.skip("not test")
    @testinfo()
    def test_fft2_numpy(self):
        dtype_list = [torch.float, torch.complex64]
        norm_modes = REFERENCE_NORM_MODES

        # input_ndim, s
        transform_desc = [
            *product(range(2, 5), (None, (4, 10))),
        ]

        fft_functions = ["fft2", "ifft2"]

        device = "mlu"
        for dtype in dtype_list:
            for input_ndim, s in transform_desc:
                shape = itertools.islice(itertools.cycle(range(4, 9)), input_ndim)
                input = torch.randn(*shape, device=device, dtype=dtype)
                for fname, norm in product(fft_functions, norm_modes):
                    torch_fn = getattr(torch.fft, fname)
                    if "hfft" in fname:
                        if not has_scipy_fft:
                            continue  # Requires scipy to compare against
                        numpy_fn = getattr(scipy.fft, fname)
                    else:
                        numpy_fn = getattr(np.fft, fname)

                    def fn(
                        t: torch.Tensor,
                        s: Optional[List[int]],
                        dim: List[int] = (-2, -1),
                        norm: Optional[str] = None,
                    ):
                        return torch_fn(t, s, dim, norm)

                    torch_fns = (torch_fn, torch.jit.script(fn))

                    # Once with dim defaulted
                    input_np = input.cpu().numpy()
                    expected = numpy_fn(input_np, s, norm=norm)
                    for fn in torch_fns:
                        actual = fn(input, s, norm=norm)
                        self.assertEqual(actual, expected, exact_dtype=False)

                    # Once with explicit dims
                    dim = (1, 0)
                    expected = numpy_fn(input_np, s, dim, norm)
                    for fn in torch_fns:
                        actual = fn(input, s, dim, norm)
                        self.assertEqual(actual, expected, exact_dtype=False)

    # @unittest.skip("not test")
    @testinfo()
    def test_fft2_out(self):
        in_cpu = torch.randn(4, 4, 3, dtype=torch.complex64)
        out_cpu = torch.randn(4, 4, 3, dtype=torch.complex64)
        result_cpu = torch.fft.fft2(in_cpu, out=out_cpu)
        in_mlu = in_cpu.mlu()
        out_mlu = out_cpu.mlu()
        result_mlu = torch.fft.fft2(in_mlu, out=out_mlu)
        self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0.003)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003)

    # @unittest.skip("not test")
    @testinfo()
    def test_fft_c2c_exception(self):
        x = torch.randn(0, 16, device="mlu", dtype=torch.complex64)
        ref_msg = "currently do not support empty Tensor input"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            o = torch.fft.fft(x)  # pylint: disable=W0612
        x = torch.randn(48001, device="mlu", dtype=torch.complex64)
        ref_msg = r"when the length of FFT \> 4096, the length must can be factorized "
        ref_msg += r"into 2 \^ m \* l, and l \<\= 4096"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            o = torch.fft.fft(x)  # pylint: disable=W0612


if __name__ == "__main__":
    unittest.main()
