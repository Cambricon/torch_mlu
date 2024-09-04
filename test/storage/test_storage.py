from copy import deepcopy
import tempfile
import sys
import os
import logging
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_utils import bytes_to_scalar
from torch.package import PackageExporter, PackageImporter
import numpy as np

import unittest  # pylint: disable=C0411

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()

logging.basicConfig(level=logging.DEBUG)


class Storage(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_storage_clone(self):
        if torch.mlu.device_count() > 1:
            x = torch.randn(4, 4, device="mlu:1").storage()
            y = x.clone()
            self.assertTrue(x.get_device() == y.get_device())
            self.assertEqual(x.cpu(), y.cpu())
            x = torch.randn(4, 4, device="mlu:1").untyped_storage()
            y = x.clone()
            self.assertTrue(x.get_device() == y.get_device())
            self.assertEqual(x.tolist(), y.tolist())

    # @unittest.skip("not test")
    @testinfo()
    def test_serialization_array_with_storage(self):
        default_type = torch.Tensor().type()
        torch.set_default_tensor_type(torch.FloatTensor)
        x = torch.randn(5, 5).mlu()
        y = torch.IntTensor(2, 5).fill_(0).mlu()
        q = [x, y, x, y.storage()]
        with tempfile.NamedTemporaryFile() as f:
            torch.save(q, f)
            f.seek(0)
            q_copy = torch.load(f)
        self.assertEqual(q_copy, q, atol=0, rtol=0)
        q_copy[0].fill_(5)
        self.assertEqual(q_copy[0], q_copy[2], atol=0, rtol=0)
        self.assertTrue(isinstance(q_copy[0], torch.Tensor))
        self.assertTrue(isinstance(q_copy[1], torch.Tensor))
        self.assertTrue(isinstance(q_copy[2], torch.Tensor))
        self.assertTrue(isinstance(q_copy[3], torch.storage.TypedStorage))
        # now use torch.UntypedStorage to support mlu Untyped.
        self.assertTrue(isinstance(q_copy[3].untyped(), torch.UntypedStorage))
        q_copy[1].fill_(10)
        self.assertEqual(q_copy[3], torch.IntStorage(10).fill_(10))
        torch.set_default_tensor_type(default_type)

    # @unittest.skip("not test")
    @testinfo()
    def test_type(self):
        default_type = torch.Tensor().type()
        torch.set_default_tensor_type(torch.FloatTensor)
        x = torch.randn(5, 5).storage()
        self.assertIsInstance(x.mlu(), torch.TypedStorage)
        self.assertIsInstance(x.untyped().mlu(), torch.UntypedStorage)
        self.assertIsInstance(x.mlu().untyped(), torch.UntypedStorage)
        self.assertIsInstance(x.mlu().double(), torch.mlu.DoubleStorage)
        self.assertIsInstance(x.double().mlu(), torch.mlu.DoubleStorage)
        self.assertIsInstance(x.mlu().float(), torch.mlu.FloatStorage)
        self.assertIsInstance(x.mlu().char(), torch.mlu.CharStorage)
        self.assertIsInstance(x.mlu().byte(), torch.mlu.ByteStorage)
        self.assertIsInstance(x.mlu().long(), torch.mlu.LongStorage)
        self.assertIsInstance(x.mlu().short(), torch.mlu.ShortStorage)
        self.assertIsInstance(x.mlu().half(), torch.mlu.HalfStorage)
        self.assertIsInstance(x.mlu().bool(), torch.mlu.BoolStorage)
        self.assertIsInstance(x.mlu().float().cpu(), torch.FloatStorage)
        self.assertIsInstance(x.mlu().float().cpu().int(), torch.IntStorage)

        x = torch.randn(5, 5).untyped_storage()
        self.assertIsInstance(x.mlu(), torch.UntypedStorage)
        self.assertIsInstance(x.mlu().double(), torch.mlu.DoubleStorage)
        self.assertIsInstance(x.double().mlu(), torch.mlu.DoubleStorage)
        self.assertIsInstance(x.mlu().float(), torch.mlu.FloatStorage)
        self.assertIsInstance(x.mlu().char(), torch.mlu.CharStorage)
        self.assertIsInstance(x.mlu().byte(), torch.mlu.ByteStorage)
        self.assertIsInstance(x.mlu().long(), torch.mlu.LongStorage)
        self.assertIsInstance(x.mlu().short(), torch.mlu.ShortStorage)
        self.assertIsInstance(x.mlu().half(), torch.mlu.HalfStorage)
        self.assertIsInstance(x.mlu().bool(), torch.mlu.BoolStorage)
        self.assertIsInstance(x.mlu().float().cpu(), torch.FloatStorage)
        self.assertIsInstance(x.mlu().float().cpu().int(), torch.IntStorage)
        torch.set_default_tensor_type(default_type)

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_type_bfloat16(self):
        default_type = torch.Tensor().type()
        torch.set_default_tensor_type(torch.FloatTensor)
        x = torch.randn(5, 5).storage()
        self.assertIsInstance(x.mlu().bfloat16(), torch.mlu.BFloat16Storage)
        x = torch.randn(5, 5).untyped_storage()
        self.assertIsInstance(x.mlu().bfloat16(), torch.mlu.BFloat16Storage)
        torch.set_default_tensor_type(default_type)

    # @unittest.skip("not test")
    @testinfo()
    def test_has_storage(self):
        self.assertIsNotNone(torch.tensor([]).mlu().storage())
        self.assertIsNotNone(torch.empty(0).mlu().storage())
        self.assertIsNotNone(torch.tensor([]).clone().mlu().storage())
        self.assertIsNotNone(torch.tensor([0, 0, 0]).nonzero().mlu().storage())
        self.assertIsNotNone(torch.tensor([]).new().mlu().storage())

        self.assertIsNotNone(torch.tensor([]).mlu().untyped_storage())
        self.assertIsNotNone(torch.empty(0).mlu().untyped_storage())
        self.assertIsNotNone(torch.tensor([]).clone().mlu().untyped_storage())
        self.assertIsNotNone(torch.tensor([0, 0, 0]).nonzero().mlu().untyped_storage())
        self.assertIsNotNone(torch.tensor([]).new().mlu().untyped_storage())

    # @unittest.skip("not test")
    @testinfo()
    def test_tensor_set_cdata(self):
        a = torch.UntypedStorage(sequence=[4, 5, 6], device="mlu")
        b = torch.UntypedStorage(sequence=[1, 2, 3], device="mlu")
        b._set_cdata(a._cdata)
        self.assertEqual(a._cdata, b._cdata)
        self.assertEqual(a.tolist(), b.tolist())

    # @unittest.skip("not test")
    @testinfo()
    def test_tensor_set(self):
        # torch.set_default_dtype(torch.float32)
        t1 = torch.Tensor().mlu()
        t2 = torch.Tensor([3, 4, 9, 10]).mlu()
        t1.set_(t2)
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)
        size = torch.Size([9, 3, 4, 10])
        t1.set_(t2.storage(), 0, size)
        self.assertEqual(t1.size(), size)
        t1.set_(t2.storage(), 0, tuple(size))
        self.assertEqual(t1.size(), size)
        self.assertEqual(t1.stride(), (120, 40, 10, 1))
        stride = (10, 360, 90, 1)
        t1.set_(t2.storage(), 0, size, stride)
        self.assertEqual(t1.stride(), stride)
        t1.set_(t2.storage(), 0, size=size, stride=stride)
        self.assertEqual(t1.size(), size)
        self.assertEqual(t1.stride(), stride)

        t1 = torch.tensor([]).mlu()
        # 1. case when source is tensor
        t1.set_(source=t2)
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)
        # 2. case when source is storage
        t1.set_(source=t2.storage())
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)
        # 3. case when source is storage, and other args also specified
        t1.set_(source=t2.storage(), storage_offset=0, size=size, stride=stride)
        self.assertEqual(t1.size(), size)
        self.assertEqual(t1.stride(), stride)
        t1 = torch.tensor([True, True], dtype=torch.bool)
        t2 = torch.tensor([False, False], dtype=torch.bool)
        t1.set_(t2)
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)

        zero_d = torch.randn((), device="mlu")
        one_d = torch.randn((1,), device="mlu")
        zero_d_clone = zero_d.clone()
        one_d_clone = one_d.clone()
        self.assertEqual((), zero_d_clone.set_(one_d.storage(), 0, (), ()).shape)
        self.assertEqual((1,), zero_d_clone.set_(one_d.storage(), 0, (1,), (1,)).shape)
        self.assertEqual((), one_d_clone.set_(one_d.storage(), 0, (), ()).shape)
        self.assertEqual((1,), one_d_clone.set_(one_d.storage(), 0, (1,), (1,)).shape)

        self.assertEqual((), zero_d.clone().set_(zero_d).shape)
        self.assertEqual((), one_d.clone().set_(zero_d).shape)
        self.assertEqual((1,), zero_d.clone().set_(one_d).shape)
        self.assertEqual((1,), one_d.clone().set_(one_d).shape)

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_tensor_set_bfloat16(self):
        t1 = torch.Tensor().mlu().bfloat16()
        t2_bfloat16 = torch.Tensor([3, 4, 9, 10]).mlu().bfloat16()
        size = torch.Size([9, 3, 4, 10])
        t1.set_(t2_bfloat16.storage(), 0, tuple(size))
        self.assertEqual(t1.storage()._cdata, t2_bfloat16.storage()._cdata)
        self.assertEqual(t1.size(), size)
        self.assertEqual(t1.stride(), (120, 40, 10, 1))

    # @unittest.skip("not test")
    @testinfo()
    def test_storage_casts(self):
        storage = torch.IntStorage([-1, 0, 1, 2, 3, 4])
        self.assertEqual(storage.size(), 6)
        self.assertEqual(storage.tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertEqual(storage.type(), "torch.IntStorage")
        self.assertEqual(storage.short().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(storage.dtype, torch.int32)

        floatStorage = storage.float()
        self.assertEqual(floatStorage.size(), 6)
        self.assertEqual(floatStorage.tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertEqual(floatStorage.type(), "torch.FloatStorage")
        self.assertEqual(floatStorage.int().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(floatStorage.dtype, torch.float32)

        halfStorage = storage.half()
        self.assertEqual(halfStorage.size(), 6)
        self.assertEqual(halfStorage.tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertEqual(halfStorage.type(), "torch.HalfStorage")
        self.assertEqual(halfStorage.int().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(halfStorage.dtype, torch.float16)

        longStorage = storage.long()
        self.assertEqual(longStorage.size(), 6)
        self.assertEqual(longStorage.tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertEqual(longStorage.type(), "torch.LongStorage")
        self.assertEqual(longStorage.int().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(longStorage.dtype, torch.int64)

        shortStorage = storage.short()
        self.assertEqual(shortStorage.size(), 6)
        self.assertEqual(shortStorage.tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertEqual(shortStorage.type(), "torch.ShortStorage")
        self.assertEqual(shortStorage.int().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(shortStorage.dtype, torch.int16)

        doubleStorage = storage.double()
        self.assertEqual(doubleStorage.size(), 6)
        self.assertEqual(doubleStorage.tolist(), [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(doubleStorage.type(), "torch.DoubleStorage")
        self.assertEqual(doubleStorage.int().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(doubleStorage.dtype, torch.float64)

        charStorage = storage.char()
        self.assertEqual(charStorage.size(), 6)
        self.assertEqual(charStorage.tolist(), [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(charStorage.type(), "torch.CharStorage")
        self.assertEqual(charStorage.int().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(charStorage.dtype, torch.int8)

        byteStorage = storage.byte()
        self.assertEqual(byteStorage.size(), 6)
        self.assertEqual(byteStorage.tolist(), [255, 0, 1, 2, 3, 4])
        self.assertEqual(byteStorage.type(), "torch.ByteStorage")
        self.assertEqual(byteStorage.int().tolist(), [255, 0, 1, 2, 3, 4])
        self.assertIs(byteStorage.dtype, torch.uint8)

        boolStorage = storage.bool()
        self.assertEqual(boolStorage.size(), 6)
        self.assertEqual(boolStorage.tolist(), [True, False, True, True, True, True])
        self.assertEqual(boolStorage.type(), "torch.BoolStorage")
        self.assertEqual(boolStorage.int().tolist(), [1, 0, 1, 1, 1, 1])
        self.assertIs(boolStorage.dtype, torch.bool)

        complexfloat_storage = torch.ComplexFloatStorage(
            [-1, 0, 1 + 2j, 2.5j, 3.5, 4 - 2j]
        )
        self.assertEqual(complexfloat_storage.size(), 6)
        self.assertEqual(
            complexfloat_storage.tolist(), [-1, 0, 1 + 2j, 2.5j, 3.5, 4 - 2j]
        )
        self.assertEqual(complexfloat_storage.type(), "torch.ComplexFloatStorage")
        self.assertIs(complexfloat_storage.dtype, torch.complex64)

        complexdouble_storage = complexfloat_storage.complex_double()
        self.assertEqual(complexdouble_storage.size(), 6)
        self.assertEqual(
            complexdouble_storage.tolist(), [-1, 0, 1 + 2j, 2.5j, 3.5, 4 - 2j]
        )
        self.assertEqual(complexdouble_storage.type(), "torch.ComplexDoubleStorage")
        self.assertIs(complexdouble_storage.dtype, torch.complex128)

    # @unittest.skip("not test")
    @testinfo()
    def test_sizeof(self):
        sizeof_empty = torch.randn(0).mlu().storage().__sizeof__()
        sizeof_10 = torch.randn(10).mlu().storage().__sizeof__()
        sizeof_100 = torch.randn(100).mlu().storage().__sizeof__()
        self.assertEqual((sizeof_100 - sizeof_empty) // (sizeof_10 - sizeof_empty), 10)
        self.assertEqual((sizeof_100 - sizeof_empty) % (sizeof_10 - sizeof_empty), 0)

        sizeof_empty = torch.randn(0).to(torch.uint8).mlu().storage().__sizeof__()
        sizeof_10 = torch.randn(10).to(torch.uint8).mlu().storage().__sizeof__()
        sizeof_100 = torch.randn(100).to(torch.uint8).mlu().storage().__sizeof__()
        self.assertEqual((sizeof_100 - sizeof_empty) // (sizeof_10 - sizeof_empty), 10)
        self.assertEqual((sizeof_100 - sizeof_empty) % (sizeof_10 - sizeof_empty), 0)

        sizeof_empty = torch.randn(0).mlu().untyped_storage().__sizeof__()
        sizeof_10 = torch.randn(10).mlu().untyped_storage().__sizeof__()
        sizeof_100 = torch.randn(100).mlu().untyped_storage().__sizeof__()
        self.assertEqual((sizeof_100 - sizeof_empty) // (sizeof_10 - sizeof_empty), 10)
        self.assertEqual((sizeof_100 - sizeof_empty) % (sizeof_10 - sizeof_empty), 0)

        sizeof_empty = (
            torch.randn(0).to(torch.uint8).mlu().untyped_storage().__sizeof__()
        )
        sizeof_10 = torch.randn(10).to(torch.uint8).mlu().untyped_storage().__sizeof__()
        sizeof_100 = (
            torch.randn(100).to(torch.uint8).mlu().untyped_storage().__sizeof__()
        )
        self.assertEqual((sizeof_100 - sizeof_empty) // (sizeof_10 - sizeof_empty), 10)
        self.assertEqual((sizeof_100 - sizeof_empty) % (sizeof_10 - sizeof_empty), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_storage(self):
        element_size_dtypes = {
            torch.float64: torch.float32,
            torch.complex128: torch.complex64,
        }
        dtypes = [
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.bool,
            torch.float32,
            torch.complex64,
        ]
        for dtype in dtypes:
            v = make_tensor((3, 5), dtype=dtype, device="cpu", low=-9, high=9).mlu()
            # now mlu cast 64bits to 32.
            if dtype in element_size_dtypes.keys():
                v = v._to(element_size_dtypes[dtype])
            self.assertEqual(v.storage()[0], v[0][0])
            self.assertEqual(v.storage()[14], v[2][4])
            v_s = v.storage()

            for el_num in range(v.numel()):
                dim0 = el_num // v.size(1)
                dim1 = el_num % v.size(1)
                self.assertEqual(v_s[el_num], v[dim0][dim1])

            v_s_byte = v.storage().untyped()
            el_size = v.element_size()

            if dtype == torch.complex64:
                v_s_byte = v_s_byte.cpu()
                v = v.cpu()

            for el_num in range(v.numel()):
                start = el_num * el_size
                end = start + el_size
                dim0 = el_num // v.size(1)
                dim1 = el_num % v.size(1)
                self.assertEqual(
                    bytes_to_scalar(v_s_byte[start:end], dtype, "mlu"), v[dim0][dim1]
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_deepcopy(self):
        a = torch.randn(5, 5, dtype=torch.float32, device="mlu")
        b = torch.randn(5, 5, dtype=torch.float32, device="mlu")
        c = a.view(25)
        q = [a, [a.storage(), b.storage()], b, c]
        w = deepcopy(q)
        self.assertEqual(w[0], q[0], atol=0, rtol=0)
        self.assertEqual(w[1][0], q[1][0], atol=0, rtol=0)
        self.assertEqual(w[1][1], q[1][1], atol=0, rtol=0)
        self.assertEqual(w[1], q[1], atol=0, rtol=0)
        self.assertEqual(w[2], q[2], atol=0, rtol=0)

    # @unittest.skip("not test")
    @testinfo()
    def test_has_storage_numpy(self):
        for dtype in [np.float32, np.float64, np.int64, np.int32, np.int16, np.uint8]:
            arr = np.array([1], dtype=dtype)
            self.assertIsNotNone(
                torch.tensor(arr, device="mlu", dtype=torch.float32).storage()
            )
            self.assertIsNotNone(
                torch.tensor(arr, device="mlu", dtype=torch.double).storage()
            )
            self.assertIsNotNone(
                torch.tensor(arr, device="mlu", dtype=torch.int).storage()
            )
            self.assertIsNotNone(
                torch.tensor(arr, device="mlu", dtype=torch.long).storage()
            )
            self.assertIsNotNone(
                torch.tensor(arr, device="mlu", dtype=torch.uint8).storage()
            )

            self.assertIsNotNone(
                torch.tensor(arr, device="mlu", dtype=torch.float32).untyped_storage()
            )
            self.assertIsNotNone(
                torch.tensor(arr, device="mlu", dtype=torch.double).untyped_storage()
            )
            self.assertIsNotNone(
                torch.tensor(arr, device="mlu", dtype=torch.int).untyped_storage()
            )
            self.assertIsNotNone(
                torch.tensor(arr, device="mlu", dtype=torch.long).untyped_storage()
            )
            self.assertIsNotNone(
                torch.tensor(arr, device="mlu", dtype=torch.uint8).untyped_storage()
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_storage_dtype(self):
        x = torch.tensor([], device="mlu")
        self.assertEqual(x.dtype, x.storage().dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_storage_pin_memory_and_is_pinned(self):
        x = torch.tensor((2,)).to("mlu")
        x_storage = x.storage()
        self.assertFalse(x_storage.is_pinned())

        x_UntypedStorage = torch.UntypedStorage([1, 2], device="cpu")
        y_UntypedStorage = x_UntypedStorage.pin_memory()
        self.assertFalse(x_UntypedStorage.is_pinned())
        self.assertTrue(y_UntypedStorage.is_pinned())

    # @unittest.skip("not test")
    @testinfo()
    def test_storage_resize(self):
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        x = torch.tensor([2.2, 3.1], device="mlu").storage()
        self.assertEqual(x.size(), 2)
        x_clone = x.clone()
        x.resize_(4)
        self.assertEqual(x.size(), 4)
        self.assertEqual(x[0], x_clone[0], atol=1.0e-12, rtol=1.0e-12)
        self.assertEqual(x[1], x_clone[1], atol=1.0e-12, rtol=1.0e-12)

        x = torch.tensor([2.2, 3.1], device="mlu").untyped_storage()
        self.assertEqual(x.size(), 16)
        x_clone = x.clone()
        x.resize_(32)
        self.assertEqual(x.size(), 32)
        self.assertEqual(x[0], x_clone[0], atol=1.0e-12, rtol=1.0e-12)
        self.assertEqual(x[15], x_clone[15], atol=1.0e-12, rtol=1.0e-12)

        x.resize_(0)
        self.assertEqual(x.size(), 0, atol=1.0e-12, rtol=1.0e-12)
        x = torch.UntypedStorage([1, 2, 3, 4], device="cpu")
        x.resize_(0)
        self.assertEqual(x.size(), 0, atol=1.0e-12, rtol=1.0e-12)
        torch.set_default_dtype(default_dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_storage_fill(self):
        x = torch.rand((4, 5), dtype=torch.float).to("mlu")
        y = x.clone()
        x_storage = x.storage()
        x_storage.fill_(2)
        y.fill_(2)
        self.assertTensorsEqual(x.cpu(), y.cpu(), 0.0, use_MSE=True)
        x = torch.rand((4, 5), dtype=torch.float).to("mlu")
        y = x.cpu().clone()
        x_storage = x.untyped_storage()
        x_storage.fill_(3)
        y.untyped_storage().fill_(3)
        self.assertTensorsEqual(x.cpu(), y.cpu(), 0.0, use_MSE=True)

        x = torch.UntypedStorage([1, 2, 3, 4, 5], device="mlu")
        y = x.fill_(0)
        self.assertEqual(y.tolist(), [0, 0, 0, 0, 0])

    # @unittest.skip("not test")
    @testinfo()
    def test_storage_weak_ref(self):
        x = torch.rand((4, 5), dtype=torch.float).to("mlu")
        x_storage = x.storage()
        cdata = x_storage._weak_ref()
        torch.UntypedStorage._free_weak_ref(cdata)

    # @unittest.skip("not test")
    @testinfo()
    def test_storage_new_with_weak_ptr(self):
        for device in ["cpu", "mlu"]:
            tmp = torch.randn(2, 2, dtype=torch.float32).storage().untyped().tolist()
            a = torch.UntypedStorage(tmp, device=device)
            r = a._weak_ref()
            self.assertEqual(r, a._cdata)
            x = torch.UntypedStorage._new_with_weak_ptr(r)
            self.assertEqual(x.tolist(), a.tolist())
            self.assertEqual(x._cdata, a._cdata)
            self.assertEqual(x.device.type, a.device.type)

    # @unittest.skip("not test")
    @testinfo()
    def test_storage_expired(self):
        x = torch.ones((4, 5), dtype=torch.int).to("mlu")
        x_storage = x.storage()
        cdata = x_storage._weak_ref()
        # static methods
        expired = torch.UntypedStorage._expired(cdata)
        self.assertEqual(expired, False)

    # @unittest.skip("not test")
    @testinfo()
    def test_storage_incdecref(self):
        x = torch.rand((4, 5), dtype=torch.float).to("mlu")
        x_storage = x.storage()
        x_storage._shared_incref()
        x_storage._shared_decref()

        y = torch.UntypedStorage([1, 2, 3, 4], device="cpu")
        y._shared_incref()
        y._shared_decref()

    # @unittest.skip("not test")
    @testinfo()
    def test_storage_new_share(self):
        sharing_strategy = ["file_system", "file_descriptor"]
        size = 100
        for strategy in sharing_strategy:
            torch.multiprocessing.set_sharing_strategy(strategy)
            x = torch.UntypedStorage._new_shared(size, device="cpu")
            shared = x.is_shared()
            self.assertTrue(shared)
            x = torch.randn(3, 3, device="mlu").storage()
            x._new_shared(size, device="mlu")
            shared = x.is_shared()
            self.assertTrue(shared)
            x = torch.randn(3, 3, device="mlu").storage()
            x._new_shared(size, device="cpu")
            shared = x.is_shared()
            self.assertTrue(shared)

    # @unittest.skip("not test")
    @testinfo()
    def test_storage_new_shared_fd_cpu(self):
        a = torch.UntypedStorage(
            [189, 0, 89, 63, 227, 64, 194, 61, 7, 234, 129, 63, 29, 147, 120, 63],
            device="cpu",
        )
        fd, size = a._share_fd_cpu_()
        x = torch.UntypedStorage._new_shared_fd_cpu(fd, size)
        self.assertEqual(x.tolist(), a.tolist())
        self.assertFalse(x._cdata == a._cdata)
        self.assertTrue(x.is_shared())

    # @unittest.skip("not test")
    @testinfo()
    def test_storage_get_shared_fd_cpu(self):
        a = torch.UntypedStorage(
            [189, 0, 89, 63, 227, 64, 194, 61, 7, 234, 129, 63, 29, 147, 120, 63],
            device="cpu",
        )
        a.share_memory_()
        fd = a._get_shared_fd()
        size = a.size()
        x = torch.UntypedStorage._new_shared_fd_cpu(fd, size)
        self.assertEqual(x.tolist(), a.tolist())
        self.assertFalse(x._cdata == a._cdata)
        self.assertTrue(x.is_shared())

    # @unittest.skip("not test")
    @testinfo()
    def test_storage_new_shared_filename_cpu(self):
        a_list = torch.randn(2, 2).storage().untyped().tolist()
        a = torch.UntypedStorage(a_list, device="cpu")
        manager, handle, size = a._share_filename_cpu_()
        b = torch.UntypedStorage._new_shared_filename_cpu(manager, handle, size)
        self.assertTrue(b.is_shared())
        self.assertFalse(a._cdata == b._cdata)
        self.assertEqual(a.tolist(), b.tolist())

    # @unittest.skip("not test")
    @testinfo()
    def test_storage_share_memory(self):
        sharing_strategy = ["file_system", "file_descriptor"]
        for strategy in sharing_strategy:
            torch.multiprocessing.set_sharing_strategy(strategy)
            x = torch.UntypedStorage([1, 2, 3, 4], device="cpu")
            x.share_memory_()
            shared = x.is_shared()
            self.assertTrue(shared)

    # @unittest.skip("not test")
    @testinfo()
    def test_storage_is_shared(self):
        x = torch.rand((4, 5), dtype=torch.float).to("mlu")
        x_storage = x.storage()
        shared = x_storage.is_shared()
        self.assertEqual(shared, True)

    # @unittest.skip("not test")
    @testinfo()
    def test_constructor_dtypes(self):
        if torch.mlu.is_available():
            default_type = torch.Tensor().type()
            torch.set_default_tensor_type(torch.FloatTensor)
            self.assertIs(torch.float32, torch.get_default_dtype())
            self.assertIs(torch.float32, torch.FloatTensor.dtype)
            self.assertIs(torch.FloatStorage, torch.Storage)

            # set_default_dtype() depends on the current default tensor type
            # if default tensor type == torch.FloatTensor, torch.Storage would be torch.DoubleStorage after torch.set_default_dtype(torch.float64),
            # indicating that pytorch would adjust the torch.Storage according to the dtype and backend,
            # if instead default tensor type == torch.FloatTensor, torch.Storage would be torch.DoubleStorage
            # refer to the function py_set_default_dtype() in torch/csrc/tensor/python_tensor.cpp to verify such behaviour
            default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float64)
            self.assertIs(torch.float64, torch.get_default_dtype())
            self.assertIs(torch.DoubleStorage, torch.Storage)
            torch.set_default_dtype(default_dtype)
            torch.set_default_tensor_type(default_type)

    # @unittest.skip("not test")
    @testinfo()
    def test_tensor_set_errors(self):
        f_cpu = torch.randn((2, 3), dtype=torch.float32)
        d_cpu = torch.randn((2, 3), dtype=torch.float64)

        # change dtype
        self.assertRaises(RuntimeError, lambda: f_cpu.set_(d_cpu.storage()))
        self.assertRaises(
            RuntimeError,
            lambda: f_cpu.set_(d_cpu.storage(), 0, d_cpu.size(), d_cpu.stride()),
        )
        self.assertRaises(RuntimeError, lambda: f_cpu.set_(d_cpu))

        # change device
        if torch.mlu.is_available():
            f_mlu = torch.randn((2, 3), dtype=torch.float32, device="mlu")

            # cpu -> mlu
            self.assertRaises(RuntimeError, lambda: f_cpu.set_(f_mlu.storage()))
            self.assertRaises(
                RuntimeError,
                lambda: f_cpu.set_(f_mlu.storage(), 0, f_mlu.size(), f_mlu.stride()),
            )
            self.assertRaises(RuntimeError, lambda: f_cpu.set_(f_mlu))

            # mlu -> cpu
            self.assertRaises(RuntimeError, lambda: f_mlu.set_(f_cpu.storage()))
            self.assertRaises(
                RuntimeError,
                lambda: f_mlu.set_(f_cpu.storage(), 0, f_cpu.size(), f_cpu.stride()),
            )
            self.assertRaises(RuntimeError, lambda: f_mlu.set_(f_cpu))

    # @unittest.skip("not test")
    @testinfo()
    def test_element_size(self):
        byte = torch.ByteStorage().element_size()
        char = torch.CharStorage().element_size()
        short = torch.ShortStorage().element_size()
        int = torch.IntStorage().element_size()
        long = torch.LongStorage().element_size()
        float = torch.FloatStorage().element_size()
        double = torch.DoubleStorage().element_size()
        bool = torch.BoolStorage().element_size()
        # bfloat16 = torch.BFloat16Storage().element_size()
        complexfloat = torch.ComplexFloatStorage().element_size()
        complexdouble = torch.ComplexDoubleStorage().element_size()

        self.assertEqual(byte, torch.ByteTensor().element_size())
        self.assertEqual(char, torch.CharTensor().element_size())
        self.assertEqual(short, torch.ShortTensor().element_size())
        self.assertEqual(int, torch.IntTensor().element_size())
        self.assertEqual(long, torch.LongTensor().element_size())
        self.assertEqual(float, torch.FloatTensor().element_size())
        self.assertEqual(double, torch.DoubleTensor().element_size())
        self.assertEqual(bool, torch.BoolTensor().element_size())
        # self.assertEqual(bfloat16, torch.mlu.tensor([], dtype=torch.bfloat16).element_size())
        self.assertEqual(
            complexfloat, torch.tensor([], dtype=torch.complex64).element_size()
        )
        self.assertEqual(
            complexdouble, torch.tensor([], dtype=torch.complex128).element_size()
        )

        self.assertGreater(byte, 0)
        self.assertGreater(char, 0)
        self.assertGreater(short, 0)
        self.assertGreater(int, 0)
        self.assertGreater(long, 0)
        self.assertGreater(float, 0)
        self.assertGreater(double, 0)
        self.assertGreater(bool, 0)
        # self.assertGreater(bfloat16, 0)
        self.assertGreater(complexfloat, 0)
        self.assertGreater(complexdouble, 0)

        # These tests are portable, not necessarily strict for your system.
        self.assertEqual(byte, 1)
        self.assertEqual(char, 1)
        self.assertEqual(bool, 1)
        self.assertGreaterEqual(short, 2)
        self.assertGreaterEqual(int, 2)
        self.assertGreaterEqual(int, short)
        self.assertGreaterEqual(long, 4)
        self.assertGreaterEqual(long, int)
        self.assertGreaterEqual(double, float)

        t_cpu = torch.randn(2, 2)
        t_mlu = t_cpu.mlu()
        s_cpu = t_cpu.storage().untyped()
        s_mlu = t_mlu.storage().untyped()
        self.assertEqual(s_cpu.element_size(), s_mlu.element_size())

    # failed because from_buffer is not supported
    # @unittest.skip("not test")
    @testinfo()
    def test_from_buffer(self):
        for order in ["big", "little", "native"]:
            a = bytearray([1, 2, 3, 4])
            untyped_mlu = torch.UntypedStorage.from_buffer(
                a, byte_order=order, dtype=torch.uint8
            )
            untyped_cpu = torch.UntypedStorage.from_buffer(
                a, byte_order=order, dtype=torch.uint8
            )
            self.assertEqual(untyped_mlu.tolist(), untyped_cpu.tolist())
            untyped_mlu = torch.UntypedStorage.from_buffer(
                a, byte_order=order, dtype=torch.short
            )
            untyped_cpu = torch.UntypedStorage.from_buffer(
                a, byte_order=order, dtype=torch.short
            )
            self.assertEqual(untyped_mlu.tolist(), untyped_cpu.tolist())
            f = bytearray([0x40, 0x10, 0x00, 0x00])
            untyped_mlu = torch.UntypedStorage.from_buffer(
                f, byte_order=order, dtype=torch.float32
            )
            untyped_cpu = torch.UntypedStorage.from_buffer(
                f, byte_order=order, dtype=torch.float32
            )
            self.assertEqual(untyped_mlu.tolist(), untyped_cpu.tolist())
            f = bytearray([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x40])
            untyped_mlu = torch.UntypedStorage.from_buffer(
                f, byte_order=order, dtype=torch.bool
            )
            untyped_cpu = torch.UntypedStorage.from_buffer(
                f, byte_order=order, dtype=torch.bool
            )
            self.assertEqual(untyped_mlu.tolist(), untyped_cpu.tolist())
            f = bytearray([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x40])
            untyped_mlu = torch.UntypedStorage.from_buffer(
                f, byte_order=order, dtype=torch.float64
            )
            untyped_cpu = torch.UntypedStorage.from_buffer(
                f, byte_order=order, dtype=torch.float64
            )
            self.assertEqual(untyped_mlu.tolist(), untyped_cpu.tolist())
            f = bytearray([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x40])
            untyped_mlu = torch.UntypedStorage.from_buffer(
                f, byte_order=order, dtype=torch.int32
            )
            untyped_cpu = torch.UntypedStorage.from_buffer(
                f, byte_order=order, dtype=torch.int32
            )
            self.assertEqual(untyped_mlu.tolist(), untyped_cpu.tolist())
            f = bytearray([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x40])
            untyped_mlu = torch.UntypedStorage.from_buffer(
                f, byte_order=order, dtype=torch.int64
            )
            untyped_cpu = torch.UntypedStorage.from_buffer(
                f, byte_order=order, dtype=torch.int64
            )
            self.assertEqual(untyped_mlu.tolist(), untyped_cpu.tolist())
            f = bytearray([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x40])
            untyped_mlu = torch.UntypedStorage.from_buffer(
                f, byte_order=order, dtype=torch.float16
            )
            untyped_cpu = torch.UntypedStorage.from_buffer(
                f, byte_order=order, dtype=torch.float16
            )
            self.assertEqual(untyped_mlu.tolist(), untyped_cpu.tolist())
            f = bytearray([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x40])
            untyped_mlu = torch.UntypedStorage.from_buffer(
                f, byte_order=order, dtype=torch.bfloat16
            )
            untyped_cpu = torch.UntypedStorage.from_buffer(
                f, byte_order=order, dtype=torch.bfloat16
            )
            self.assertEqual(untyped_mlu.tolist(), untyped_cpu.tolist())
            f = bytearray([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x40])
            untyped_mlu = torch.UntypedStorage.from_buffer(
                f, byte_order=order, dtype=torch.complex64
            )
            untyped_cpu = torch.UntypedStorage.from_buffer(
                f, byte_order=order, dtype=torch.complex64
            )
            self.assertEqual(untyped_mlu.tolist(), untyped_cpu.tolist())
            # TODO(PYTORCH-10290): open this later.
            # f = bytearray([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x40,
            #               0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x40])
            # untyped_mlu = torch.UntypedStorage.from_buffer(f, byte_order=order, dtype=torch.complex128)
            # untyped_cpu = torch.UntypedStorage.from_buffer(f, byte_order=order, dtype=torch.complex128)
            # self.assertEqual(untyped_mlu.tolist(), untyped_cpu.tolist())

    # @unittest.skip("not test")
    @testinfo()
    def test_from_file(self):
        sizes = [0, 10000]
        for size in sizes:
            with tempfile.NamedTemporaryFile() as f:
                s1 = torch.UntypedStorage.from_file(f.name, True, size)
                t1 = torch.DoubleTensor(s1)
                t1 = t1.copy_(torch.randn(t1.size()))

                # check mapping
                s2 = torch.UntypedStorage.from_file(f.name, True, size)
                t2 = torch.DoubleTensor(s2)
                self.assertEqual(t1, t2, atol=0, rtol=0)

                # check changes to t1 from t2
                t1.fill_(1)
                self.assertEqual(t1, t2, atol=0, rtol=0)

                # check changes to t2 from t1
                t2.fill_(2)
                self.assertEqual(t1, t2, atol=0, rtol=0)

    # @unittest.skip("not test")
    @testinfo()
    def test_filename(self):
        sizes = [0, 10000]
        for size in sizes:
            with tempfile.NamedTemporaryFile() as f:
                s1 = torch.UntypedStorage.from_file(f.name, True, size)
                self.assertIsNotNone(s1.filename)
        a = torch.randn(2, 2).mlu().storage().untyped()
        self.assertIsNone(a.filename)

    # @unittest.skip("not test")
    @testinfo()
    def test_contiguous(self):
        x = torch.randn(1, 16, 5, 5, device="mlu")
        self.assertTrue(x.is_contiguous())
        stride = list(x.stride())
        stride[0] = 20
        # change the stride in dimension 0. the tensor is still contiguous because size[0] is 1
        x.set_(x.storage(), 0, x.size(), stride)
        self.assertTrue(x.is_contiguous())

    # @unittest.skip("not test")
    @testinfo()
    def test_type_conversions(self):
        default_type = torch.Tensor().type()
        torch.set_default_tensor_type(torch.FloatTensor)
        x = torch.randn(5, 5)
        self.assertIsInstance(x.float(), torch.FloatTensor)
        self.assertIsInstance(x.mlu().double(), torch.mlu.DoubleTensor)
        self.assertIsInstance(x.mlu().float(), torch.mlu.FloatTensor)
        self.assertIsInstance(x.mlu().float().cpu(), torch.FloatTensor)
        self.assertIsInstance(x.mlu().float().cpu().int(), torch.IntTensor)

        y = x.storage()
        self.assertIsInstance(y.float(), torch.FloatStorage)
        self.assertIsInstance(y.mlu().double(), torch.mlu.DoubleStorage)
        self.assertIsInstance(y.mlu().float(), torch.mlu.FloatStorage)
        self.assertIsInstance(y.mlu().float().cpu(), torch.FloatStorage)
        self.assertIsInstance(y.mlu().float().cpu().int(), torch.IntStorage)
        torch.set_default_tensor_type(default_type)

    # @unittest.skip("not test")
    @testinfo()
    def test_pynew(self):
        floatStorage = torch.FloatStorage()
        self.assertEqual(floatStorage.size(), 0)
        self.assertEqual(floatStorage.type(), "torch.FloatStorage")
        self.assertIs(floatStorage.dtype, torch.float32)
        self.assertIs(floatStorage.element_size(), torch.FloatTensor().element_size())

        floatStorage = torch.FloatStorage(8)
        self.assertEqual(floatStorage.size(), 8)
        self.assertEqual(floatStorage.type(), "torch.FloatStorage")
        self.assertIs(floatStorage.dtype, torch.float32)
        self.assertIs(floatStorage.element_size(), torch.FloatTensor().element_size())

        floatStorage = torch.FloatStorage([1, 2, 3, 4])
        self.assertEqual(floatStorage.size(), 4)
        self.assertEqual(floatStorage.type(), "torch.FloatStorage")
        self.assertIs(floatStorage.dtype, torch.float32)
        self.assertIs(floatStorage.element_size(), torch.FloatTensor().element_size())

        complexfloatStorage = torch.ComplexFloatStorage([1 + 1j, 2 - 1j, 3, 4])
        self.assertEqual(complexfloatStorage.size(), 4)
        self.assertEqual(complexfloatStorage.type(), "torch.ComplexFloatStorage")
        self.assertIs(complexfloatStorage.dtype, torch.complex64)
        self.assertIs(
            complexfloatStorage.element_size(),
            torch.tensor([], dtype=torch.complex64).element_size(),
        )

        complexdoubleStorage = torch.ComplexDoubleStorage([1 + 1j, 2 - 1j, 3, 4])
        self.assertEqual(complexdoubleStorage.size(), 4)
        self.assertEqual(complexdoubleStorage.type(), "torch.ComplexDoubleStorage")
        self.assertIs(complexdoubleStorage.dtype, torch.complex128)
        self.assertIs(
            complexdoubleStorage.element_size(),
            torch.tensor([], dtype=torch.complex128).element_size(),
        )

        mlu_untyped_storage = torch.UntypedStorage([1, 2, 3, 4, 5], device="mlu")
        mlu_new_storage = mlu_untyped_storage.new()
        self.assertTrue(mlu_new_storage.device.type == "mlu")

        cpu_untyped_storage = torch.UntypedStorage([1, 2, 3, 4, 5], device="mlu")
        cpu_new_storage = cpu_untyped_storage.new()
        self.assertEqual(cpu_new_storage.size(), mlu_new_storage.size())

    # @unittest.skip("not test")
    @testinfo()
    def test_copy_(self):
        a = torch.randn(2, 2, device="mlu")
        b = torch.zeros(1, 4, 1, device="mlu")
        b.storage().copy_(a.storage())
        self.assertEqual(a, b.reshape([2, 2]), atol=1.0e-12, rtol=1.0e-12)

        a = torch.randn(2, 2, device="mlu")
        b = torch.zeros(2, 2, device="mlu")

        b.storage().copy_(a.storage())
        self.assertIsInstance(a.char().storage(), torch.storage.TypedStorage)
        self.assertIsInstance(b.char().storage(), torch.storage.TypedStorage)
        self.assertEqual(a.char(), b.char(), atol=1.0e-12, rtol=1.0e-12)

        b.storage().copy_(a.storage())
        self.assertIsInstance(a.byte().storage(), torch.storage.TypedStorage)
        self.assertIsInstance(b.byte().storage(), torch.storage.TypedStorage)
        self.assertEqual(a.byte(), b.byte(), atol=1.0e-12, rtol=1.0e-12)

        b.storage().copy_(a.storage())
        self.assertIsInstance(a.double().storage(), torch.storage.TypedStorage)
        self.assertIsInstance(b.double().storage(), torch.storage.TypedStorage)
        self.assertEqual(a.double(), b.double(), atol=1.0e-12, rtol=1.0e-12)

        b.storage().copy_(a.storage())
        self.assertIsInstance(a.long().storage(), torch.storage.TypedStorage)
        self.assertIsInstance(b.long().storage(), torch.storage.TypedStorage)
        self.assertEqual(a.long(), b.long(), atol=1.0e-12, rtol=1.0e-12)

        b.storage().copy_(a.storage())
        self.assertIsInstance(a.short().storage(), torch.storage.TypedStorage)
        self.assertIsInstance(b.short().storage(), torch.storage.TypedStorage)
        self.assertEqual(a.short(), b.short(), atol=1.0e-12, rtol=1.0e-12)

        b.storage().copy_(a.storage())
        self.assertIsInstance(a.half().storage(), torch.storage.TypedStorage)
        self.assertIsInstance(b.half().storage(), torch.storage.TypedStorage)
        self.assertEqual(a.half(), b.half(), atol=1.0e-12, rtol=1.0e-12)

        b.storage().copy_(a.storage())
        self.assertIsInstance(a.bool().storage(), torch.storage.TypedStorage)
        self.assertIsInstance(b.bool().storage(), torch.storage.TypedStorage)
        self.assertEqual(a.bool(), b.bool(), atol=1.0e-12, rtol=1.0e-12)

    # @unittest.skip("not test")
    @testinfo()
    def test_data_ptr(self):
        tensor_a = torch.randn(4, 5, device="mlu")
        storage_a = tensor_a.storage()
        self.assertEqual(storage_a.data_ptr(), tensor_a.data_ptr())

        tensor_b = torch.Tensor().mlu().new(storage_a)
        self.assertEqual(tensor_b.data_ptr(), tensor_a.data_ptr())
        self.assertEqual(tensor_b.data_ptr(), storage_a.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test__write_and_new_with_file(self):
        # test float
        tensor_a = torch.randn((4, 5), dtype=torch.float, device="mlu")
        tensor_c = torch.tensor(tensor_a.storage())
        self.assertEqual(tensor_a, tensor_c.reshape(4, 5), atol=1.0e-12, rtol=1.0e-12)
        with open("_write_file.pt", "wb") as writer:
            tensor_a.untyped_storage()._write_file(
                writer, True, True, torch._utils._element_size(tensor_a.dtype)
            )
        with open("_write_file.pt", "rb") as reader:
            storage_b = torch.UntypedStorage._new_with_file(
                reader, torch._utils._element_size(tensor_a.dtype)
            )
        tensor_b = torch.FloatTensor(storage_b)
        self.assertEqual(tensor_a, tensor_b.reshape(4, 5), atol=1.0e-12, rtol=1.0e-12)

        untyped_a = torch.UntypedStorage([1, 2, 3, 4], device="cpu")
        with open("_write_file_1.pt", "wb") as writer:
            untyped_a._write_file(
                writer, False, True, torch._utils._element_size(torch.float16)
            )
        with open("_write_file_1.pt", "rb") as reader:
            untyped_b = torch.UntypedStorage._new_with_file(
                reader, torch._utils._element_size(torch.float16)
            )
        self.assertEqual(
            untyped_a.tolist(), untyped_b.tolist(), atol=1.0e-12, rtol=1.0e-12
        )

        untyped_a = torch.UntypedStorage([1, 2, 3, 4], device="cpu")
        with open("_write_file_1.pt", "wb") as writer:
            untyped_a._write_file(
                writer, True, True, torch._utils._element_size(torch.float16)
            )
        with open("_write_file_1.pt", "rb") as reader:
            untyped_b = torch.UntypedStorage._new_with_file(
                reader, torch._utils._element_size(torch.float16)
            )
        self.assertEqual(
            untyped_a.tolist(), untyped_b.tolist(), atol=1.0e-12, rtol=1.0e-12
        )

        # test double
        tensor_a = torch.randn((4, 5), dtype=torch.double, device="mlu")
        with open("_write_file.pt", "wb") as writer:
            tensor_a.untyped_storage()._write_file(
                writer, True, True, torch._utils._element_size(tensor_a.dtype)
            )
        with open("_write_file.pt", "rb") as reader:
            storage_b = torch.UntypedStorage._new_with_file(
                reader, torch._utils._element_size(tensor_a.dtype)
            )
        tensor_b = torch.mlu.DoubleTensor(storage_b.mlu())
        self.assertEqual(storage_b.type(), "torch.storage.UntypedStorage")
        self.assertEqual(
            tensor_a, tensor_b.double().reshape(4, 5), atol=1.0e-12, rtol=1.0e-12
        )
        # we have to call tensor_b.double() to convert the tensor to a double tensor first
        # because when the default tensor type is torch.FloatTensor, the tensor constructed by tensor_b = torch.tensor(storage_b) is a float tensor,
        # even though the storage is double

        # test complex_float
        tensor_a = torch.randn((4, 5), dtype=torch.complex64, device="mlu")
        tensor_b = torch.empty_like(tensor_a, device="mlu")
        with open("_write_file.pt", "wb") as writer:
            tensor_a.untyped_storage()._write_file(
                writer, True, True, torch._utils._element_size(tensor_a.dtype)
            )
        with open("_write_file.pt", "rb") as reader:
            storage_b = torch.UntypedStorage._new_with_file(
                reader, torch._utils._element_size(tensor_a.dtype)
            )
        typedstorage_b = torch.ComplexFloatStorage(wrap_storage=storage_b)
        tensor_b.set_(typedstorage_b.mlu())
        self.assertEqual(
            tensor_a.cpu(), tensor_b.reshape(4, 5).cpu(), atol=1.0e-12, rtol=1.0e-12
        )
        self.assertEqual(
            tensor_a.storage().cpu(), typedstorage_b.cpu(), atol=1.0e-12, rtol=1.0e-12
        )

        tensor_a = torch.randn((4, 5), dtype=torch.complex128, device="mlu")
        with open("_write_file.pt", "wb") as writer:
            tensor_a.untyped_storage()._write_file(
                writer, True, True, torch._utils._element_size(tensor_a.dtype)
            )
        with open("_write_file.pt", "rb") as reader:
            storage_b = torch.UntypedStorage._new_with_file(
                reader, torch._utils._element_size(tensor_a.dtype)
            )
        typedstorage_b = torch.mlu.ComplexDoubleStorage(wrap_storage=storage_b.mlu())
        self.assertEqual(
            tensor_a.storage().cpu(), typedstorage_b.cpu(), atol=1.0e-12, rtol=1.0e-12
        )
        self.assertEqual(typedstorage_b.type(), "torch_mlu.mlu.ComplexDoubleStorage")
        self.assertIs(typedstorage_b.dtype, torch.complex128)
        os.remove("_write_file.pt")

    # @unittest.skip("not test")
    @testinfo()
    def test__set_from_file(self):
        # write consecutively two tensors into the file and then load them
        tensor_a = torch.randn((4, 5), dtype=torch.float, device="mlu")
        tensor_b = torch.randn((3, 3), dtype=torch.double, device="mlu")
        tensor_c = torch.randn((5, 6), dtype=torch.complex128, device="mlu")
        with tempfile.NamedTemporaryFile() as f:
            tensor_a.untyped_storage()._write_file(
                f, True, True, torch._utils._element_size(tensor_a.dtype)
            )
            tensor_b.untyped_storage()._write_file(
                f, True, True, torch._utils._element_size(tensor_b.dtype)
            )
            tensor_c.untyped_storage()._write_file(
                f, True, True, torch._utils._element_size(tensor_c.dtype)
            )
            tensor_a_shadow = torch.empty((4, 5), dtype=torch.float, device="mlu")
            tensor_b_shadow = torch.empty((3, 3), dtype=torch.double, device="mlu")
            tensor_c_shadow = torch.empty((5, 6), dtype=torch.complex128, device="mlu")

            tensor_a_shadow.untyped_storage()._set_from_file(
                f, 0, True, torch._utils._element_size(tensor_a.dtype)
            )
            tensor_b_shadow.untyped_storage()._set_from_file(
                f, None, True, torch._utils._element_size(tensor_b.dtype)
            )
            tensor_c_shadow.untyped_storage()._set_from_file(
                f, None, True, torch._utils._element_size(tensor_c.dtype)
            )

            self.assertEqual(tensor_a.cpu(), tensor_a_shadow.cpu())
            self.assertEqual(tensor_b.cpu(), tensor_b_shadow.cpu())
            self.assertEqual(tensor_c.cpu(), tensor_c_shadow.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_new_with_file1(self):
        x = torch.tensor([1.0, 2.0], dtype=torch.double)
        y = torch.tensor([3.0, 4.0], dtype=torch.float)
        z = torch.randn((2, 2), dtype=torch.complex128)
        w = torch.randn((3, 3), dtype=torch.float)
        x_copy = torch.empty_like(x, device="mlu")
        y_copy = torch.empty_like(y, device="mlu")
        z_copy = torch.empty_like(z, device="mlu")
        w_copy = torch.empty_like(w, device="mlu")
        with tempfile.NamedTemporaryFile() as f:
            x.untyped_storage()._write_file(
                f, True, True, torch._utils._element_size(x.dtype)
            )
            y.untyped_storage()._write_file(
                f, True, True, torch._utils._element_size(y.dtype)
            )
            z.untyped_storage()._write_file(
                f, True, True, torch._utils._element_size(z.dtype)
            )
            w.untyped_storage()._write_file(
                f, True, True, torch._utils._element_size(w.dtype)
            )
            f.seek(0)
            x_untypedstorage = torch.UntypedStorage._new_with_file(
                f, torch._utils._element_size(x.dtype)
            )
            y_untypedstorage = torch.UntypedStorage._new_with_file(
                f, torch._utils._element_size(y.dtype)
            )
            z_untypedstorage = torch.UntypedStorage._new_with_file(
                f, torch._utils._element_size(z.dtype)
            )
            w_untypedstorage = torch.UntypedStorage._new_with_file(
                f, torch._utils._element_size(w.dtype)
            )
            x_typedstorage = torch.DoubleStorage(wrap_storage=x_untypedstorage)
            y_typedstorage = torch.FloatStorage(wrap_storage=y_untypedstorage)
            z_typedstorage = torch.ComplexDoubleStorage(wrap_storage=z_untypedstorage)
            w_typedstorage = torch.FloatStorage(wrap_storage=w_untypedstorage)
            x_copy.set_(x_typedstorage.mlu())
            y_copy.set_(y_typedstorage.mlu())
            z_copy.set_(z_typedstorage.mlu())
            w_copy.set_(w_typedstorage.mlu())
            # self.assertEqual(x_copy.cpu(), x)
            self.assertEqual(y_copy.cpu(), y)
            # self.assertEqual(z_copy.reshape(2,2).cpu(), z)
            self.assertEqual(w_copy.reshape(3, 3).cpu(), w)
            # The API currently does not support 64bit data between different devices.

    # @unittest.skip("not test")
    @testinfo()
    def test_new_with_file2(self):
        x = torch.tensor([1.0, 2.0], dtype=torch.double, device="mlu")
        y = torch.tensor([3.0, 4.0], dtype=torch.float, device="mlu")
        z = torch.randn((2, 2), dtype=torch.complex128, device="mlu")
        w = torch.randn((3, 3), dtype=torch.float, device="mlu")
        x_copy = torch.empty_like(x, device="cpu")
        y_copy = torch.empty_like(y, device="cpu")
        z_copy = torch.empty_like(z, device="cpu")
        w_copy = torch.empty_like(w, device="cpu")
        with tempfile.NamedTemporaryFile() as f:
            x.untyped_storage()._write_file(
                f, True, True, torch._utils._element_size(x.dtype)
            )
            y.untyped_storage()._write_file(
                f, True, True, torch._utils._element_size(y.dtype)
            )
            z.untyped_storage()._write_file(
                f, True, True, torch._utils._element_size(z.dtype)
            )
            w.untyped_storage()._write_file(
                f, True, True, torch._utils._element_size(w.dtype)
            )
            f.seek(0)
            x_untypedstorage = torch.UntypedStorage._new_with_file(
                f, torch._utils._element_size(x.dtype)
            )
            y_untypedstorage = torch.UntypedStorage._new_with_file(
                f, torch._utils._element_size(y.dtype)
            )
            z_untypedstorage = torch.UntypedStorage._new_with_file(
                f, torch._utils._element_size(z.dtype)
            )
            w_untypedstorage = torch.UntypedStorage._new_with_file(
                f, torch._utils._element_size(w.dtype)
            )
            x_typedstorage = torch.DoubleStorage(wrap_storage=x_untypedstorage)
            y_typedstorage = torch.FloatStorage(wrap_storage=y_untypedstorage)
            z_typedstorage = torch.ComplexDoubleStorage(wrap_storage=z_untypedstorage)
            w_typedstorage = torch.FloatStorage(wrap_storage=w_untypedstorage)
            x_copy.set_(x_typedstorage)
            y_copy.set_(y_typedstorage)
            z_copy.set_(z_typedstorage)
            w_copy.set_(w_typedstorage)
            # self.assertEqual(x_copy.mlu(), x)
            self.assertEqual(y_copy.mlu(), y)
            # self.assertEqual(z_copy.mlu().reshape(2,2), z)
            self.assertEqual(w_copy.mlu().reshape(3, 3), w)
            # The API currently does not support 64bit data between different devices.

    # @unittest.skip("not test")
    @testinfo()
    def test_save_load_1(self):
        for is_new in [False, True]:
            x = torch.tensor([1, 2], dtype=torch.long)
            y = torch.tensor([3.0, 4.0], dtype=torch.complex128)
            w = torch.tensor([3.0, 4.0], dtype=torch.float)
            q = [x.storage(), y.storage(), w.storage()]
            with tempfile.NamedTemporaryFile() as f:
                torch.save(q, f, _use_new_zipfile_serialization=is_new)
                f.seek(0)
                q_copy_cpu = torch.load(f, map_location="cpu")
            with tempfile.NamedTemporaryFile() as f:
                torch.save(q, f, _use_new_zipfile_serialization=is_new)
                f.seek(0)
                q_copy_mlu = torch.load(f, map_location="mlu")
            self.assertEqual(q_copy_mlu[0].cpu(), q_copy_cpu[0])
            self.assertEqual(q_copy_mlu[1].cpu(), q_copy_cpu[1])
            self.assertEqual(q_copy_mlu[2].cpu(), q_copy_cpu[2])
            self.assertEqual(q_copy_mlu[0].cpu(), q[0])
            self.assertEqual(q_copy_mlu[1].tolist(), q[1].tolist())
            self.assertEqual(q_copy_mlu[2].tolist(), q[2].tolist())

    # @unittest.skip("not test")
    @testinfo()
    def test_save_load_2(self):
        for is_new in [False, True]:
            x = torch.tensor([1, 2], dtype=torch.long, device="mlu")
            y = torch.tensor([3.0, 4.0], dtype=torch.complex128, device="mlu")
            w = torch.tensor([3.0, 4.0], dtype=torch.float, device="mlu")
            q = [x.storage(), y.storage(), w.storage()]
            with tempfile.NamedTemporaryFile() as f:
                torch.save(q, f, _use_new_zipfile_serialization=is_new)
                f.seek(0)
                q_copy_cpu = torch.load(f, map_location="cpu")
            with tempfile.NamedTemporaryFile() as f:
                torch.save(q, f, _use_new_zipfile_serialization=is_new)
                f.seek(0)
                q_copy_mlu = torch.load(f, map_location="mlu")
            self.assertEqual(q_copy_mlu[0].cpu(), q_copy_cpu[0])
            self.assertEqual(q_copy_mlu[1].cpu(), q_copy_cpu[1])
            self.assertEqual(q_copy_mlu[2].cpu(), q_copy_cpu[2])
            self.assertEqual(q_copy_cpu[0].tolist(), q[0].tolist())
            self.assertEqual(q_copy_cpu[1].tolist(), q[1].tolist())
            self.assertEqual(q_copy_cpu[2].tolist(), q[2].tolist())

    # @unittest.skip("not test")
    @testinfo()
    def test_storage_get_set(self):
        # test 64bit get set ability
        a = torch.randn(3, 3, dtype=torch.float, device="mlu")
        a_s = a.storage()
        self.assertEqual(a_s[0], a[0][0], atol=1.0e-12, rtol=1.0e-12)
        self.assertEqual(a_s[8], a[2][2], atol=1.0e-12, rtol=1.0e-12)
        a_s[4] = 6.0
        self.assertEqual(a[1][1], 6.0, atol=1.0e-12, rtol=1.0e-12)

        a = torch.randn(3, 3, dtype=torch.double, device="mlu")
        a_s = a.storage()
        self.assertEqual(a_s[0], a[0][0], atol=1.0e-12, rtol=1.0e-12)
        self.assertEqual(a_s[8], a[2][2], atol=1.0e-12, rtol=1.0e-12)
        a_s[4] = 6.0
        self.assertEqual(a[1][1], 6.0, atol=1.0e-12, rtol=1.0e-12)

        # TODO(PYTORCH-9217): fill is not implemented for complex types on MLU,
        # uncomment the following the following tests after complex types are supported
        # a = torch.randn(3, 3, dtype=torch.complex64, device="mlu")
        # a_s = a.storage()
        # self.assertEqual(a_s[0], a[0][0], atol=1.e-12, rtol=1.e-12)
        # self.assertEqual(a_s[8], a[2][2], atol=1.e-12, rtol=1.e-12)
        # a_s[4] = 5.0+3j
        # self.assertEqual(a[1][1], 5.0+3j, atol=1.e-12, rtol=1.e-12)

        # a = torch.randn(3, 3, dtype=torch.complex128, device="mlu")
        # a_s = a.storage()
        # self.assertEqual(a_s[0], a[0][0], atol=1.e-12, rtol=1.e-12)
        # self.assertEqual(a_s[8], a[2][2], atol=1.e-12, rtol=1.e-12)
        # a_s[4] = 5.0+3j
        # self.assertEqual(a[1][1], 5.0+3j, atol=1.e-12, rtol=1.e-12)

    # @unittest.skip("not test")
    @testinfo()
    def test_storage_copy_(self):
        # test 64bit copy ability
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

        a = torch.randn(2, 2, device="cpu")
        b = torch.zeros(2, 2, device="mlu")
        b.storage().copy_(a.storage())
        self.assertTensorsEqual(a, b.cpu(), 1.0e-12, use_MSE=True)

        a = torch.zeros(2, 2, device="cpu")
        b = torch.randn(2, 2, device="mlu")
        a.storage().copy_(b.storage())
        self.assertTensorsEqual(a, b.cpu(), 1.0e-12, use_MSE=True)
        torch.set_default_dtype(default_dtype)

        # TODO(PYTORCH-9217): fill is not implemented for complex types on MLU,
        # uncomment the following the following tests after complex types are supported
        # a = torch.randn(2, 2, device="cpu", dtype=torch.complex128)
        # b = torch.zeros(2, 2, device="mlu", dtype=torch.complex128)
        # b.storage().copy_(a.storage())
        # self.assertTensorsEqual(a.cfloat(), b.cpu().cfloat(), 1.e-12)

        # a = torch.zeros(2, 2, device="cpu", dtype=torch.complex128)
        # b = torch.randn(2, 2, device="mlu", dtype=torch.complex128)
        # a.storage().copy_(b.storage())
        # self.assertTensorsEqual(a, b.cpu(), 1.e-12)
        # torch.set_default_dtype(default_dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_untyped_init(self):
        a = torch.UntypedStorage([1, 2, 3]).device.type
        self.assertTrue(a == "cpu")
        b = torch.UntypedStorage([1, 2, 3], device="mlu").device.type
        self.assertTrue(b == "mlu")

    # @unittest.skip("not test")
    @testinfo()
    def test_untyped_get_set(self):
        a = torch.TypedStorage([1, 2, 3, 4], device="mlu").untyped()
        b = torch.UntypedStorage(a.tolist(), device="mlu")

        self.assertEqual(a[0], b[0], atol=1.0e-12, rtol=1.0e-12)
        self.assertEqual(a[1], b[1], atol=1.0e-12, rtol=1.0e-12)
        self.assertEqual(a[0:2].tolist(), b[0:2].tolist(), atol=1.0e-12, rtol=1.0e-12)
        self.assertEqual(a[2:-1].tolist(), b[2:-1].tolist(), atol=1.0e-12, rtol=1.0e-12)
        b[4] = 6
        self.assertEqual(b[4], 6, atol=1.0e-12, rtol=1.0e-12)

    # @unittest.skip("not test")
    @testinfo()
    def test_storage_exception(self):
        ref_msg = r"torch.UntypedStorage\(\): Storage device not recognized: cuda"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.UntypedStorage([1, 2, 3], device="cuda")

        ref_msg = r"Expected a sequence type, but got int"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.UntypedStorage(sequence=123)

        ref_msg = r"offset must be non-negative and no greater than buffer length \(-1\) , but got 4"
        with self.assertRaisesRegex(ValueError, ref_msg):
            a = bytearray([1, 2, 3, 4])
            untyped_mlu = torch.UntypedStorage.from_buffer(
                a, byte_order="native", offset=-1, dtype=torch.uint8
            )

        ref_msg = (
            r"invalid byte_order 'error' \(expected 'big', 'little', or 'native'\)"
        )
        with self.assertRaisesRegex(ValueError, ref_msg):
            a = bytearray([1, 2, 3, 4])
            untyped_mlu = torch.UntypedStorage.from_buffer(
                a, byte_order="error", dtype=torch.float32
            )

        ref_msg = r"buffer size \(4\) must be a multiple of element size \(4\)"
        with self.assertRaisesRegex(ValueError, ref_msg):
            a = bytearray([1, 2, 3, 4])
            untyped_mlu = torch.UntypedStorage.from_buffer(
                a, byte_order="native", count=-1, offset=3, dtype=torch.float32
            )

        ref_msg = (
            r"buffer has only 1 elements after offset 3, but specified a size of 5"
        )
        with self.assertRaisesRegex(ValueError, ref_msg):
            a = bytearray([1, 2, 3, 4])
            untyped_mlu = torch.UntypedStorage.from_buffer(
                a, byte_order="native", count=5, offset=3, dtype=torch.float32
            )

        ref_msg = r"couldn't retrieve a shared file descriptor"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            a = torch.UntypedStorage(
                [189, 0, 89, 63, 227, 64, 194, 61, 7, 234, 129, 63, 29, 147, 120, 63],
                device="cpu",
            )
            fd = a._get_shared_fd()

    # @unittest.skip("not test")
    @testinfo()
    def test_64_bit_storage_raise_warn(self):
        a = torch.tensor([3, 4], dtype=torch.double)
        a_storage = a.storage()
        warning_msg = "64-bit data were casted to 32-bit data on MLU memory, Although dtype of TypedStorage is 64-bit, real data is 32-bit, which is stored on MLU memory."
        with self.assertWarnsRegex(UserWarning, warning_msg):
            b = a_storage.mlu().untyped()
        with tempfile.NamedTemporaryFile() as f:
            torch.save(a, f)
            f.seek(0)
            self.assertNotWarnRegex(lambda: torch.load(f), warning_msg)
            with PackageExporter(f.name) as he:
                he.save_pickle("obj", "obj.pkl", a)
            hi = PackageImporter(f.name)
            self.assertNotWarnRegex(
                lambda: hi.load_pickle("obj", "obj.pkl"), warning_msg
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_package_save_and_load_64bit(self):
        a = torch.tensor([2, 54], device="mlu")
        f = tempfile.NamedTemporaryFile()
        with PackageExporter(f.name) as he:
            he.save_pickle("obj", "obj.pkl", a)
        hi = PackageImporter(f.name)
        b = hi.load_pickle("obj", "obj.pkl", map_location="cpu")
        self.assertEqual(a.cpu(), b)

    # @unittest.skip("not test")
    @testinfo()
    def test_pyobject_property(self):
        tensor_mlu = torch.randn(2, 2).mlu()
        untyped_a = tensor_mlu.storage().untyped()
        new_property = "mlu with new property."
        untyped_a.__dict__["test_property"] = new_property
        del untyped_a
        untyped_b = tensor_mlu.storage().untyped()
        self.assertIsNotNone(untyped_b.test_property)
        self.assertEqual(untyped_b.test_property, new_property)

    # @unittest.skip("not test")
    @testinfo()
    def test_64bit_typed_copy_to_untyped_storage(self):
        a = torch.randn(3, 4, dtype=torch.double)
        b = torch.randn(3, 4, dtype=torch.double, device="mlu")
        a.storage()._untyped_storage.copy_(b.storage())
        self.assertEqual(a, b.cpu())


if __name__ == "__main__":
    unittest.main()
