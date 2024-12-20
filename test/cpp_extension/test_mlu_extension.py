import subprocess
import shlex
import sys
import os
import logging
import glob
import unittest
import torch
import torch_mlu

from torch_mlu.utils.cpp_extension import load_inline

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import shell, gen_err_message, TestCase

logging.basicConfig(level=logging.DEBUG)


class TestMluExtension(TestCase):
    def get_executable_command(self):
        executable = [sys.executable]
        return executable

    def get_test_files(self, path):
        pyfiles = [
            filename
            for filename in glob.glob("{}/**/test*.py".format(path), recursive=True)
        ]
        return pyfiles

    def build_extension(self, extension_dir):
        # from test file to mlu_extension's directory
        total_error_info = []
        os.chdir(extension_dir)
        command = "python setup.py install"
        args = shlex.split(command)
        return_code = shell(args, extension_dir)
        gen_err_message(return_code, "Mlu_extension", total_error_info)

        print(
            "*********** MLUExtension Mlu_custom_ext Build : Error Message Summaries **************"
        )
        for err_message in total_error_info:
            logging.error("\033[31;1m {}\033[0m .".format(err_message))
        print(
            "*******************************************************************************"
        )

        if total_error_info:
            raise RuntimeError("MLUExtension Mlu_custom_ex Build Failed")

    def run_test(self, executable, test_files, test_directory):
        total_error_info = []
        commands = (executable + [argv] for argv in test_files)
        for command in commands:
            return_code = shell(command, test_directory)
            gen_err_message(return_code, command[-1], total_error_info)

        # Print total error message
        print(
            "*********** MLUExtension test_sigmoid : Error Message Summaries **************"
        )
        for err_message in total_error_info:
            logging.error("\033[31;1m {}\033[0m .".format(err_message))
        print(
            "*******************************************************************************"
        )

        if total_error_info:
            raise RuntimeError("MLUExtension test_sigmoid case Failed")

    def test_mlu_extension(self):
        if not os.getenv("MLU_VISIBLE_DEVICES"):
            os.environ["MLU_VISIBLE_DEVICES"] = "0"
        executable_ = self.get_executable_command()
        base_dir = os.path.join(cur_dir, "../..", "examples", "mlu_extension")
        self.build_extension(base_dir)
        pyfiles_ = self.get_test_files(base_dir)
        self.run_test(executable_, pyfiles_, cur_dir)

    def test_load_inline(self):
        source = """
        at::Tensor sin_add(at::Tensor x, at::Tensor y) {
          return x.sin() + y.sin();
        }
        """
        module = load_inline(
            name="inline_extension", cpp_sources=[source], functions=["sin_add"]
        )
        # Test the sin_add function
        x = torch.tensor([0.1, 0.2, 0.3])
        y = torch.tensor([0.4, 0.5, 0.6])
        expected_result = torch.sin(x) + torch.sin(y)
        result = module.sin_add(x, y)

        # Assert that the result matches the expected result
        self.assertTrue(torch.allclose(result, expected_result))

    def test_inline_jit_compile_extension_bang(self):
        bang_source = """
#define REM_FOR_STACK (128 * 1024)
#if __BANG_ARCH__
#define MAX_NRAM_SIZE (__MLU_NRAM_SIZE__ * 1024 - REM_FOR_STACK)
#else
#define MAX_NRAM_SIZE (384 * 1024)
#endif

__nram__ char nram_buffer[MAX_NRAM_SIZE];
template <typename T>
__mlu_global__ void cos_add_block_kernel(const T* __restrict__ x,
                                 const T* __restrict__ y,
                                 T* __restrict__ output,
                                 const int size) {
    if (__is_mpu()) {
      return;
    }
    int32_t num_per_core = size / taskDim;
    int32_t num_rem = size % taskDim;
    T *addr_x = (T *)x + taskId * num_per_core;
    T *addr_y = (T *)y + taskId * num_per_core;
    T *addr_output = (T *)output + taskId * num_per_core;
    if (num_rem > 0 && taskId == taskDim - 1) {
      // throw the remain data to the last core
      num_per_core = num_per_core + num_rem;
    }
    // nram data is split into 2 parts: x, y
    int32_t nram_split_num = 2;
    int32_t num_deal = MAX_NRAM_SIZE / (nram_split_num * sizeof(T));
    int32_t repeat = num_per_core / num_deal;
    int32_t rem = num_per_core % num_deal;
    // load data from gdrm to nram
    T *nram_x = (T *)nram_buffer;
    T *nram_y = (T *)nram_buffer + num_deal;
    // handle the repeat parts
    for (int i = 0; i < repeat; i++) {
      // loda data to nram
      __memcpy(nram_x, addr_x + i * num_deal, num_deal * sizeof(T), GDRAM2NRAM);
      __memcpy(nram_y, addr_y + i * num_deal, num_deal * sizeof(T), GDRAM2NRAM);
      // compute on nram
      if (std::is_same<T, float>::value) {
        __bang_cos((float *)nram_x, (float *)nram_x, num_deal);
        __bang_cos((float *)nram_y, (float *)nram_y, num_deal);
        __bang_add((float *)nram_y, (float *)nram_x, (float *)nram_y, num_deal);
      }
      // store data to gdram
       __memcpy(addr_output + i * num_deal, nram_y, num_deal * sizeof(T), NRAM2GDRAM);
    }
    // handle the remain parts
    if (rem > 0) {
      __memcpy(nram_x, addr_x + repeat * num_deal, rem * sizeof(T), GDRAM2NRAM);
      __memcpy(nram_y, addr_y + repeat * num_deal, rem * sizeof(T), GDRAM2NRAM);
      // compute on nram
      if (std::is_same<T, float>::value) {
        __bang_cos((float *)nram_x, (float *)nram_x, rem);
        __bang_cos((float *)nram_y, (float *)nram_y, rem);
        __bang_add((float *)nram_y, (float *)nram_x, (float *)nram_y, rem);
      }
      // store data to gdram
      __memcpy(addr_output + repeat * num_deal, nram_y, rem * sizeof(T), NRAM2GDRAM);
    }
}

void cos_add_internal(void *x, void *y, void *output, cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue, int size) {
    cos_add_block_kernel<<<k_dim, k_type, queue>>>((float *)x, (float *)y, (float *)output, size);
}
        """

        # Here, CUDA C++ source only declare the function signature.
        # TODO(COM-14579): cncc compilation restrict the maximam length of string
        # move the bang_cos_add function implementation to .mlu file once supported
        cpp_source = """
#include "aten/utils/tensor_util.h"
#include "framework/core/MLUStream.h"
void cos_add_internal(
    void* x,
    void* y,
    void* output,
    cnrtDim3_t k_dim,
    cnrtFunctionType_t k_type,
    cnrtQueue_t queue,
    int32_t size);

at::Tensor bang_cos_add(const at::Tensor& x, const at::Tensor& y)
{
  auto output = torch::zeros_like(x);
  cnrtDim3_t k_dim;
  k_dim.x = 1;
  k_dim.y = 1;
  k_dim.z = 1;
  int32_t tensor_num = x.numel();
  cnrtFunctionType_t k_type = cnrtFuncTypeBlock;
  auto queue = torch_mlu::getCurMLUStream();
  auto x_impl = torch_mlu::getMluTensorImpl(x);
  auto x_ptr = torch_mlu::mlu_data_ptr(x_impl);
  auto y_impl = torch_mlu::getMluTensorImpl(y);
  auto y_ptr = torch_mlu::mlu_data_ptr(y_impl);
  auto output_impl = torch_mlu::getMluTensorImpl(output);
  auto output_ptr = torch_mlu::mlu_data_ptr(output_impl);
  cos_add_internal(x_ptr, y_ptr, output_ptr, k_dim, k_type, queue, tensor_num);
  cnrtQueueSync(queue);
  return output;
}
        """
        module = torch_mlu.utils.cpp_extension.load_inline(
            name="inline_jit_extension_bang",
            cpp_sources=cpp_source,
            bang_sources=bang_source,
            functions=["bang_cos_add"],
            with_bang=True,
            verbose=True,
        )

        self.assertEqual(module.bang_cos_add.__doc__.split("\n")[2], "bang_cos_add")

        x = torch.randn(4, 4, device="mlu", dtype=torch.float32)
        y = torch.randn(4, 4, device="mlu", dtype=torch.float32)

        z = module.bang_cos_add(x, y)
        self.assertEqual(z, x.cos() + y.cos())


if __name__ == "__main__":
    unittest.main()
