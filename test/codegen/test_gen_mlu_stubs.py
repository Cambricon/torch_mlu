import os
import sys
import tempfile
import unittest
from argparse import Namespace

import expecttest
from torchgen.gen import _GLOBAL_PARSE_NATIVE_YAML_CACHE  # noqa: F401

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../")
from codegen.gen_mlu_stubs import run

gen_backend_stubs_path = os.path.join(cur_dir, "../../codegen/gen_mlu_stubs.py")

options = Namespace()


# gen_mlu_stubs.py is an integration point that is called directly by MLU backend.
# The tests here are to confirm that badly formed inputs result in reasonable error messages.
class TestGenBackendStubs(expecttest.TestCase):
    def setUp(self) -> None:
        global _GLOBAL_PARSE_NATIVE_YAML_CACHE
        _GLOBAL_PARSE_NATIVE_YAML_CACHE.clear()

    def assert_success_from_gen_backend_stubs(self, yaml_str: str) -> None:
        with tempfile.NamedTemporaryFile(mode="w") as fp:
            fp.write(yaml_str)
            fp.flush()
            options.source_yaml = fp.name
            options.dry_run = True
            options.use_bang = True
            options.use_mluop = True
            run(options)

    def get_errors_from_gen_backend_stubs(self, yaml_str: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w") as fp:
            fp.write(yaml_str)
            fp.flush()
            try:
                options.source_yaml = fp.name
                options.dry_run = True
                options.use_bang = True
                options.use_mluop = True
                run(options)
            except AssertionError as e:
                # Scrub out the temp file name from any error messages to simplify assertions.
                return str(e).replace(fp.name, "")
            self.fail(
                "Expected gen_backend_stubs to raise an AssertionError, but it did not."
            )

    def test_valid_single_op(self) -> None:
        yaml_str = """\
aten:
- func: abs"""
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_valid_multiple_ops(self) -> None:
        yaml_str = """\
aten:
- func: add.Tensor
- func: abs"""
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_valid_zero_ops(self) -> None:
        yaml_str = """\
aten:
"""
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_valid_zero_ops_doesnt_require_backend_dispatch_key(self) -> None:
        yaml_str = """\
unsupported:
"""
        # External codegen on a yaml file with no operators is effectively a no-op,
        # so there's no reason to parse the backend
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_valid_with_custom_ops(self) -> None:
        yaml_str = """\
aten:
- func: abs
custom:
- func: my_op(Tensor a) -> Tensor"""
        self.assert_success_from_gen_backend_stubs(yaml_str)

    # supported is a single item (it should be a list)
    def test_nonlist_supported(self) -> None:
        yaml_str = """\
aten:
  abs"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """expected "aten" to be a list, but got: abs""",
        )

    # op that isn't in native_functions.yaml
    def test_invalid_op(self) -> None:
        yaml_str = """\
aten:
- func: abs_BAD"""
        # skip bad op default and generate nothing
        self.assert_success_from_gen_backend_stubs(yaml_str)

    # op missing the "func" keyword
    def test_missing_func(self) -> None:
        yaml_str = """\
aten:
  - xxx: abs"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """You must provide the keyword func for aten ops""",
        )

    # custom op provide concrete function schema
    def test_missing_schema(self) -> None:
        yaml_str = """\
custom:
  - func: my_op"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """You must provide concrete function schema for custom ops""",
        )

    def test_nondict_msg(self) -> None:
        yaml_str = """\
aten:
  - abs"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """expected op msg to be a dict, but got: abs""",
        )

    def test_valid_structured_op(self) -> None:
        yaml_str = """\
aten:
  - func: neg
    structured_delegate: neg.out
  - func: neg_
    structured_delegate: neg.out
  - func: neg.out
    structured: True
    structured_inherits: TensorIteratorBase
  - func: reflection_pad1d
    structured_delegate: reflection_pad1d.out
  - func: reflection_pad1d.out
    structured: True"""
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_valid_derived_type(self) -> None:
        yaml_str = """\
aten:
  - func: abs
    derived_type: cnnl
custom:
  - func: my_op1(Tensor a) -> Tensor
    derived_type: bang
  - func: my_op2(Tensor a) -> Tensor
    derived_type: mluop"""
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_invalid_derived_type(self) -> None:
        yaml_str = """\
custom:
  - func: my_op1(Tensor a) -> Tensor
    derived_type: xxx"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """derived_type support cnnl, bang or mluop, but got: xxx""",
        )

    def test_valid_dispatch(self) -> None:
        yaml_str = """\
aten:
  - func: abs
  - func: _copy_from_and_resize
    dispatch: PrivateUse1, SparsePrivateUse1
  - func: _coalesce
    dispatch: SparsePrivateUse1
  - func: add.Tensor
    dispatch: PrivateUse1, SparsePrivateUse1
    custom_autograd: True
custom:
  - func: my_op1(Tensor a) -> Tensor
  - func: my_op2(Tensor a) -> Tensor
    derived_type: bang
    custom_autograd: True
    dispatch: PrivateUse1, SparsePrivateUse1"""
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_invalid_dispatch(self) -> None:
        yaml_str = """\
aten:
  - func: abs
    dispatch: PrivateUse1, TestDispatch
  - func: add.Tensor
    dispatch: TestDispatch"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """dispatch only support PrivateUse1 and SparsePrivateUse1, but got: TestDispatch""",
        )


if __name__ == "__main__":
    unittest.main()
