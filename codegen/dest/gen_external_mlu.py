# Generates parts of C++ header/source files:
# csrc/aten/generated/RegisterMLU.cpp
# csrc/aten/generated/MLUFunctions.h
# csrc/aten/operators/cnnl/cnnl_kernel.h
# csrc/aten/operators/bang/bang_kernel.h
# csrc/aten/operators/mluop/mluop_kernel.h

import itertools
import textwrap
import warnings
from dataclasses import dataclass
from typing import List, Optional, Union, Dict

from typing_extensions import Literal
from enum import Enum

import torchgen.api.cpp as cpp
import torchgen.api.meta as meta
import torchgen.api.structured as structured
import torchgen.api.dispatcher as dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import (
    BaseCType,
    Binding,
    ConstRefCType,
    CppSignature,
    CppSignatureGroup,
    DispatcherSignature,
    Expr,
    MutRefCType,
    NamedCType,
    NativeSignature,
    tensorT,
)

from torchgen.context import method_with_native_function, native_function_manager
from torchgen.model import (
    Argument,
    OperatorName,
    DeviceCheckType,
    NativeFunction,
    NativeFunctionsGroup,
    SchemaKind,
    TensorOptionsArguments,
)
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import assert_never, mapMaybe, Target

Target = Enum('Target', (
    'CNNL_KERNEL_DECLARATION',
    'BANG_KERNEL_DECLARATION',
    'MLUOP_KERNEL_DECLARATION',
    'REGISTRATION',
    'AUTOGRAD_REGISTRATION',
    'ANONYMOUS_DEFINITION',
    'NAMESPACED_DEFINITION',
    'NAMESPACED_DECLARATION',
))

# Aten ops listed here do not provide fallback implementation
# because at::native::cpu_fallback may use these ops.
# Besides, all custom ops do not provide fallback implementation.
SKIP_FALLBACK_OPS = {
    'empty.memory_format',
    'empty_strided',
    'copy_',
    '_copy_from_and_resize',
}

# Following cases should skip device guard.
# 1. backend ops that allow inputs to have different devices
# 2. compositeimplicit ops that we register our own autograd implementations,
#    device check can be handled by composite backend ops.
SKIP_DEVICE_GUARD_OPS = {
  # _copy_from_and_resize allow different devices
  '_copy_from_and_resize',
  'lstm.data',
  'ctc_loss.Tensor',
  'ctc_loss.IntList',
  'ctc_loss_forward',
  'mask_softmax_dropout_fprop',
  'mask_softmax_dropout_bprop_',
  '_batch_norm_impl_index',
  '_foreach_mul.Tensor'
  '_foreach_mul_.Tensor'
}

@dataclass(frozen=True)
class GenExternalMLU:
    target: Union[
        Literal[Target.CNNL_KERNEL_DECLARATION],
        Literal[Target.BANG_KERNEL_DECLARATION],
        Literal[Target.MLUOP_KERNEL_DECLARATION],
    ]
    selector: SelectiveBuilder
    aux: Dict[OperatorName, Dict[str, object]]

    @method_with_native_function
    def __call__(self, g: Union[NativeFunction, NativeFunctionsGroup]) -> List[str]:
        if isinstance(g, NativeFunction):
            x = self.gen_unstructured(g)
            return [x] if x else []
        elif isinstance(g, NativeFunctionsGroup):
            structured: bool = False
            # For structured kernel, we only generate structured code when
            # it is marked in mlu_functions.yaml
            for f in g.functions():
                s = self.aux.get(f.func.name, None)
                if s and s['structured']:
                    structured = True
                    break

            if structured:
                return self.gen_structured(g)
            else:
                return list(mapMaybe(self.gen_unstructured, g.functions()))

    def gen_unstructured(self, f: NativeFunction):
        if f.func.name not in self.aux.keys():
            return None

        sig = NativeSignature(f.func, prefix='', symint=False)
        impl_name = cpp.name(f.func)
        returns_type = sig.returns_type().cpp_type()
        args = sig.arguments()
        args_str = ', '.join(a.decl() for a in args)
        derived_type = self.aux[f.func.name].get('derived_type')
        dispatch = self.aux[f.func.name].get('dispatch')
        if dispatch == 'SparsePrivateUse1':
            impl_name = impl_name + '_sparse'

        # autograd kernel declaration
        has_autograd = self.aux[f.func.name].get('custom_autograd', False)
        autograd_decl = ""
        if has_autograd:
            overload_name = f.func.name.overload_name
            autograd_impl_name = impl_name
            if overload_name:
                autograd_impl_name = impl_name + '_' + overload_name
            autograd_decl = f"""
{returns_type} {derived_type}_{autograd_impl_name}_autograd({args_str});
"""

        # kernel declaration = backend decl declaration + autograd decl declaration
        kernel_decl = f"""\
{returns_type} {derived_type}_{impl_name}({args_str});{autograd_decl}
"""

        if self.target == Target.CNNL_KERNEL_DECLARATION:
            if derived_type == 'cnnl':
                return kernel_decl
        elif self.target == Target.BANG_KERNEL_DECLARATION:
            if not self.aux[f.func.name].get('use_bang', None):
                return None
            if derived_type == 'bang':
                return kernel_decl
        elif self.target == Target.MLUOP_KERNEL_DECLARATION:
            if not self.aux[f.func.name].get('use_mluop', None):
                return None
            if derived_type == 'mluop':
                return kernel_decl
        else:
            assert_never(self.target)

    def gen_structured(self, g: NativeFunctionsGroup) -> List[str]:
        meta_name = meta.name(g)
        out_args = structured.impl_arguments(g)
        args = structured.meta_arguments(g)
        args_str = ", ".join(a.decl() for a in args)
        meta_return = "void"

        metadata = self.aux[g.out.func.name].get('metadata', None)
        override_meta = self.aux[g.out.func.name].get('override_meta', False)
        override_impl = self.aux[g.out.func.name].get('override_impl', False)
        # NB:
        # 1. If metadata is not None, we will reuse declarations generated by Pytorch
        # by default. If override_meta or override_impl is True, we generate our own declarations.
        # 2. If metadata is None, we generate our own declarations.
        if metadata is not None and (not override_meta and not override_impl):
            return []
        else:
            prefix = ""
            kernel_name = dispatcher.name(g.out.func)

            precomputed = g.out.precomputed if g.structured else None
            if precomputed:
                # Generate the template declaration with one bool parameter for each
                # precomputed element. Each parameter is true if the corresponding (in
                # terms of position) precomputed element has been set.
                precomputed_values = [*precomputed.replace.values(), precomputed.add]
                precomputed_elements = [
                    elem for replace_list in precomputed_values for elem in replace_list
                ]
                precomputed_template_parameters = [
                    elem.name.upper() for elem in precomputed_elements
                ]
                precomputed_template_params_str = ", ".join(
                    f"bool {param} = false" for param in precomputed_template_parameters
                )
                precompute_template_decl = f"template <{precomputed_template_params_str}>"

                # Generate a string containing declarations of all precomputed elements.
                precomputed_elements_with_cpp_types = [
                    structured.argument_type(elem, binds=elem.name)
                    for elem in precomputed_elements
                ]

                precomputed_elements_decl = ";\n".join(
                    f"{elem.cpp_type(strip_ref=True)} {elem.name}"
                    for elem in precomputed_elements_with_cpp_types
                )

                # Generate "setter" methods for each precomputed element. Each method will return
                # a new instance of precompute_out with the template parameter that corresponds to
                # the member set by the method to true (to indicate that it has been set).
                setter_methods = []
                for i, elem in enumerate(precomputed_elements):
                    # Generate the signature. The return type will be the same
                    # as the type of `this` but with the template parameter
                    # corresponding to the element set by this method set to true.
                    # The assert generated below will ensure that this template
                    # parameter is false on the type of `this`.
                    return_ty_templates = ", ".join(
                        precomputed_template_parameters[:i]
                        + ["true"]
                        + precomputed_template_parameters[i + 1 :]
                    )
                    return_ty = f"precompute_out<{return_ty_templates}>"
                    elem_cpp_ty = precomputed_elements_with_cpp_types[i].cpp_type(
                        strip_ref=True
                    )
                    signature = f"{return_ty} set_{elem.name}({elem_cpp_ty} value)"

                    # Generate an assert which checks that the
                    # template parameter corresponding to the precomputed
                    # element that is set by this method is false on the
                    # class corresponding to the object that `this` points to.
                    # This ensures that each element can be set only once.
                    assert_msg = f'"{precomputed_elements[i].name} already set"'
                    assert_stmt = f"static_assert({precomputed_template_parameters[i]} == false, {assert_msg});"

                    # Generate the new object construction block. All state
                    # except the element that this method sets is copied from the
                    # object that `this` points to. The value for the element that
                    # the method sets is taken from a method parameter.
                    construction_stmts = []
                    construction_stmts.append(f"{return_ty} ret;")

                    for j, elem in enumerate(precomputed_elements):
                        if i == j:
                            construction_stmts.append(f"ret.{elem.name} = value;")
                        else:
                            construction_stmts.append(
                                f"ret.{elem.name} = this->{elem.name};"
                            )

                    construction_stmts.append("return ret;")
                    construction_block = "\n".join(construction_stmts)

                    setter_methods.append(
                        f"""
                        {signature} {{
                            {assert_stmt}
                            {construction_block}
                        }}
                    """
                    )
                setter_methods_decl = "\n".join(setter_methods)

                # Meta should return an instance of the struct containing the precomputed elements.
                meta_return_template_params = ", ".join(
                    ["true"] * len(precomputed_template_parameters)
                )
                # This typedef (actually a using statement) is needed so that TORCH_META_FUNC can reuse the return
                # type (which has a variable number of template parameters).
                meta_return_typedef = f"using meta_return_ty = precompute_out <{meta_return_template_params}>;"
                meta_return = "meta_return_ty"
                precomputed_decl = f"""
                    {precompute_template_decl}
                    struct precompute_out {{
                        {setter_methods_decl}
                        {precomputed_elements_decl};
                }};"""
            else:
                meta_return_typedef = ""
                precomputed_decl = ""

            meta_str = ""
            if override_meta:
                meta_str = f"""\
{precomputed_decl}
{meta_return_typedef}
{meta_return} meta({args_str});
"""

            res = [
                f"""\
struct {prefix}structured_{kernel_name}_mlu : public at::meta::structured_{meta_name} {{
{meta_str}
void impl({', '.join(a.decl() for a in out_args)});
}};
"""
            ]
        derived_type = self.aux[g.out.func.name].get('derived_type')
        if self.target == Target.CNNL_KERNEL_DECLARATION:
            if derived_type == 'cnnl':
                return res
        elif self.target == Target.BANG_KERNEL_DECLARATION:
            if not self.aux[g.out.func.name].get('use_bang', None):
                return []
            if derived_type == 'bang':
                return res
        elif self.target == Target.MLUOP_KERNEL_DECLARATION:
            if not self.aux[g.out.func.name].get('use_mluop', None):
                return []
            if derived_type == 'mluop':
                return res
        else:
            assert_never(self.target)


@dataclass(frozen=True)
class RegisterMLU:
    target: Union[
        Literal[Target.REGISTRATION],
        Literal[Target.AUTOGRAD_REGISTRATION],
        Literal[Target.ANONYMOUS_DEFINITION],
        Literal[Target.NAMESPACED_DEFINITION],
        Literal[Target.NAMESPACED_DECLARATION],
    ]
    selector: SelectiveBuilder

    aux: Dict[OperatorName, Dict[str, object]]

    # Whether or not to generate symint registrations or not. External users
    # of codegen who don't care about symints can set this to false to get
    # non-SymInt codegen
    symint: bool

    @staticmethod
    def gen_device_check(
        type: DeviceCheckType, args: List[Argument], method_name: str
    ) -> str:
        if type == DeviceCheckType.NoCheck:
            return "  // No device check\n"

        device_check = "std::optional<c10::Device> common_device = c10::nullopt;\n"
        device_check += "(void)common_device; // Suppress unused variable warning\n"
        for arg in args:
            # Only tensor like arguments are eligible
            if arg.type.is_tensor_like():
                device_check += f"""
  c10::impl::check_and_update_common_device(common_device, {arg.name}, "{method_name}", "{arg.name}");"""
        return device_check

    @method_with_native_function
    def __call__(self, g: Union[NativeFunction, NativeFunctionsGroup]) -> List[str]:
        if isinstance(g, NativeFunction):
            x = self.gen_unstructured(g)
            return [x] if x else []
        elif isinstance(g, NativeFunctionsGroup):
            structured: bool = False
            # For structured kernel, we only generate structured code when
            # it is marked in mlu_functions.yaml
            for f in g.functions():
                s = self.aux.get(f.func.name, None)
                if s and s['structured']:
                    structured = True
                    break

            if structured:
                return self.gen_structured(g)
            else:
                return list(mapMaybe(self.gen_unstructured, g.functions()))

    def wrapper_kernel_sig(
        self, f: NativeFunction
    ) -> Union[NativeSignature, DispatcherSignature]:
        # The prefix is just to ensure uniqueness. The Dispatcher API doesn't guarantee unique kernel names.
        return DispatcherSignature.from_schema(
            f.func, prefix=f"wrapper_{f.func.name.overload_name}_", symint=self.symint
        )

    def gen_structured(self, g: NativeFunctionsGroup) -> List[str]:
        structured_gen = StructuredRegisterMLU(
            self.target,
            self.selector,
            self.aux,
            self.symint,
            g,
        )
        return list(mapMaybe(structured_gen.gen_one, g.functions()))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           UNSTRUCTURED
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def gen_unstructured(self, f: NativeFunction) -> Optional[str]:
        with native_function_manager(f):
            if f.manual_kernel_registration:
                return None

            if (
                self.target is Target.REGISTRATION
                and not self.selector.is_native_function_selected(f)
            ):
                return None

            # You should register functional, inplace and out completely,
            # but now some case is lost, we skip the lost op.
            if f.func.name not in self.aux.keys():
                return None

            sig = self.wrapper_kernel_sig(f)

            name = sig.name()
            returns_type = sig.returns_type().cpp_type()
            args = sig.arguments()
            args_str = ", ".join(a.defn() for a in args)
            cpp_sig_group = CppSignatureGroup.from_native_function(
                f, method=False, fallback_binding=False
            )

            impl_name = cpp.name(f.func)
            derived_type = self.aux[f.func.name].get('derived_type')
            dispatch = self.aux[f.func.name].get('dispatch')
            if dispatch == 'SparsePrivateUse1' and self.target is not Target.AUTOGRAD_REGISTRATION:
                impl_name = impl_name + '_sparse'
            metadata = self.aux[f.func.name].get('metadata', None)

        if self.target is Target.REGISTRATION:
            if f.manual_kernel_registration:
                return None
            else:
                payload = f"TORCH_FN(wrapper_{name})"
                return f'm.impl("{f.func.name}", {payload});\n'
        elif self.target is Target.AUTOGRAD_REGISTRATION:
            has_autograd = self.aux[f.func.name].get('custom_autograd', False)
            if not has_autograd:
                return None
            overload_name = f.func.name.overload_name
            autograd_impl_name = impl_name
            if overload_name:
                autograd_impl_name = impl_name + '_' + overload_name
            autograd_impl_name = f"torch_mlu::ops::{derived_type}_{autograd_impl_name}_autograd"
            payload = f"TORCH_FN({autograd_impl_name})"
            return f'm.impl("{f.func.name}", {payload});\n'
        elif self.target is Target.NAMESPACED_DECLARATION:
            result = ""
            for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                result += f"{cpp_sig.decl()};\n"
            return result
        elif self.target is Target.NAMESPACED_DEFINITION:
            def generate_defn(cpp_sig: CppSignature) -> str:
                return f"""
{cpp_sig.defn()} {{
return wrapper_{sig.name()}({', '.join(e.expr for e in translate(cpp_sig.arguments(), sig.arguments()))});
}}
"""
            result = ""
            for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                result += generate_defn(cpp_sig)
            return result

        elif self.target is Target.ANONYMOUS_DEFINITION:
            impl_name = f"torch_mlu::ops::{derived_type}_{impl_name}"
            symint = False
            if dispatch == "SparsePrivateUse1" and metadata is not None:
                impl_name = f"{metadata.cpp_namespace}::{metadata.kernel}"
                symint = metadata.supports_symint()
                if symint:
                    assert (
                        f.func.has_symint()
                    ), f"attempted to define symint kernel for {dispatch} without SymInt in schema"

            kernel_sig = NativeSignature(f.func, prefix='', symint=symint)
            args_exprs_str = ", ".join(
                e.expr
                for e in translate(
                    sig.arguments(), kernel_sig.arguments(), method=False
                )
            )
            sig_args_expr_str = ", ".join(a.name for a in sig.arguments())
            impl_call = f"return {impl_name}({args_exprs_str});"

            impl_fn = f"auto impl_fn = {name};"
            # add fallback str, only support aten fallback
            # TODO(PYTORCH-12760): support fallback to SparseCPU
            if self.aux[f.func.name].get('ns', None) == 'aten' and str(f.func.name) not in SKIP_FALLBACK_OPS \
               and dispatch != "SparsePrivateUse1":
                if len(f.func.name.overload_name):
                    aten_op_str = f"ATEN_OP2({f.func.name.name}, {f.func.name.overload_name})"
                else:
                    aten_op_str = f"ATEN_OP({f.func.name.name})"
                fallback_str = f"at::native::call_fallback_fn_symint<&mlu_fail_fallback, {aten_op_str}>::call"
                fallback_fn = f"auto fallback_fn = {fallback_str};"
                param_str = ', '.join(['impl_fn', 'fallback_fn', f"{sig_args_expr_str}"])
                call_str = f"""return op_call<{returns_type}>({param_str});"""
            else:
                fallback_fn = '// No fallback implementation'
                param_str = ', '.join(['impl_fn', f"{sig_args_expr_str}"])
                call_str = f"""return op_call2<{returns_type}>({param_str});"""

            device_check = "  // No device check\n"
            # Backends that require device guards presumably also require device checks.
            if str(f.func.name) not in SKIP_DEVICE_GUARD_OPS:
                device_check_args = itertools.chain(
                    f.func.arguments.out, f.func.arguments.flat_positional
                )
                device_check = RegisterMLU.gen_device_check(
                    f.device_check, list(device_check_args), name
                )
            # MLU needs device guard.
            device_guard = "// DeviceGuard omitted"  # default
            if f.device_guard and str(f.func.name) not in SKIP_DEVICE_GUARD_OPS:
                has_tensor_options = any(
                    isinstance(a, TensorOptionsArguments)
                    for a in f.func.arguments.non_out
                )
                if has_tensor_options:
                    # kernel is creating a tensor
                    device_guard = """
const c10::DeviceGuard device_guard(device_or_default(device));"""

                    # MLU requires special handling
                    device_guard = (
                        f"at::globalContext().lazyInitPrivateUse1();\n{device_guard}"
                    )
                else:
                    # kernel is operating on existing tensors

                    # There is precedence for which argument we use to do
                    # device guard.  This describes the precedence order.
                    self_arg = (
                        [f.func.arguments.self_arg.argument]
                        if f.func.arguments.self_arg is not None
                        else []
                    )
                    candidate_args = itertools.chain(
                        self_arg,
                        f.func.arguments.out,
                        f.func.arguments.flat_positional,
                    )

                    # Only tensor like arguments are eligible
                    device_of = next(
                        (
                            f"{a.name}"
                            for a in candidate_args
                            if a.type.is_tensor_like()
                        ),
                        None,
                    )
                    if device_of is not None:
                        device_guard = f"const c10::OptionalDeviceGuard device_guard(device_of({device_of}));"
            return f"""\
namespace {{
{returns_type} {name}({args_str}) {{
  {device_check}
  {device_guard}
  {impl_call}
}}

{returns_type} wrapper_{name}({args_str}) {{
  {impl_fn}
  {fallback_fn}
  {call_str}
}}
}}  // anonymous namespace
"""
        else:
            assert_never(self.target)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           STRUCTURED
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@dataclass(frozen=True)
class StructuredRegisterMLU(RegisterMLU):
    g: NativeFunctionsGroup = None

    def gen_class_set_output_functions(
        self, k: SchemaKind, parent_class: str, generate_super: bool
    ) -> str:
        if generate_super:
            set_output_super = f"{parent_class}::set_output_raw_strided(output_idx, sizes, strides, options, names);"
        else:
            set_output_super = ""

        def gen_set_output_function(name: str, maybe_create_proxy: bool) -> str:
            maybe_star = "*" if k is SchemaKind.functional else ""
            return f"""
void set_output_{name}(
    int64_t output_idx, at::IntArrayRef sizes, at::IntArrayRef strides,
    at::TensorOptions options, at::DimnameList names
) override {{
{textwrap.indent(self.gen_class_set_output_body(k, maybe_create_proxy), "    ")}
    if (!names.empty()) {{
      at::namedinference::propagate_names({maybe_star}outputs_[output_idx], names);
    }}
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
{textwrap.indent(set_output_super, "    ")}
}}
"""

        return f"""
{gen_set_output_function("strided", maybe_create_proxy=True)}
{gen_set_output_function("raw_strided", maybe_create_proxy=False)}
"""

    def gen_class_set_output_body(self, k: SchemaKind, maybe_create_proxy: bool) -> str:
        maybe_set_guard_line = """
auto current_device = guard_.current_device();
if (C10_UNLIKELY(current_device.has_value())) {
  TORCH_INTERNAL_ASSERT(*current_device == options.device(),
    "structured kernels don't support multi-device outputs");
} else {
  guard_.reset_device(options.device());
}
"""
        if maybe_create_proxy:
            create_proxy = """
auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
if (C10_UNLIKELY(maybe_proxy.has_value())) {
    proxy_outputs_[output_idx] = c10::ExclusivelyOwned<at::Tensor>(std::move(maybe_proxy).value());
}
"""
        else:
            create_proxy = ""


        if k is SchemaKind.functional:
            if self.g.out.structured_inherits == "TensorIteratorBase":
              return f"""{maybe_set_guard_line}
auto& out = *outputs_[output_idx];
create_out_or_resize(out, sizes, strides, options);"""
            else:
              return f"""{maybe_set_guard_line}
outputs_[output_idx] = create_out(sizes, strides, options);"""
        elif k is SchemaKind.inplace:
            return f"""{maybe_set_guard_line}
const auto& out = outputs_[output_idx].get();
check_inplace(out, sizes, options);
{create_proxy}"""
        elif k is SchemaKind.out:
            return f"""{maybe_set_guard_line}
const auto& out = outputs_[output_idx].get();
resize_out(out, sizes, strides, options);
{create_proxy}"""
        elif k is SchemaKind.mutable or k is SchemaKind.scratch:
            raise AssertionError(
                f"{k} structured operators are currently not supported"
            )
        else:
            assert_never(k)

    # returns the definition of a ctor, as well as how to construct
    # this class to a variable named op
    def gen_class_ctor(self, k: SchemaKind, class_name: str, returns: int) -> str:
        if k is SchemaKind.functional:
            return ""
        elif k is SchemaKind.inplace:
            # TODO: Make sure out argument is guaranteed to be self
            return f"{class_name}(at::Tensor& self) : outputs_{{std::ref(self)}} {{}}"
        elif k is SchemaKind.out:
            out_args = ", ".join(f"at::Tensor& out{i}" for i in range(returns))
            out_refs = ", ".join(f"std::ref(out{i})" for i in range(returns))
            return f"{class_name}({out_args}) : outputs_{{ {out_refs} }} {{}}"
        elif k is SchemaKind.mutable or k is SchemaKind.scratch:
            raise AssertionError(
                f"{k} structured operators are currently not supported"
            )
        else:
            assert_never(k)

    def gen_class(
        self,
        f: NativeFunction,
        k: SchemaKind,
        *,
        class_name: str,
        parent_class: str,
        generate_super: bool,
    ) -> str:
        if k is SchemaKind.functional:
            output_type = "c10::ExclusivelyOwned<at::Tensor>"
            output_value = "*outputs_[output_idx]"
            proxy_field = ""
        elif k is SchemaKind.inplace:
            output_type = "std::reference_wrapper<at::Tensor>"
            output_value = "proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx] : outputs_[output_idx].get()"
            proxy_field = f"std::array<std::optional<c10::ExclusivelyOwned<at::Tensor>>, {len(f.func.returns)}> proxy_outputs_;"
        elif k is SchemaKind.out:
            output_type = "std::reference_wrapper<at::Tensor>"
            output_value = "proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx] : outputs_[output_idx].get()"
            proxy_field = f"std::array<std::optional<c10::ExclusivelyOwned<at::Tensor>>, {len(f.func.returns)}> proxy_outputs_;"

        guard_field = "torch_mlu::mlu::OptionalMLUGuard guard_;"

        indent = " " * 4
        class_ctor_str = self.gen_class_ctor(k, class_name, len(f.func.returns))
        lines = (
            f"struct {class_name} final : public {parent_class} {{",
            f"{textwrap.indent(class_ctor_str, indent)}",
            f"{textwrap.indent(self.gen_class_set_output_functions(k, parent_class, generate_super), indent)}",
            "    const at::Tensor& maybe_get_output(int64_t output_idx) override {",
            f"      return {output_value};",
            "    }",
            f"    std::array<{output_type}, {len(f.func.returns)}> outputs_;",
            f"{textwrap.indent(proxy_field, indent)}",
            f"{textwrap.indent(guard_field, indent)}",
            "};",
        )
        return "\n".join(line for line in lines if line)

    @method_with_native_function
    def gen_one(self, f: NativeFunction) -> Optional[str]:
        assert not f.manual_kernel_registration

        if (
            self.target is Target.REGISTRATION
            and not self.selector.is_native_function_selected(f)
        ):
            return None

        if f.func.name not in self.aux.keys():
            return None

        # Note [Direct dispatch bindings]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Signature of the non-dispatched function we'll expose in a header
        # (e.g., at::cpu::add).  We don't generate methods (TODO: do this
        # when CPUTensor class is a thing); nor do we generate fallback
        # bindings for manual_cpp_binding functions.
        cpp_sig_group = CppSignatureGroup.from_native_function(
            f, method=False, fallback_binding=False
        )

        # Signature of the wrapper function we'll register to the dispatcher
        sig = NativeSignature(
            f.func,
            prefix="wrapper_",
            symint=False,
        )

        if self.target is Target.NAMESPACED_DECLARATION:
            result = ""
            for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                result += f"{cpp_sig.decl()};\n"
            return result

        elif self.target is Target.NAMESPACED_DEFINITION:

            def generate_defn(cpp_sig: CppSignature) -> str:
                return f"""
{cpp_sig.defn()} {{
return wrapper_{sig.name()}({', '.join(e.expr for e in translate(cpp_sig.arguments(), sig.arguments()))});
}}
"""

            result = ""
            for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                result += generate_defn(cpp_sig)
            return result

        elif self.target is Target.ANONYMOUS_DEFINITION:

            k = f.func.kind()

            # Construct the body of the wrapper function with signature sig
            sig_body = []
            # We'll use context to keep track of any variables we've brought
            # into scope while generating code
            context: List[Union[Binding, Expr]] = list(sig.arguments())

            # 1. If metadata is not None, CPU and CUDA has same kernel name, that means
            # TensorIterator is used. MLU reuse the kernel by default. If override_meta
            # or override_impl is appointed, we generate MLU unique structured kernel.
            # 2. If metadata is None, we generate MLU unique structured kernel by default.
            metadata = self.aux[self.g.out.func.name].get('metadata', None)
            override_meta = self.aux[self.g.out.func.name].get('override_meta', False)
            override_impl = self.aux[self.g.out.func.name].get('override_impl', False)
            if metadata and (not override_meta and not override_impl):
                class_name = f"structured_{metadata.kernel}_{k.name}"
                parent_class = f"at::native::structured_{metadata.kernel}"
            else:
                kernel_name = dispatcher.name(self.g.out.func)
                class_name = f"structured_{kernel_name}_mlu_{k.name}"
                parent_class = f"torch_mlu::ops::structured_{kernel_name}_mlu"

            #  MLU needs device guard
            device_check_args = itertools.chain(
                f.func.arguments.out, f.func.arguments.flat_positional
            )
            sig_body.append(
                RegisterMLU.gen_device_check(
                    f.device_check, list(device_check_args), sig.name()
                )
            )

            if k is SchemaKind.functional:
                sig_body.append(f"{class_name} op;")
            elif k is SchemaKind.inplace:
                sig_body.append(f"{class_name} op(self);")
            elif k is SchemaKind.out:
                out_args_str = ", ".join(a.name for a in f.func.arguments.out)
                sig_body.append(f"{class_name} op({out_args_str});")

            # Translate the input native arguments into structured
            # arguments for the meta call
            meta_exprs = ", ".join(
                e.expr
                for e in translate(
                    context, structured.meta_arguments(self.g), method=False
                )
            )

            if self.g.out.precomputed:
                # If this function group has precomputed elements, the meta function
                # returns a struct containing them which must be saved so that it
                # can be unpacked when generating code to call the impl.
                sig_body.append(f"auto precompute = op.meta({meta_exprs});")

                # Put all of the contents of the precompute struct into the context
                # so that translate will be able to return the correct args for the
                # call to the impl.
                precomputed_values = [
                    *self.g.out.precomputed.replace.values(),
                    self.g.out.precomputed.add,
                ]
                for precomputed_elems in precomputed_values:
                    for arg in precomputed_elems:
                        context.append(
                            Expr(
                                expr=f"precompute.{arg.name}",
                                type=structured.argument_type(arg, binds=arg.name),
                            )
                        )

                # Add a use of the precompute struct so FB internal compilers don't
                # complain that there is an unused variable.
                sig_body.append("(void)precompute;")
            else:
                sig_body.append(f"op.meta({meta_exprs});")

            # After running meta, op.outputs_ is guaranteed to be valid;
            # add it to the context
            out_args = structured.out_arguments(self.g)
            for i, out_arg in enumerate(out_args):
                assert ConstRefCType(BaseCType(tensorT)) == out_arg.nctype.type

                if k is SchemaKind.out:
                    expr = f"op.maybe_get_output({i})"
                else:
                    maybe_star = "*" if k is SchemaKind.functional else ""
                    expr = f"{maybe_star}op.outputs_[{i}]"

                context.append(
                    Expr(
                        expr=expr,
                        # TODO: Stop hardcoding that the output type is a Tensor.  Note
                        # that for the codegen here this is fine because outputs_ is
                        # hardcoded to be tensor already
                        type=NamedCType(
                            out_arg.nctype.name, MutRefCType(BaseCType(tensorT))
                        ),
                    )
                )

            impl_exprs = ", ".join(
                e.expr
                for e in translate(
                    context, structured.impl_arguments(self.g), method=False
                )
            )

            # add fallback str
            returns_type = sig.returns_type().cpp_type()
            args_str = ", ".join(a.defn() for a in sig.arguments())
            # only support aten fallback
            if len(f.func.name.overload_name):
                aten_op_str = f"ATEN_OP2({f.func.name.name}, {f.func.name.overload_name})"
            else:
                aten_op_str = f"ATEN_OP({f.func.name.name})"

            sig_args_expr_str = ', '.join(e.name for e in sig.arguments())
            impl_fn = f"auto impl_fn = {sig.name()};"
            fallback_str = f"at::native::call_fallback_fn<&mlu_fail_fallback, {aten_op_str}>::call"
            fallback_fn = f"auto fallback_fn = {fallback_str};"
            param_str = ', '.join(['impl_fn', 'fallback_fn', f"{sig_args_expr_str}"])
            call_str = f"""return op_call<{returns_type}>({param_str});"""

            tensor_bridge = f""
            if self.g.out.structured_inherits == "TensorIteratorBase":
                tensor_bridge = f"""TensorIteratorBridge iter_bridge;
iter_bridge.to_build(op, "{self.g.functional.func.name.name}");"""
            sig_body.append(f"""\
{tensor_bridge}
op.impl({impl_exprs});
""")
            if self.g.out.structured_inherits == "TensorIteratorBase":
                sig_body.append(f"""iter_bridge.cast_outputs(op);""");
            # Go over each output, and check if there is a proxy created for it.
            # If so, copy it over to the original output.
            if k is SchemaKind.out or k is SchemaKind.inplace:
                for i in range(len(f.func.returns)):
                    sig_body.append(
                        f"if (op.proxy_outputs_[{i}].has_value()) op.outputs_[{i}].get().copy_(**op.proxy_outputs_[{i}]);"
                    )

            # Destructively return the final tensors
            # TODO: Do this in translate instead
            if k is SchemaKind.functional:
                if len(f.func.returns) == 1:
                    ret_expr = "std::move(op.outputs_[0]).take()"  # small optimization
                else:
                    moved = ", ".join(
                        f"std::move(op.outputs_[{i}]).take()"
                        for i in range(len(f.func.returns))
                    )
                    ret_expr = f"std::make_tuple({moved})"
            elif k is SchemaKind.inplace:
                ret_expr = "self"
            elif k is SchemaKind.out:
                if len(f.func.returns) == 1:
                    ret_expr = f.func.arguments.out[0].name
                else:
                    refs = ", ".join(a.name for a in f.func.arguments.out)
                    ret_expr = f"std::forward_as_tuple({refs})"
            sig_body.append(f"return {ret_expr};")

            sig_body_str = "\n".join(sig_body)

            # For an overview of what this template code looks like, see
            # https://github.com/pytorch/rfcs/pull/9
            return f"""\
{self.gen_class(
f, k,
class_name=class_name,
parent_class=parent_class,
generate_super=self.g.out.structured_inherits is not None
)}

{sig.defn()} {{
{sig_body_str}
}}

{returns_type} wrapper_{sig.name()}({args_str}) {{
  {impl_fn}
  {fallback_fn}
  {call_str}
}}
"""

        elif self.target is Target.REGISTRATION:
            return f'm.impl("{f.func.name}", TORCH_FN(wrapper_{sig.name()}));'
        elif self.target is Target.AUTOGRAD_REGISTRATION:
            has_autograd = self.aux[f.func.name].get('custom_autograd', False)
            if has_autograd:
                warnings.warn(f"""Override strutured kernel's autograd impl is not supported""")
            return None
        else:
            assert_never(self.target)
            # Silence mypy's "Missing return statement" error
            return None
