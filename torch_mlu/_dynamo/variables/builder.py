# mypy: ignore-errors

import collections
import contextlib
import enum
import functools
import inspect
import itertools
import operator
import re
import sys
import types

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

import torch
import torch_mlu

from torch._ops import HigherOrderOperator
from torch._streambase import _EventBase, _StreamBase
from torch._subclasses.fake_tensor import FakeTensor, maybe_get_fake_mode
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
)
from torch.fx.immutable_collections import immutable_list
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch._dynamo import config, replay_record, trace_rules

from torch._dynamo.device_interface import get_registered_device_interfaces
from torch._dynamo.exc import InternalTorchDynamoError, unimplemented
from torch._dynamo.guards import GuardBuilder, install_guard
from torch._dynamo.side_effects import SideEffects
from torch._dynamo.source import (
    AttrSource,
    ConstDictKeySource,
    ConvertIntSource,
    GetItemSource,
)
from torch._dynamo.trace_rules import is_callable_allowed, is_numpy
from torch._dynamo.utils import (
    build_checkpoint_variable,
    clone_input,
    get_fake_value,
    is_function_or_wrapper,
    is_namedtuple,
    is_typing,
    is_utils_checkpoint,
    istype,
    preserve_rng_state,
    unwrap_with_attr_name_if_wrapper,
)

from torch._dynamo.variables.base import MutableLocal, typestr, VariableTracker
from torch._dynamo.variables.constant import ConstantVariable, EnumVariable
from torch._dynamo.variables.ctx_manager import (
    AutocastModeVariable,
    EventVariable,
    NullContextVariable,
    PreserveVersionContextVariable,
    StreamContextVariable,
    StreamVariable,
)
from torch._dynamo.variables.dicts import (
    ConstDictVariable,
    DataClassVariable,
    DefaultDictVariable,
    HFPretrainedConfigVariable,
    PythonSysModulesVariable,
)
from torch._dynamo.variables.distributed import (
    DeviceMeshVariable,
    PlacementClassVariable,
    PlacementVariable,
    ProcessGroupVariable,
)
from torch._dynamo.variables.functions import (
    CollectiveFunctionRewriteVariable,
    FunctoolsPartialVariable,
    TritonKernelVariable,
    UserMethodVariable,
)
from torch._dynamo.variables.higher_order_ops import TorchHigherOrderOperatorVariable
from torch._dynamo.variables.iter import ItertoolsVariable
from torch._dynamo.variables.lazy import LazyVariableTracker
from torch._dynamo.variables.lists import (
    ListVariable,
    NamedTupleVariable,
    RestrictedListSubclassVariable,
    SizeVariable,
    TupleVariable,
)
from torch._dynamo.variables.misc import (
    AutogradFunctionContextVariable,
    AutogradFunctionVariable,
    DebuggingVariable,
    GetAttrVariable,
    GetSetDescriptorVariable,
    MethodWrapperVariable,
    NumpyVariable,
    PythonModuleVariable,
    SavedTensorBox,
    TypingVariable,
)
from torch._dynamo.variables.optimizer import OptimizerVariable

from torch._dynamo.variables.sdpa import SDPAParamsVariable
from torch._dynamo.variables.tensor import (
    SymNodeVariable,
    TensorSubclassVariable,
    TensorVariable,
)
from torch._dynamo.variables.torch import (
    TorchCtxManagerClassVariable,
    TorchInGraphFunctionVariable,
)
from torch._dynamo.variables.torch_function import TensorWithTFOverrideVariable
from torch._dynamo.variables.user_defined import (
    KeyedJaggedTensorVariable,
    UserDefinedClassVariable,
    UserDefinedObjectVariable,
)
from torch._dynamo.variables.builder import (
    GraphArg,
    VariableBuilder,
    TrackedFake,
    wrap_to_fake_tensor_and_record,
    _missing,
)


def _wrap(self, value):
    # import here to avoid circular dependencies
    from torch.utils._triton import has_triton

    if has_triton():
        from triton.runtime.autotuner import Autotuner
        from triton.runtime.jit import JITFunction
    else:

        class JITFunction:
            pass

        class Autotuner:
            pass

    # Handle exact type() match
    type_dispatch = self._type_dispatch().get(type(value))
    if type_dispatch is not None:
        return type_dispatch(self, value)

    # Handle exact id() match
    id_dispatch = self._id_dispatch().get(id(value))
    if id_dispatch is not None:
        return id_dispatch(self, value)

    # Note - There are some nested values where types mismatch!
    # We want to get those out and wrap those.
    value = inspect.getattr_static(value, "_torchdynamo_inline", value)

    # Everything else (NB: order matters!)
    if is_traceable_wrapper_subclass(value) or istype(
        value, config.traceable_tensor_subclasses
    ):
        return self.wrap_tensor(value)
    elif is_namedtuple(value):
        return self.wrap_listlike(value)

    elif value is torch.utils._pytree.SUPPORTED_NODES:
        # For SUPPORTED_NODES, we guard on the dictionary version (PEP509)
        # under the assumption that the values themselves don't change.
        self.install_guards(GuardBuilder.DICT_VERSION)
        result = {
            ConstantVariable.create(k): UserDefinedObjectVariable(
                v,
                source=GetItemSource(
                    self.get_source(), ConstDictKeySource(self.get_source(), i)
                ),
            )
            for i, (k, v) in enumerate(value.items())
        }
        return ConstDictVariable(result, type(value))
    elif value is sys.modules:
        self.install_guards(GuardBuilder.FUNCTION_MATCH)
        return PythonSysModulesVariable(source=self.source)
    elif istype(value, (dict, collections.defaultdict, collections.OrderedDict)):
        if not value and self.get_source().is_nn_module():
            # It is faster to guard on 'false' property than to guard
            # on actual dict keys, but we can't do this fast guard in general because
            # it omits a crucial type check that ensures the value is actually still a dict at runtime.

            # Why is this OK for (specialized) nnmodules? We set up a setattr hook
            # to check for module property mutations, which does a reasonable,
            # but not completely secure job ensuring a property wasn't changed.
            self.install_guards(GuardBuilder.BOOL_FALSE)
        else:
            self.install_guards(GuardBuilder.DICT_LENGTH)

        # Optimisation for the common case strings, ints, etc
        all_const = all(ConstantVariable.is_literal(k) for k in value.keys())
        if all_const:
            self.install_guards(GuardBuilder.DICT_CONST_KEYS)

        # We need all the keys to be hashable. We do this within the
        # _HashableTracker class in dicts.py
        def build_key_value(i, k, v):
            if all_const:
                key = ConstantVariable.create(k)
                source_key = k
            else:
                source_key = ConstDictKeySource(self.get_source(), i)
                key = LazyVariableTracker.create(k, source_key)

            source_value = GetItemSource(self.get_source(), source_key)
            value = LazyVariableTracker.create(v, source_value)

            return key, value

        result = dict(
            build_key_value(i, k, v) for i, (k, v) in enumerate(value.items())
        )

        if istype(value, collections.defaultdict):
            factory_source = AttrSource(self.source, "default_factory")
            result = DefaultDictVariable(
                result,
                type(value),
                default_factory=VariableBuilder(self.tx, factory_source)(
                    value.default_factory
                ),
                source=self.source,
            )
        else:
            result = ConstDictVariable(result, type(value), source=self.source)

        return self.set_source_and_track_mutable(value, result)
    elif isinstance(value, torch.nn.Module):
        return self.wrap_module(value)
    elif ConstantVariable.is_literal(value):  # non-atomic literals
        return self.wrap_literal(value)
    elif istype(value, frozenset) and (ConstantVariable.is_literal(x) for x in value):
        # For frozenset, we can guard by object ID instead of value
        # equality, this allows us to handle non-literal values
        self.install_guards(GuardBuilder.ID_MATCH)
        return ConstantVariable.create(value=value, source=self.source)
    elif isinstance(value, enum.Enum):
        self.install_guards(GuardBuilder.ID_MATCH)
        return EnumVariable(value=value, source=self.source)
    elif DebuggingVariable.is_reorderable_logging_function(value):
        # Put this above builtin_callable so that print() can be handled
        # along with other builtin debugging functions
        self.install_guards(GuardBuilder.BUILTIN_MATCH)
        return DebuggingVariable(value, source=self.source)
    elif is_utils_checkpoint(value):
        return build_checkpoint_variable(source=self.source)
    elif isinstance(value, functools.partial):
        func_src = AttrSource(self.get_source(), "func")
        func_obj = VariableBuilder(self.tx, func_src)(value.func)

        args = []
        args_source = AttrSource(self.get_source(), "args")
        for i, arg in enumerate(value.args):
            args.append(VariableBuilder(self.tx, GetItemSource(args_source, i))(arg))

        keywords = {}
        keywords_source = AttrSource(self.get_source(), "keywords")
        for k, v in value.keywords.items():
            if not ConstantVariable.is_literal(k):
                unimplemented("functools.partial with non-literal keyword")
            keywords[k] = VariableBuilder(self.tx, GetItemSource(keywords_source, k))(v)

        install_guard(
            self.get_source().make_guard(GuardBuilder.TYPE_MATCH),
            keywords_source.make_guard(GuardBuilder.DICT_KEYS),
            args_source.make_guard(GuardBuilder.SEQUENCE_LENGTH),
        )
        return FunctoolsPartialVariable(func_obj, args, keywords)
    elif is_typing(value):
        # typing.List, typing.Mapping, etc.
        self.install_guards(GuardBuilder.ID_MATCH)
        return TypingVariable(
            value,
            source=self.source,
        )
    elif np is not None and isinstance(value, np.generic):
        # numpy array scalars: convert to 0D arrays
        return self.wrap_numpy_ndarray(np.asarray(value))
    elif is_numpy(value):
        assert np
        self.install_guards(
            GuardBuilder.FUNCTION_MATCH if callable(value) else GuardBuilder.TYPE_MATCH
        )
        return NumpyVariable(value, source=self.source)
    # NB: These can't be put in type_dispatch, they have to run later
    elif CollectiveFunctionRewriteVariable.can_rewrite(value):
        self.install_guards(GuardBuilder.FUNCTION_MATCH)
        return CollectiveFunctionRewriteVariable.create(
            self.tx,
            value,
            source=self.source,
        )
    elif istype(value, torch.autograd.function.FunctionMeta):
        self.install_guards(GuardBuilder.FUNCTION_MATCH)
        return AutogradFunctionVariable(
            value,
            source=self.source,
        )
    elif isinstance(value, torch.autograd.function.FunctionCtx):
        saved_tensors_source = AttrSource(self.source, "saved_tensors")
        install_guard(
            self.source.make_guard(GuardBuilder.TYPE_MATCH),
            saved_tensors_source.make_guard(GuardBuilder.SEQUENCE_LENGTH),
        )
        saved_tensors = [
            VariableBuilder(self.tx, GetItemSource(saved_tensors_source, n))(v)
            for n, v in enumerate(value.saved_tensors)
        ]
        return self.tx.output.side_effects.track_object_existing(
            value,
            AutogradFunctionContextVariable(
                value,
                source=self.source,
                saved_tensors=SavedTensorBox(saved_tensors),
            ),
        )
    elif (
        isinstance(value, types.MethodType)
        and istype(
            getattr(value, "__self__", None), torch.autograd.function.FunctionMeta
        )
        and getattr(value, "__name__", "") == "apply"
        and value == getattr(value.__self__, "apply", None)
    ):
        # handle aliased autograd function `apply` calls
        self.install_guards(GuardBuilder.FUNCTION_MATCH)
        return GetAttrVariable(
            AutogradFunctionVariable(
                value.__self__, source=AttrSource(self.source, member="__self__")
            ),
            "apply",
        )
    elif callable(value) and trace_rules.lookup_callable(value) is not None:
        if is_callable_allowed(value):
            self.tx.output.has_user_defined_allowed_in_graph = True
        return trace_rules.lookup_callable(value).create_with_source(
            value, source=self.source
        )
    elif np and isinstance(value, np.number):
        return self.wrap_unspecialized_primitive(value)
    elif DataClassVariable.is_matching_object(value):
        self.install_guards(GuardBuilder.TYPE_MATCH)
        return DataClassVariable.wrap(self, value)
    elif HFPretrainedConfigVariable.is_matching_object(value):
        self.install_guards(GuardBuilder.TYPE_MATCH)
        return HFPretrainedConfigVariable(value)
    elif isinstance(value, HigherOrderOperator):
        self.install_guards(GuardBuilder.TYPE_MATCH, GuardBuilder.NAME_MATCH)
        return TorchHigherOrderOperatorVariable.make(value, source=self.source)
    elif isinstance(value, torch.mlu.StreamContext):
        self.install_guards(GuardBuilder.ID_MATCH)
        stream_source = AttrSource(self.source, "stream")
        stream_var = VariableBuilder(self.tx, stream_source)(value.stream)
        return StreamContextVariable.create(self.tx, stream_var)
    elif isinstance(value, _StreamBase):
        self.install_guards(GuardBuilder.ID_MATCH)
        return StreamVariable(
            None,
            value,
            value.device,
            source=self.source,
        )
    elif isinstance(value, (torch._C._SDPAParams)):
        self.install_guards(GuardBuilder.TYPE_MATCH)
        return SDPAParamsVariable.create(self.tx, value, self.source)
    elif isinstance(value, _EventBase):
        self.install_guards(GuardBuilder.ID_MATCH)
        return EventVariable(
            None,
            value,
            source=self.source,
        )
    elif (
        isinstance(value, torch._C._TensorMeta)
        and value in config.traceable_tensor_subclasses
    ):
        return TensorSubclassVariable(value, source=self.source)
    elif (
        istype(value, contextlib.nullcontext)
        and inspect.getattr_static(value, "enter_result", None) is None
    ):
        self.install_guards(GuardBuilder.TYPE_MATCH)
        return NullContextVariable(source=self.source)
    elif KeyedJaggedTensorVariable.is_matching_object(value):
        self.install_guards(GuardBuilder.TYPE_MATCH)
        result = KeyedJaggedTensorVariable(value, source=self.source)
        # TODO: this doing it manually is bad
        return self.tx.output.side_effects.track_object_existing(value, result)
    elif isinstance(value, torch.optim.Optimizer):
        self.install_guards(GuardBuilder.TYPE_MATCH)
        return OptimizerVariable(value, source=self.source)
    elif ProcessGroupVariable.is_process_group(value):
        self.install_guards(GuardBuilder.ID_MATCH)
        return ProcessGroupVariable(value, source=self.source)
    elif DeviceMeshVariable.is_device_mesh(value):
        # TODO: see if we need to add custom guard instead of a simple ID_MATCH
        self.install_guards(GuardBuilder.ID_MATCH)
        return DeviceMeshVariable(value, source=self.source)
    elif PlacementClassVariable.is_placement_type(value):
        # TODO: see if we need to add custom guard instead of a simple ID_MATCH
        self.install_guards(GuardBuilder.ID_MATCH)
        return PlacementClassVariable(value, source=self.source)
    elif PlacementVariable.is_placement(value):
        # TODO: see if we need to add custom guard instead of a simple ID_MATCH
        self.install_guards(GuardBuilder.ID_MATCH)
        return PlacementVariable(
            value,
            source=self.source,
        )
    elif istype(value, type) and value in itertools.__dict__.values():
        self.install_guards(GuardBuilder.FUNCTION_MATCH)
        return ItertoolsVariable(value, source=self.source)
    elif isinstance(value, torch.SymBool):
        # Note: the idea here is to re-use the infra we've built for SymInt by simulating the
        # user provided SymBool with a SymInt in dynamo.

        # Concretely,
        # 1. We create a SymInt in dynamo's shape_env, whose source is constructed as ConvertIntSource(self.source).
        # so that guards on the SymInts can be effectively applied on the original SymBool in user program.
        # 2. We create a SymBool based on the SymInt in dynamo's ShapeEnv. Because the original user program
        # depends on the value being a SymBool. This allows dynamo to interpret the user's program correctly.

        value_hint = value.node.require_hint()
        new_source = ConvertIntSource(self.source)

        new_symint = self.tx.output.shape_env.create_unspecified_symint_and_symbol(
            int(value_hint),
            new_source,
            dynamic_dim=DimDynamic.DYNAMIC,
        )

        sym_node_proxy = self.tx.output.root_tracer.create_graph_input(
            re.sub(r"[^a-zA-Z0-9]+", "_", self.name),
            type(new_symint),
            source=new_source,
        )

        sym_node_proxy.node.meta["grapharg"] = GraphArg(
            new_source,
            new_symint,
            False,
            None,
            is_tensor=False,
            example_strong_ref=new_symint,
        )
        self.tx.output.bound_symbols.add(new_symint.node.expr)
        self.tx.output.tracked_fakes.append(TrackedFake(new_symint, new_source, None))
        return SymNodeVariable(
            sym_node_proxy,
            new_symint == 1,
        )
    elif isinstance(value, (JITFunction, Autotuner)):
        self.install_guards(GuardBuilder.ID_MATCH)
        return TritonKernelVariable(
            value,
            None,  # No kernel idx provided
            None,  # No grid provided
            source=self.source,
        )
    elif isinstance(value, torch.amp.autocast_mode.autocast):
        self.install_guards(GuardBuilder.ID_MATCH)
        return AutocastModeVariable(
            target_values=[
                value.device,
                value.fast_dtype,
                value._enabled,
                value._cache_enabled,
            ],
            source=self.source,
        )
    elif TorchCtxManagerClassVariable.is_matching_cls(value):
        self.install_guards(GuardBuilder.FUNCTION_MATCH)
        return TorchCtxManagerClassVariable(value, source=self.source)
    elif is_function_or_wrapper(value):
        value, attr_name = unwrap_with_attr_name_if_wrapper(value)
        # For these wrappers, Dynamo points to the wrapped function,
        # so source needs to be updated as well.
        if attr_name is not None:
            self.source = AttrSource(self.source, attr_name)
        return trace_rules.lookup(value).create_with_source(value, source=self.source)
    # Don't use istype, since some python modules are not subclasses of types.ModuleType directly.
    # E.g, type(torch.ops) -> <class 'torch._ops._Ops'>,
    # type(torch.backends.cudnn) -> <class 'torch.backends.cudnn.CudnnModule'>
    elif isinstance(value, (types.ModuleType, replay_record.DummyModule)):
        self.install_guards(GuardBuilder.FUNCTION_MATCH)
        return PythonModuleVariable(
            value,
            source=self.source,
        )
    elif isinstance(value, types.MethodType) and isinstance(
        value.__self__, (torch.nn.Module, torch.utils._pytree.TreeSpec)
    ):
        # don't let MethodTypes fall through to UserDefinedObject,
        # which doesn't support 'CALL_FUNCTION'

        # TODO(whc): Why do we limit this to methods on NNModules?
        # I don't have a good reason for this, but it preserves the existing behavior
        # for MBartForConditionalGeneration, which generates many graph breaks and OOMs otherwise.
        # I suspect we probably want to relax this check and dig deeper there.

        # In order to construct a MethodVariable in Dynamo, we start with an actual method obj from python,
        # but need to separately wrap its underlying `__func__` and its `self` argument.  We wrap `self` here
        # and then `__func__` gets wrapped inside UserMethodVariable.
        self_obj = VariableBuilder(self.tx, source=AttrSource(self.source, "__self__"))(
            value.__self__
        )
        assert self_obj and isinstance(
            self_obj, VariableTracker
        ), "Failed to produce a valid self obj"
        self.install_guards(GuardBuilder.FUNCTION_MATCH)
        return UserMethodVariable(
            value.__func__,
            self_obj,
            source=self.source,
        )
    elif isinstance(value, types.GetSetDescriptorType):
        self.install_guards(GuardBuilder.FUNCTION_MATCH)
        return GetSetDescriptorVariable(value)
    elif isinstance(value, types.MethodWrapperType):
        self.install_guards(GuardBuilder.FUNCTION_MATCH)
        return MethodWrapperVariable(value)
    elif issubclass(type(value), type):
        if value in (torch.utils.hooks.BackwardHook, torch.nn.Parameter):
            # TODO(jansel): combine this case with the one above
            return trace_rules.lookup(value).create_with_source(
                value, source=self.source
            )
        if value is torch.autograd._unsafe_preserve_version_counter:
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return PreserveVersionContextVariable.constructor(self.tx)
        # This is a userdefined class, so install an ID_MATCH even if its a
        # global variable.
        self.install_guards(GuardBuilder.ID_MATCH)
        return UserDefinedClassVariable(
            value,
            source=self.source,
        )
    elif RestrictedListSubclassVariable.is_matching_cls(type(value)):
        self.install_guards(GuardBuilder.SEQUENCE_LENGTH)
        return self.set_source_and_track_mutable(
            value,
            RestrictedListSubclassVariable(
                [
                    LazyVariableTracker.create(
                        value=value[i], source=GetItemSource(self.source, i)
                    )
                    for i in range(len(value))
                ],
                user_cls=type(value),
                user_cls_source=AttrSource(self.source, "__class__"),
            ),
        )
    else:
        self.install_guards(GuardBuilder.TYPE_MATCH)
        result = UserDefinedObjectVariable(value, source=self.source)
        if not SideEffects.cls_supports_mutation_side_effects(type(value)):
            # don't allow STORE_ATTR mutation with custom __setattr__
            return result
        return self.tx.output.side_effects.track_object_existing(value, result)


VariableBuilder._wrap = _wrap


def wrap_fx_proxy_cls(
    target_cls, tx, proxy, example_value=None, subclass_type=None, **options
):
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

    assert isinstance(tx, InstructionTranslatorBase)
    if "guards" in options and options["guards"] is not None:
        tx.output.guards.update(options["guards"])

    assert "example_value" not in proxy.node.meta, f"{proxy.node.meta['example_value']}"

    initial_example_value = example_value

    def _clone_input(value):
        if isinstance(value, torch.Tensor):
            # tensor subclasses will not be converted to FakeTensors and need to be cloned
            if not (
                isinstance(value, FakeTensor)
                or (
                    # Is functional tensor fakeified by this instance of Dynamo
                    torch._is_functional_tensor(value)
                    and maybe_get_fake_mode(value) is tx.fake_mode
                )
                or value.is_nested
            ):
                # NB: ensure strides are preserved
                value = clone_input(value)

        return value

    with preserve_rng_state():
        if example_value is None:
            # only allow_non_graph_fake in this instance because we handle the non-fake
            # cases properly below.
            example_value = get_fake_value(proxy.node, tx, allow_non_graph_fake=True)

        # Handle recursive calls here
        elif maybe_get_fake_mode(example_value) is tx.fake_mode:
            pass

        elif isinstance(example_value, torch.Tensor):
            if tx.export:
                # The legacy behavior for real value cache with subclasses was
                # to perform a clone WITHOUT preserving the subclass.  It's
                # not entirely clear this is what you actually want though.
                with torch._C.DisableTorchFunctionSubclass():
                    proxy.tracer.real_value_cache[proxy.node] = _clone_input(
                        example_value
                    )
            # NB: If we're ignoring subclass, then the expectation is you will
            # take the returned TensorVariable and wrap it into a more
            # accurate TensorVariable that is able to track subclass-ness;
            # otherwise this is wrong!
            kwargs = {
                "is_tensor": target_cls
                in (TensorVariable, TensorWithTFOverrideVariable),
            }
            assert "source" in options and options["source"] is not None
            kwargs["source"] = options["source"]
            example_value = wrap_to_fake_tensor_and_record(
                example_value, tx=tx, **kwargs
            )
        if isinstance(example_value, torch.Tensor) and (
            maybe_get_fake_mode(example_value) is not tx.fake_mode
        ):
            raise InternalTorchDynamoError(
                "`example_value` needs to be a `FakeTensor`"
                f"wrapped by this instance of Dynamo. Found: {example_value}"
            )

    if isinstance(example_value, torch.Tensor):
        is_parameter = isinstance(example_value, torch.nn.Parameter)

        # NB: In most (all?) cases, this does not actually do a clone.
        # (WARNING: this means that if we mutate metadata on the fake
        # tensor, the stored example value will update too!)
        example_value = _clone_input(example_value)
        proxy.node.meta["example_value"] = example_value
        specialized_props = target_cls.specialize(example_value)
        # TODO: not sure about this fake mode test
        if (
            isinstance(example_value, torch._subclasses.fake_tensor.FakeTensor)
            and example_value.fake_mode is tx.fake_mode
        ):
            tensor_type = subclass_type if subclass_type else torch.Tensor
            specialized_props["class_type"] = (
                torch.nn.Parameter if is_parameter else tensor_type
            )

        options.update(specialized_props)
        return target_cls(proxy, **options)
    elif (
        hasattr(proxy.node.target, "__name__")
        and proxy.node.target.__name__ == "set_state"
        and isinstance(proxy.node.target.__self__, torch._C.Generator)
        or proxy.node.target == torch.random.set_rng_state
    ):
        return TorchInGraphFunctionVariable(proxy.node.target)
    elif (
        proxy.node.target == torch._C._DisableFuncTorch
        or proxy.node.target == torch.mlu._is_in_bad_fork
    ):
        return UserDefinedObjectVariable(example_value)
    elif istype(example_value, torch.Size) and all(
        isinstance(x, int) for x in example_value
    ):
        sizes = [ConstantVariable.create(x) for x in example_value]
        return SizeVariable(sizes, **options)
    elif isinstance(example_value, (tuple, list)):
        proxy.node.meta["example_value"] = example_value
        unpacked = []
        for i, val in enumerate(example_value):
            if val is None:
                # nn.MultiheadAttention() can return None, see issue #175
                unpacked.append(
                    ConstantVariable.create(None, **options),
                )
            else:
                unpacked.append(
                    wrap_fx_proxy_cls(
                        target_cls,
                        tx,
                        proxy.tracer.create_proxy(
                            "call_function", operator.getitem, (proxy, i), {}
                        ),
                        example_value=val,
                        **options,
                    )
                )
        if isinstance(example_value, torch.Size):
            # NB: Keep the old proxy around.  See SizeVariable for an
            # explanation why
            return SizeVariable(unpacked, proxy, **options)
        elif istype(example_value, tuple):
            return TupleVariable(unpacked, **options)
        elif istype(example_value, (list, immutable_list)):
            return ListVariable(unpacked, mutable_local=MutableLocal(), **options)
        else:
            assert example_value.__class__.__module__ == "torch.return_types" or hasattr(
                example_value, "_fields"
            ), f"expected {example_value.__class__.__module__} == torch.return_types or named tuple but got {type(example_value)}"
            return NamedTupleVariable(unpacked, example_value.__class__, **options)
    elif example_value is None or proxy.node.target is torch.manual_seed:
        return ConstantVariable.create(None, **options)
    elif isinstance(example_value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        proxy.node.meta["example_value"] = example_value
        return SymNodeVariable(proxy, example_value, **options)
    elif (
        inspect.isclass(proxy.node.target)
        and issubclass(proxy.node.target, _StreamBase)
    ) or proxy.node.target in [
        device_interface.current_stream
        for _, device_interface in get_registered_device_interfaces()
    ]:
        proxy.node.meta["example_value"] = example_value
        return StreamVariable(proxy, example_value, example_value.device, **options)
    elif (
        inspect.isclass(proxy.node.target) and issubclass(proxy.node.target, _EventBase)
    ) or proxy.node.target in [
        device_interface.Event
        for _, device_interface in get_registered_device_interfaces()
    ]:
        proxy.node.meta["example_value"] = example_value
        return EventVariable(proxy, example_value, **options)
    elif proxy.node.target == "query" and proxy.node.op == "call_method":
        proxy.node.meta["example_value"] = example_value
        return ConstantVariable(example_value, **options)
    elif (
        example_value is not None
        and isinstance(example_value, _EventBase)
        and proxy.node.target == "record_event"
        and proxy.node.op == "call_method"
    ):
        proxy.node.meta["example_value"] = example_value
        return EventVariable(proxy, example_value, **options)
    elif isinstance(example_value, int) and proxy.node.target in [
        torch.sym_int,
        getattr,
        operator.getitem,
        torch._utils._element_size,
        torch.seed,
        operator.mod,
        torch._C._functorch._vmap_increment_nesting,
        torch._C._functorch._vmap_decrement_nesting,
        torch._functorch.vmap._validate_and_get_batch_size,
        torch._C._functorch._grad_increment_nesting,
        torch._C._functorch._grad_decrement_nesting,
        # some mac builds are missing torch.distributed.get_rank()
        getattr(torch.distributed, "get_rank", _missing),
        getattr(torch.distributed, "get_world_size", _missing),
        # This always wants to be in the graph, even if the constraint
        # results in a constant int
        torch._constrain_as_value,
        torch._constrain_as_size,
    ]:
        proxy.node.meta["example_value"] = example_value
        return ConstantVariable.create(example_value, **options)
    elif isinstance(example_value, torch_mlu.backends.mlu.SDPAParams):
        from torch._dynamo.variables.sdpa import SDPAParamsVariable

        proxy.node.meta["example_value"] = example_value
        return SDPAParamsVariable(proxy, **options)
    elif isinstance(example_value, bool) and proxy.node.target in [
        torch_mlu.backends.mlu.can_use_flash_attention,
        torch_mlu.backends.mlu.can_use_efficient_attention,
    ]:
        proxy.node.meta["example_value"] = example_value
        return ConstantVariable.create(example_value, **options)
    else:
        unimplemented(
            "torch.* op returned non-Tensor "
            + f"{typestr(example_value)} {proxy.node.op} {proxy.node.target}"
        )


torch._dynamo.variables.builder.wrap_fx_proxy_cls = wrap_fx_proxy_cls
