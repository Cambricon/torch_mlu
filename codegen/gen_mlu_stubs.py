import argparse
import os
import yaml
import json
from typing import Dict, List, Sequence, Union, Tuple

import torchgen
from torchgen.gen import (
    get_grouped_native_functions,
    parse_native_yaml,
    parse_tags_yaml,
    RegisterSchema,
    LineLoader,
)
from torchgen.model import (
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
    Location,
    BackendIndex,
    DispatchKey,
)
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import (
    concatMap,
    context,
    FileManager,
    assert_never,
    mapMaybe,
)

from codegen.dest import (
    RegisterMLU,
    GenExternalMLU,
    Target,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate backend stub files")
    parser.add_argument(
        "-s",
        "--source_yaml",
        help="path to source yaml file containing operator external definitions",
    )
    parser.add_argument("-o", "--output_dir", help="output directory")
    parser.add_argument("--dry_run", type=bool, default=False,
                        help="output directory")
    # We have different operator computing libraries(CNNL, BANG, MLUOPS).
    # CNNL is used by default, and BANG or MLUOPS is an optional backend.
    parser.add_argument(
        '--use_bang', action='store_true',
        help='select whether to generate the bang operator.')
    parser.add_argument(
        '--use_mluop', action='store_true',
        help='select whether to generate the mlu-ops operator.')
    options = parser.parse_args()

    run(options)


def parse_mlu_aten_yaml(
    path: str,
    backend_indices: Dict[DispatchKey, BackendIndex],
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    options
) -> Tuple[Dict[OperatorName, Dict[str, object]], Sequence[Union[NativeFunction, NativeFunctionsGroup]]]:
    with open(path, 'r') as f:
        yaml_values = yaml.load(f, Loader=LineLoader)

    # record extra metadata from mlu_functions.yaml like derived_type
    aux: Dict[OperatorName, Dict[str, object]] = {}

    supported_aten = yaml_values.pop("aten", [])
    if supported_aten is None:
        supported_aten = []
    assert isinstance(
        supported_aten, list
    ), f'expected "aten" to be a list, but got: {supported_aten}'
    for op in supported_aten:
        assert isinstance(
            op, dict
        ), f'expected op msg to be a dict, but got: {op}'
        assert 'func' in op.keys(), f'You must provide the keyword func for aten ops'

        op_name = OperatorName.parse(op['func'].split('(')[0])
        d: Dict[str, object] = {}
        d['structured'] = True if 'structured_delegate' in op.keys(
        ) or 'structured' in op.keys() else False

        if 'override_meta' in op.keys():
            d['override_meta'] = True
        if 'override_impl' in op.keys():
            d['override_impl'] = True

        if 'custom_autograd' in op.keys():
            d['custom_autograd'] = True

        d['derived_type'] = op['derived_type'] if 'derived_type' in op.keys() else 'cnnl'
        assert d['derived_type'] in ['cnnl', 'bang', 'mluop'], \
            f"derived_type support cnnl, bang or mluop, but got: {d['derived_type']}"
        if d['derived_type'] == 'bang' and options.use_bang:
            d['use_bang'] = True
        if d['derived_type'] == 'mluop' and options.use_mluop:
            d['use_mluop'] = True
        d['ns'] = 'aten'
        aux[op_name] = d

    # add selected NativeFunction or NativeFunctionsGroup
    f: Sequence[Union[NativeFunction, NativeFunctionsGroup]] = []
    for g in grouped_native_functions:
        if isinstance(g, NativeFunction):
            if g.func.name in aux:
                f.append(g)
        elif isinstance(g, NativeFunctionsGroup):
            if g.functional.func.name in aux or \
                (g.inplace and g.inplace.func.name in aux) or \
                    g.out.func.name in aux:
                f.append(g)

                if g.out.structured:
                    op_name = g.out.func.name
                    metadata = backend_indices[DispatchKey.CPU].get_kernel(g)
                    if metadata.kernel == backend_indices[DispatchKey.CUDA].get_kernel(g).kernel:
                        aux[op_name]['metadata'] = metadata
        else:
            assert_never(g)

    return aux, f


def parse_mlu_custom_yaml(
    source_yaml_path: str,
    tags_yaml_path: str,
    key: str,
    options
) -> Tuple[Dict[OperatorName, Dict[str, object]], Sequence[Union[NativeFunction, NativeFunctionsGroup]]]:
    with open(source_yaml_path, 'r') as f:
        yaml_values = yaml.load(f, Loader=LineLoader)
    valid_tags = parse_tags_yaml(tags_yaml_path)

    # record extra metadata from mlu_functions.yaml like derived_type
    aux: Dict[OperatorName, Dict[str, object]] = {}
    supported = yaml_values.pop(key, [])
    if supported is None:
        supported = []
    assert isinstance(
        supported, list
    ), f'expected "{key}" to be a list, but got: {supported}'

    rs: List[NativeFunction] = []
    for e in supported:
        d: Dict[str, object] = {}
        assert isinstance(
            e, dict
        ), f'expected op msg to be a dict, but got: {e}'
        assert 'func' in e.keys(), f'You must provide the keyword func for {key} ops'
        assert len(e['func'].split('(')) > 1, f'You must provide concrete function schema for {key} ops'

        op_name = OperatorName.parse(e['func'].split('(')[0])
        if 'derived_type' in e.keys():
            d['derived_type'] = e['derived_type']
            e.pop('derived_type')
            if d['derived_type'] == 'bang' and options.use_bang:
                d['use_bang'] = True
            if d['derived_type'] == 'mluop' and options.use_mluop:
                d['use_mluop'] = True
        else:
            d['derived_type'] = 'cnnl'

        assert d['derived_type'] in ['cnnl', 'bang', 'mluop'], \
            f"derived_type support cnnl, bang or mluop, but got: {d['derived_type']}"

        if 'custom_autograd' in e.keys():
            d['custom_autograd'] = True
            e.pop('custom_autograd')

        aux[op_name] = d
        assert isinstance(e.get('__line__'), int), e
        loc = Location(source_yaml_path, e['__line__'])
        funcs = e.get('func')
        with context(f'in {loc}:\n  {funcs}'):
            rs.append(NativeFunction.from_yaml(e, loc, valid_tags)[0])

    return aux, rs


def version_collect(path: str) -> List[Dict[str, object]]:
    json_file = os.path.join(path, 'build.property')
    with open(json_file) as fp:
        json_dict = json.load(fp)
    versions = json_dict['build_requires']
    versions['driver'] = json_dict['driver_requires']
    version_collects: List[Dict[str, object]] = []
    version_torch_mlu: Dict[str, object] = {}
    version_torch_mlu['package_name'] = 'torch_mlu'
    version_torch_mlu['version_num'] = json_dict['version']
    version_collects.append(version_torch_mlu)
    for key in versions:
        version_info: Dict[str, object] = {}
        version_info['package_name'] = key + '_required_str'
        version_info['version_num'] = versions[key][1]
        version_collects.append(version_info)
    return version_collects


def gen_version_info(version_collect: Dict[str, object]) -> str:
    return f"""\
std::string {version_collect['package_name']} = "{version_collect['version_num']}";
"""


def run(options) -> None:
    cur_path = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(cur_path, "templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(install_dir=install_dir, template_dir=template_dir, dry_run=options.dry_run)

    torchgen_path = os.path.dirname(os.path.abspath(torchgen.__file__))
    native_yaml_path = os.path.join(torchgen_path, 'packaged/ATen/native/native_functions.yaml')
    tags_yaml_path = os.path.join(torchgen_path, "packaged/ATen/native/tags.yaml")
    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path)

    native_functions = parsed_yaml.native_functions
    # There is a trick here. Generally, backend_indices are not needed
    # because we only register MLU or AutogradMLU. But for structured kernel,
    # we still need CPU/CUDA metadata, we want to reuse some structs generated in Pytorch.
    backend_indices = parsed_yaml.backend_indices
    grouped_native_functions = get_grouped_native_functions(native_functions)

    # parse aten functions in mlu_functions.yaml
    aux_aten, aten_functions = parse_mlu_aten_yaml(
        options.source_yaml, backend_indices, grouped_native_functions, options)

    # parse vision/audio/custom functions in mlu_functions.yaml
    aux_vision, vision_functions = parse_mlu_custom_yaml(
        options.source_yaml, tags_yaml_path, 'vision', options)
    aux_audio, audio_functions = parse_mlu_custom_yaml(
        options.source_yaml, tags_yaml_path, 'audio', options)
    aux_custom, custom_functions = parse_mlu_custom_yaml(
        options.source_yaml, tags_yaml_path, 'custom', options)

    functions = aten_functions + vision_functions + \
        custom_functions + audio_functions
    aux = {
        **aux_aten,
        **aux_vision,
        **aux_audio,
        **aux_custom,
    }
    selector = SelectiveBuilder.get_nop_selector()

    generated_comment = 'Autogenerated file by gen_mlu_stubs.py. Do not edit directly!'
    install_dir = os.path.join(
        cur_path, '../torch_mlu/csrc/aten/generated/')
    if not os.path.exists(install_dir):
        os.makedirs(install_dir)
    fm = make_file_manager(install_dir)

    # We generate symint API by default.
    fm.write('RegisterMLU.cpp', lambda: {
        'generated_comment': generated_comment,
        'dispatch_anonymous_definitions': list(concatMap(
            RegisterMLU(Target.ANONYMOUS_DEFINITION,
                        selector, aux, symint=True), functions
        )),
        'dispatch_aten_registrations': list(concatMap(
            RegisterMLU(Target.REGISTRATION, selector,
                        aux, symint=True), aten_functions
        )),
        'dispatch_aten_autograd_registrations': list(concatMap(
            RegisterMLU(Target.AUTOGRAD_REGISTRATION, selector,
                        aux, symint=True), aten_functions
        )),
        'dispatch_vision_registrations': list(concatMap(
            RegisterMLU(Target.REGISTRATION, selector,
                        aux, symint=True), vision_functions
        )),
        'dispatch_audio_registrations': list(concatMap(
            RegisterMLU(Target.REGISTRATION, selector,
                        aux, symint=True), audio_functions
        )),
        'dispatch_custom_registrations': list(concatMap(
            RegisterMLU(Target.REGISTRATION, selector,
                        aux, symint=True), custom_functions
        )),
        'dispatch_custom_autograd_registrations': list(concatMap(
            RegisterMLU(Target.AUTOGRAD_REGISTRATION, selector,
                        aux, symint=True), custom_functions
        )),
        'custom_schema_registrations': list(mapMaybe(
            RegisterSchema(selector), custom_functions
        )),
        'dispatch_namespaced_definitions': list(concatMap(
            RegisterMLU(Target.NAMESPACED_DEFINITION,
                        selector, aux, symint=True), functions
        )),
    })

    fm.write('MLUFunctions.h', lambda: {
        'generated_comment': generated_comment,
        'dispatch_namespaced_declarations': list(concatMap(
            RegisterMLU(Target.NAMESPACED_DECLARATION,
                        selector, aux, symint=True), functions
        )),
    })

    for target in [
        Target.CNNL_KERNEL_DECLARATION,
        Target.BANG_KERNEL_DECLARATION,
        Target.MLUOP_KERNEL_DECLARATION,
    ]:
        if target == Target.CNNL_KERNEL_DECLARATION:
            derived_type = 'cnnl'
        if target == Target.BANG_KERNEL_DECLARATION:
            derived_type = 'bang'
        if target == Target.MLUOP_KERNEL_DECLARATION:
            derived_type = 'mluop'
        install_dir = os.path.join(
            cur_path, f"../torch_mlu/csrc/aten/operators/{derived_type}/")
        fm = make_file_manager(install_dir)
        fm.write_with_template(
            f"{derived_type}_kernel.h",
            'KernelFunctions.h', lambda: {
                'generated_comment': generated_comment,
                'dispatch_kernel_declarations': list(concatMap(
                    GenExternalMLU(target, selector, aux), functions
                )),
            })

    version_dir = os.path.join(cur_path, '../scripts/release/')
    version_collects = version_collect(version_dir)
    install_dir = os.path.join(cur_path, '../torch_mlu/csrc/utils/')
    fm = make_file_manager(install_dir)
    fm.write('version.cpp', lambda: {
        'generated_comment': generated_comment,
        'version_info': list(mapMaybe(
            gen_version_info, version_collects
        )),
    })


if __name__ == '__main__':
    main()
