from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import os
import setuptools
import sys
import re
import shlex
import warnings

import torch
from setuptools.command.build_ext import build_ext
from torch.utils import cpp_extension
from typing import List, Optional
import subprocess

COMMON_BANG_CFLAGS = [
    "--bang-mlu-arch=mtp_372",
    "--bang-mlu-arch=mtp_592",
]

COMMON_FLAGS = [
    "-O3",
    "-Wall",
    "-fPIC",
    "-std=c++17",
    "-pthread",
]


def _find_neuware_home() -> Optional[str]:
    r"""Finds the NEUWARE_HOME install path."""
    # Guess #1
    neuware_home = os.environ.get("NEUWARE_HOME")
    if neuware_home is None:
        # Guess #2
        try:
            with open(os.devnull, "w") as devnull:
                cncc = (
                    subprocess.check_output(["which", "cncc"], stderr=devnull)
                    .decode()
                    .rstrip("\r\n")
                )
                neuware_home = os.path.dirname(os.path.dirname(cncc))
        except Exception:
            # Guess #3
            neuware_home = "/usr/local/neuware_home"
            if not os.path.exists(neuware_home):
                neuware_home = None
    if neuware_home and not torch.mlu.is_available():
        print(f"No NEUWARE runtime is found, using neuware_home='{neuware_home}'")
    return neuware_home


def _is_mlu_file(path: str) -> bool:
    valid_ext = [".mlu"]
    return os.path.splitext(path)[1] in valid_ext


def _join_neuware_home(*paths) -> str:
    r"""
    Joins paths with NEUWARE_HOME, or raises an error if NEUWARE_HOME is not set.

    This is basically a lazy way of raising an error for missing $NEUWARE_HOME
    only once we need to get any MLU-specific path.
    """
    NEUWARE_HOME = _find_neuware_home()
    if NEUWARE_HOME is None:
        raise EnvironmentError(
            "NEUWARE_HOME environment variable is not set. "
            "Please set it to your Cambricon Neuware install root."
        )
    return os.path.join(NEUWARE_HOME, *paths)


def include_paths() -> List[str]:
    """
    Get the include paths required to build a C++ or BANG extension.

    Returns:
        A list of include path strings.
    """
    # Import from Installed Pytorch
    paths = torch.utils.cpp_extension.include_paths(False)
    # Import from Cambricon Catch
    cur_path = os.path.abspath(__file__)
    torch_mlu_path = os.path.dirname(os.path.dirname(cur_path))
    paths.append(os.path.join(torch_mlu_path, "csrc"))
    paths.append(os.path.join(torch_mlu_path, "csrc", "api", "include", "torch_mlu"))
    # Import from Cambricon Neuware
    neuware_home_include = _join_neuware_home("include")
    paths.append(neuware_home_include)
    return paths


def library_paths() -> List[str]:
    r"""
    Get the library paths required to build a C++ or BANG extension.

    Returns:
        A list of library path strings.
    """
    paths = []

    # Import from Installed Pytorch
    paths = torch.utils.cpp_extension.library_paths(False)
    # Import from Cambricon Catch
    cur_path = os.path.abspath(__file__)
    torch_mlu_path = os.path.dirname(os.path.dirname(cur_path))
    lib_path = os.path.join(torch_mlu_path, "csrc/lib")
    paths.append(lib_path)
    # Import from Cambricon Neuware
    lib_dir = "lib64"
    paths.append(_join_neuware_home(lib_dir))
    return paths


def _write_ninja_file_and_compile_objects(
    sources: List[str],
    objects,
    cflags,
    post_cflags,
    bang_cflags,
    bang_post_cflags,
    build_directory: str,
    verbose: bool,
    with_bang: Optional[bool],
) -> None:
    torch.utils.cpp_extension.verify_ninja_availability()
    compiler = os.environ.get("CXX", "c++")
    torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version(compiler)
    if with_bang is None:
        with_bang = any(map(_is_mlu_file, sources))
    build_file_path = os.path.join(build_directory, "build.ninja")
    if verbose:
        print(f"Emitting ninja build file {build_file_path}...", file=sys.stderr)
    _write_ninja_file(
        path=build_file_path,
        cflags=cflags,
        post_cflags=post_cflags,
        bang_cflags=bang_cflags,
        bang_post_cflags=bang_post_cflags,
        sources=sources,
        objects=objects,
        ldflags=None,
        library_target=None,
        with_bang=with_bang,
    )
    if verbose:
        print("Compiling objects...", file=sys.stderr)
    torch.utils.cpp_extension._run_ninja_build(
        build_directory,
        verbose,
        # It would be better if we could tell users the name of the extension
        # that failed to build but there isn't a good way to get it here.
        error_prefix="Error compiling objects for extension",
    )


def _write_ninja_file(
    path,
    cflags,
    post_cflags,
    bang_cflags,
    bang_post_cflags,
    sources,
    objects,
    ldflags,
    library_target,
    with_bang,
) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `bang_cflags`: list of flags to pass to $cncc. Can be None.
    `bang_postflags`: list of flags to append to the $cncc invocation. Can be None.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    `with_bang`: If we should be compiling with BANG.
    """

    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    bang_cflags = sanitize_flags(bang_cflags)
    bang_post_cflags = sanitize_flags(bang_post_cflags)
    ldflags = sanitize_flags(ldflags)

    # Sanity checks...
    assert len(sources) == len(objects)
    assert len(sources) > 0

    compiler = os.environ.get("CXX", "c++")

    # Version 1.3 is required for the `deps` directive.
    config = ["ninja_required_version = 1.3"]
    config.append(f"cxx = {compiler}")
    if with_bang:
        cncc = _join_neuware_home("bin", "cncc")
        config.append(f"cncc = {cncc}")

    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')

    if with_bang:
        flags.append(f'bang_cflags = {" ".join(bang_cflags)}')
        flags.append(f'bang_post_cflags = {" ".join(bang_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ["rule compile"]
    compile_rule.append("  command = $cxx $cflags -c $in -o $out $post_cflags")
    compile_rule.append("  depfile = $out.d")
    compile_rule.append("  deps = gcc")

    if with_bang:
        bang_compile_rule = ["rule bang_compile"]
        bang_compile_rule.append(f"  command = $cncc -c $in -o $out $bang_post_cflags")

    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects):
        is_bang_source = _is_mlu_file(source_file) and with_bang
        rule = "bang_compile" if is_bang_source else "compile"
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f"build {object_file}: {rule} {source_file}")

    if library_target is not None:
        link_rule = ["rule link"]
        link_rule.append(
            "  command = $cxx -shared $in1 $in2 $ldflags -o $out -Wl,--as-needed"
        )
        link = [f'build {library_target}: link {" ".join(objects)}']
        default = [f"default {library_target}"]
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_bang:
        blocks.append(bang_compile_rule)
    blocks += [link_rule, build, link, default]
    with open(path, "w") as build_file:
        for block in blocks:
            lines = "\n".join(block)
            build_file.write(f"{lines}\n\n")


def MLUExtension(name, sources, *args, **kwargs):
    r"""
    Creates a :class:`setuptools.Extension` for BANG/C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a BANG/C++
    extension. This includes the mlu include path, library path and runtime
    library.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

    Example:
        >>> from setuptools import setup
        >>> from torch_mlu.utils.cpp_extension import BuildExtension, MLUExtension
        >>> setup(
                name='mlu_extension',
                ext_modules=[
                    MLUExtension(
                            name='mlu_extension',
                            sources=['extension.cpp', 'extension_kernel.mlu'],
                            extra_compile_args={'cxx': ['-g'],
                                                'cncc': ['-O2']})
                ],
                cmdclass={
                    'build_ext': BuildExtension
                })
    """
    library_dirs = kwargs.get("library_dirs", [])
    library_dirs += library_paths()
    kwargs["library_dirs"] = library_dirs

    libraries = kwargs.get("libraries", [])
    libraries.append("c10")
    libraries.append("torch")
    libraries.append("torch_cpu")
    libraries.append("torch_python")

    # Add cambricon catch library
    libraries.append("torch_mlu")

    kwargs["libraries"] = libraries

    include_dirs = kwargs.get("include_dirs", [])
    include_dirs += include_paths()
    kwargs["include_dirs"] = include_dirs

    kwargs["language"] = "c++"

    return setuptools.Extension(name, sources, *args, **kwargs)


def _get_bang_arch_flags(cflags: Optional[List[str]] = None) -> List[str]:
    r"""
    Determine Bang arch flags to use.

    For an arch, say "--bang-arch=compute_30(3.0)", the added compile flag will be
    "--bang-mlu-arch=mtp_372".

    See select_compute_arch.cmake for corresponding named and supported arches
    when building with CMake.
    """
    # If cflags is given, there may already be user-provided arch flags in it
    # (from `extra_compile_args`)
    if cflags is not None:
        for flag in cflags:
            if "--bang-mlu-arch" in flag or "--bang-arch" in flag:
                return []

    supported_arches = ["3.0", "5.0"]
    cc_to_arch = {cc: cc.replace(".", "") for cc in supported_arches}
    valid_arch_strings = COMMON_BANG_CFLAGS + [
        "--bang-arch=compute_{}".format(cc_to_arch[c]) for c in supported_arches
    ]
    # First check for an env var (same as used by the main setup.py)
    # Can be one or more architectures, e.g. "3.0" or "3.0;5.0;"
    _arch_list = os.environ.get("TORCH_BANG_ARCH_LIST", None)

    # If not given, determine what's best for the MLU/BANG version that can be found
    if not _arch_list:
        arch_list = []
        # the assumption is that the extension should run on any of the currently visible cards,
        # which could be of different types - therefore all archs for visible cards should be included.
        for i in range(torch.mlu.device_count()):
            capability = torch.mlu.get_device_capability(i)
            arch = f"{capability[0]}.{capability[1]}"
            if arch not in arch_list:
                arch_list.append(arch)
        arch_list = sorted(arch_list)
    else:
        # Deal with lists that are ' ' separated (only deal with ';' after)
        arch_list = [
            arch
            for arch in _arch_list.strip(" ").split(";")
            if arch in supported_arches
            and "--bang-arch=compute_{}".format(cc_to_arch[arch]) in valid_arch_strings
        ]

    flags = []
    for arch in arch_list:
        if not any(
            [
                len(re.findall(r"compute_{}".format(cc_to_arch[arch]), valid)) > 0
                for valid in valid_arch_strings
            ]
        ):
            raise ValueError(
                f"Unknown Bang arch (compute_{cc_to_arch[arch]}) or mlu not supported"
            )
        else:
            flags.append(f"--bang-arch=compute_{cc_to_arch[arch]}")

    return sorted(list(set(flags)))


class BuildExtension(build_ext, object):
    r"""
     A custom :mod:`setuptools` build extension.

     This :class:`setuptools.build_ext` subclass takes care of passing the
     minimum required compiler flags (e.g. ``-std=c++17``) as well as mixed
     C++/BANG compilation (and support for BANG files in general).

     When using :class:`BuildExtension`, it is allowed to supply a dictionary
     for ``extra_compile_args`` (rather than the usual list) that maps from
     languages (``cxx`` or ``cncc``) to a list of additional compiler flags to
     supply to the compiler. This makes it possible to supply different flags to
     the C++ and BANG compiler during mixed compilation.

    ``use_ninja`` (bool): If ``use_ninja`` is ``True`` (default), then we
     attempt to build using the Ninja backend. Ninja greatly speeds up
     compilation compared to the standard ``setuptools.build_ext``.
     Fallbacks to the standard distutils backend if Ninja is not available.

     .. note::
         By default, the Ninja backend uses #CPUS + 2 workers to build the
         extension. This may use up too many resources on some systems. One
         can control the number of workers by setting the `MAX_JOBS` environment
         variable to a non-negative number.

    """

    @classmethod
    def with_options(cls, **options):
        r"""
        Returns a subclass with alternative constructor that extends any original keyword
        arguments to the original constructor with the given options.
        """

        class cls_with_options(cls):
            def __init__(self, *args, **kwargs):
                kwargs.update(options)
                super().__init__(*args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs):
        super(BuildExtension, self).__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

        self.use_ninja = kwargs.get("use_ninja", True)
        if self.use_ninja:
            msg = (
                "Attempted to use ninja as the BuildExtension backend but "
                "{}. Falling back to using the slow distutils backend."
            )
            if not torch.utils.cpp_extension.is_ninja_available():
                warnings.warn(msg.format("we could not find ninja."))
                self.use_ninja = False

    def finalize_options(self) -> None:
        super().finalize_options()
        if self.use_ninja:
            self.force = True

    def build_extensions(self):
        _, _ = self._check_abi()
        for extension in self.extensions:
            # Ensure at least an empty list of flags for 'cxx' and 'cncc' when
            # extra_compile_args is a dict. Otherwise, default torch flags do
            # not get passed. Necessary when only one of 'cxx' and 'cncc' is
            # passed to extra_compile_args in MLUExtension, i.e.
            #   MLUExtension(..., extra_compile_args={'cxx': [...]})
            # or
            #   MLUExtension(..., extra_compile_args={'cncc': [...]})
            if isinstance(extension.extra_compile_args, dict):
                for ext in ["cxx", "cncc"]:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []

            self._add_compile_flag(extension, "-DTORCH_API_INCLUDE_EXTENSION_H")
            for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
                val = getattr(torch._C, f"_PYBIND11_{name}")
                if val is not None and not torch.utils.cpp_extension.IS_WINDOWS:
                    self._add_compile_flag(extension, f'-DPYBIND11_{name}="{val}"')

            self._define_torch_extension_name(extension)
            self._add_gnu_cpp_abi_flag(extension)

        # Register .mlu as valid source extensions.
        self.compiler.src_extensions += [".mlu"]
        # Save the original _compile method for later.
        original_compile = self.compiler._compile

        def append_std14_if_no_std_present(cflags) -> None:
            cpp_format_prefix = "-{}="
            cpp_flag_prefix = cpp_format_prefix.format("std")
            cpp_flag = cpp_flag_prefix + "c++17"
            if not any(flag.startswith(cpp_flag_prefix) for flag in cflags):
                cflags.append(cpp_flag)

        def unix_mlu_flags(cflags):
            cflags = COMMON_FLAGS + cflags + _get_bang_arch_flags(cflags)
            return cflags

        def convert_to_absolute_paths_inplace(paths):
            # Helper function. See Note [Absolute include_dirs].
            if paths is not None:
                for i in range(len(paths)):
                    if not os.path.isabs(paths[i]):
                        paths[i] = os.path.abspath(paths[i])

        def unix_wrap_single_compile(
            obj, src, ext, cc_args, extra_postargs, pp_opts
        ) -> None:
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_mlu_file(src):
                    cncc = _join_neuware_home("bin", "cncc")
                    if not isinstance(cncc, list):
                        cncc = [cncc]
                    self.compiler.set_executable("compiler_so", cncc)
                    if isinstance(cflags, dict):
                        cflags = cflags["cncc"]

                    cflags = unix_mlu_flags(cflags)
                    cc_args = ["-I" + _join_neuware_home("include")]
                    pp_opts = cc_args
                    cc_args.append("-c")
                elif isinstance(cflags, dict):
                    cflags = cflags["cxx"]

                append_std14_if_no_std_present(cflags)

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable("compiler_so", original_compiler)

        def unix_wrap_ninja_compile(
            sources,
            output_dir=None,
            macros=None,
            include_dirs=None,
            debug=0,
            extra_preargs=None,
            extra_postargs=None,
            depends=None,
        ):
            r"""Compiles sources by outputting a ninja file and running it."""

            # Use absolute path for output_dir so that the object file paths
            # (`objects`) get generated with absolute paths.
            output_dir = os.path.abspath(output_dir)

            # See Note [Absolute include_dirs]
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)

            _, objects, extra_postargs, pp_opts, _ = self.compiler._setup_compile(
                output_dir, macros, include_dirs, sources, depends, extra_postargs
            )
            common_cflags = self.compiler._get_cc_args(pp_opts, debug, extra_preargs)
            extra_cc_cflags = self.compiler.compiler_so[1:]
            with_bang = any(map(_is_mlu_file, sources))

            # extra_postargs can be either:
            # - a dict mapping cxx/nvcc to extra flags
            # - a list of extra flags.
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs["cxx"]
            else:
                post_cflags = list(extra_postargs)
            append_std14_if_no_std_present(post_cflags)

            bang_post_cflags = None
            bang_cflags = None
            if with_bang:
                bang_cflags = common_cflags
                if isinstance(extra_postargs, dict):
                    bang_post_cflags = extra_postargs["cncc"]
                else:
                    bang_post_cflags = list(extra_postargs)
                bang_post_cflags = unix_mlu_flags(bang_post_cflags)
                append_std14_if_no_std_present(bang_post_cflags)
                bang_cflags = [shlex.quote(f) for f in bang_cflags]
                bang_post_cflags = [shlex.quote(f) for f in bang_post_cflags]

            _write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags],
                post_cflags=[shlex.quote(f) for f in post_cflags],
                bang_cflags=bang_cflags,
                bang_post_cflags=bang_post_cflags,
                build_directory=output_dir,
                verbose=True,
                with_bang=with_bang,
            )

            # Return *all* object filenames, not just the ones we just built.
            return objects

        # Monkey-patch the _compile or compile method.
        if self.use_ninja:
            self.compiler.compile = unix_wrap_ninja_compile
        else:
            self.compiler._compile = unix_wrap_single_compile

        build_ext.build_extensions(self)

    def _check_abi(self):
        # On some platforms, like Windows, compiler_cxx is not available.
        if hasattr(self.compiler, "compiler_cxx"):
            compiler = self.compiler.compiler_cxx[0]
        else:
            compiler = os.environ.get("CXX", "c++")
        (
            _,
            version,
        ) = torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version(
            compiler
        )
        return compiler, version

    def _add_compile_flag(self, extension, flag):
        extension.extra_compile_args = copy.deepcopy(extension.extra_compile_args)
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(flag)
        else:
            extension.extra_compile_args.append(flag)

    def _define_torch_extension_name(self, extension):
        # pybind11 doesn't support dots in the names
        # so in order to support extensions in the packages
        # like torch._C, we take the last part of the string
        # as the library name.
        names = extension.name.split(".")
        name = names[-1]
        define = "-DTORCH_EXTENSION_NAME={}".format(name)
        self._add_compile_flag(extension, define)

    def _add_gnu_cpp_abi_flag(self, extension):
        # Use the same CXX ABI as what PyTorch was compiled with.
        self._add_compile_flag(
            extension,
            "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
        )
