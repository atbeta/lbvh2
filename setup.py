"""Build setup for lbvh2.

We compile src/cpp/bindings.cpp directly via pybind11's Pybind11Extension —
no scikit-build-core, no CMake. pybind11 >= 3 removed the previous
Pybind11CmakeBuild helper, so the modern path is plain setuptools +
Pybind11Extension (and the same compiler flags we used to set in
CMakeLists.txt move here).

Per-Python-version wheels (no abi3); see pyproject.toml note.
"""
from __future__ import annotations

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

# Compiler flags previously set in CMakeLists.txt. Pass as a single list
# (not a per-compiler dict) because pybind11's cxx_std setter slice-
# assigns into extra_compile_args and only handles the list form. The
# GCC/Clang-only flags are no-ops on MSVC; setuptools tolerates that.
_OPT_FLAGS = ["-O3", "-funroll-loops"]

ext_modules = [
    Pybind11Extension(
        "lbvh2._lbvh2",
        ["src/cpp/bindings.cpp"],
        include_dirs=["src/cpp"],
        extra_compile_args=_OPT_FLAGS,
        cxx_std=17,
    ),
]

setup(
    # Project metadata lives in pyproject.toml (PEP 621). Only the actual
    # extension config lives here.
    ext_modules=ext_modules,
    zip_safe=False,
)