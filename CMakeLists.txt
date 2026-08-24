cmake_minimum_required(VERSION 3.15)
project(lbvh2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

# NOTE: nanobind 2.x only supports stable-ABI (abi3) wheels on Python 3.12+.
# We target 3.10+ which means we cannot use STABLE_ABI; each Python version
# gets its own per-version wheel. This matches the original `lbvh` package.
nanobind_add_module(
    _lbvh2
    NB_DOMAIN lbvh2
    src/cpp/bindings.cpp
)

target_compile_options(_lbvh2 PRIVATE -O3 -funroll-loops)

install(TARGETS _lbvh2 LIBRARY DESTINATION lbvh2)
