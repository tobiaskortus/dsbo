cmake_minimum_required(VERSION 3.26)
project(dsbo_python LANGUAGES C CXX)

# This project is only for the Python package. ../CMakeLists.txt remains a
# separate, Python-free test/example build using the same bundled dependencies.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_SKIP_RPATH TRUE)
include(../cmake/BundledHDF5.cmake)
dsbo_add_bundled_hdf5()

add_executable(dsbo_check_hdf5 check_hdf5.cpp)
target_link_libraries(dsbo_check_hdf5 PRIVATE hdf5-static)
set_target_properties(dsbo_check_hdf5 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 3.0 CONFIG REQUIRED)
pybind11_add_module(_hdf5 MODULE NO_EXTRAS ../bindings/hdf5.cpp)
target_compile_features(_hdf5 PRIVATE cxx_std_17)
target_include_directories(_hdf5 PRIVATE ../include)
target_link_libraries(_hdf5 PRIVATE hdf5-static)
set_target_properties(_hdf5 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${DSBO_EXTENSION_DIRECTORY}"
    RUNTIME_OUTPUT_DIRECTORY "${DSBO_EXTENSION_DIRECTORY}"
    SUFFIX "${DSBO_EXTENSION_SUFFIX}")

# Only the Python entry point is public. In particular, another extension's
# HDF5/zlib (e.g. h5py's) must never interpose on DSBO's private copies.
if(APPLE)
    target_link_options(_hdf5 PRIVATE "LINKER:-exported_symbol,_PyInit__hdf5")
elseif(UNIX)
    target_link_options(_hdf5 PRIVATE "LINKER:--exclude-libs,ALL"
        "LINKER:--version-script,${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
    set_property(TARGET _hdf5 APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
endif()
# On Windows, static HDF5/zlib have no dllexport decorations; pybind11 exports
# PyInit__hdf5 explicitly. Neither dependency is built as a DLL.
