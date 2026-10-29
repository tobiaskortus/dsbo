cmake_minimum_required(VERSION 3.26)
project(dsbo_python LANGUAGES C CXX)

# This project is only for the Python package. ../CMakeLists.txt remains a
# separate, Python-free build against a caller-provided HDF5 installation.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_SKIP_RPATH TRUE)
include(FetchContent)

# FetchContent extracts checksum-verified LOCAL archives: no network access or
# system dependency discovery is needed, including when building from an sdist.
set(ZLIB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZLIB_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(ZLIB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ZLIB_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(zlib
    URL "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/zlib-1.3.2.tar.gz"
    URL_HASH SHA256=bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
FetchContent_MakeAvailable(zlib)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(HDF5_ENABLE_THREADSAFE ON CACHE BOOL "" FORCE)
set(HDF5_ENABLE_CONCURRENCY OFF CACHE BOOL "" FORCE)
set(HDF5_ENABLE_PARALLEL OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_CPP_LIB OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_FORTRAN OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_HL_LIB OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_JAVA OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HDF5_ENABLE_SZIP_SUPPORT OFF CACHE BOOL "" FORCE)
set(HDF5_ENABLE_ZLIB_SUPPORT ON CACHE BOOL "" FORCE)
set(HDF5_ALLOW_EXTERNAL_SUPPORT NO CACHE STRING "" FORCE)
set(HDF5_ENABLE_EMBEDDED_LIBINFO OFF CACHE BOOL "" FORCE)
set(HDF5_NO_PACKAGES ON CACHE BOOL "" FORCE)
# HDF5 explicitly supports parent-provided zlib through these variables.
set(H5_ZLIB_HEADER "zlib.h")
set(H5_ZLIB_INCLUDE_DIRS "${zlib_SOURCE_DIR};${zlib_BINARY_DIR}")
FetchContent_Declare(hdf5
    URL "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/hdf5-2.0.0.tar.gz"
    URL_HASH SHA256=f4c2edc5668fb846627182708dbe1e16c60c467e63177a75b0b9f12c19d7efed
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
FetchContent_MakeAvailable(hdf5)
target_link_libraries(hdf5-static PRIVATE zlibstatic)
set_target_properties(hdf5-static zlibstatic PROPERTIES
    C_VISIBILITY_PRESET hidden POSITION_INDEPENDENT_CODE ON)

add_executable(dsbo_check_hdf5 check_hdf5.cpp)
target_link_libraries(dsbo_check_hdf5 PRIVATE hdf5-static)
set_target_properties(dsbo_check_hdf5 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 3.0 CONFIG REQUIRED)
pybind11_add_module(_hdf5 MODULE NO_EXTRAS ../bindings/hdf5.cpp)
target_compile_features(_hdf5 PRIVATE cxx_std_17)
target_include_directories(_hdf5 PRIVATE ../../dsbo/utils/io/storage/include)
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
