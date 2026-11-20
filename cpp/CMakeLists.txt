cmake_minimum_required(VERSION 3.26)
project(dsbo_native LANGUAGES C CXX)

# Repository tests/examples only; adopters use their own HDF5 build and include
# the headers directly. This build needs neither Python nor an installed SDK.
set(DSBO_SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include"
    CACHE PATH "Native header include directory (may point to a copied include tree)")
include(CTest)
include(cmake/BundledHDF5.cmake)
dsbo_add_bundled_hdf5()
find_package(Threads REQUIRED)

add_library(dsbo_io INTERFACE)
add_library(dsbo::io ALIAS dsbo_io)
target_compile_features(dsbo_io INTERFACE cxx_std_17)
target_include_directories(dsbo_io INTERFACE "${DSBO_SDK_INCLUDE_DIR}")
target_link_libraries(dsbo_io INTERFACE hdf5-static Threads::Threads)

# Keep executables in the build root for existing workflows and examples.
if(NOT DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()

add_subdirectory(examples)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
