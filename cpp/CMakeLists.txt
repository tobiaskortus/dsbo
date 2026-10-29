cmake_minimum_required(VERSION 3.20)
project(dsbo_native LANGUAGES C CXX)

# This build is deliberately independent of Python and its packaging backend.
set(DSBO_SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include"
    CACHE PATH "Native header include directory (may point to a copied include tree)")
find_package(HDF5 2.0 REQUIRED COMPONENTS C)
find_package(Threads REQUIRED)

include(CheckCSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${HDF5_INCLUDE_DIRS})
check_c_source_compiles("
  #include <hdf5.h>
  #if H5_VERS_MAJOR < 2 || !defined(H5_HAVE_THREADSAFE) || !H5_HAVE_THREADSAFE
  #error DSBO requires thread-safe HDF5 >=2.0
  #endif
  int main(void) { return 0; }
" DSBO_HDF5_HEADERS_SUPPORTED)
if(NOT DSBO_HDF5_HEADERS_SUPPORTED)
  message(FATAL_ERROR "DSBO requires HDF5 >=2.0 built with HDF5_ENABLE_THREADSAFE=ON")
endif()

add_library(dsbo_io INTERFACE)
add_library(dsbo::io ALIAS dsbo_io)
target_compile_features(dsbo_io INTERFACE cxx_std_17)
target_include_directories(dsbo_io INTERFACE "${DSBO_SDK_INCLUDE_DIR}" ${HDF5_INCLUDE_DIRS})
target_link_libraries(dsbo_io INTERFACE ${HDF5_LIBRARIES} Threads::Threads)

add_executable(dsbo_native_step examples/workflow_step.cpp)
target_link_libraries(dsbo_native_step PRIVATE dsbo::io)

include(CTest)
if(BUILD_TESTING)
  add_executable(dsbo_native_tests tests/hdf5.cpp tests/second_translation_unit.cpp)
  target_link_libraries(dsbo_native_tests PRIVATE dsbo::io)
  add_test(NAME native_hdf5 COMMAND dsbo_native_tests "${CMAKE_CURRENT_BINARY_DIR}/test-files")
endif()
