cmake_minimum_required(VERSION 3.26)

if(NOT DEFINED VERSION OR NOT VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+(-[A-Za-z0-9.-]+)?(\\+[A-Za-z0-9.-]+)?$")
  message(FATAL_ERROR "Supply VERSION as X.Y.Z, optionally with a prerelease/build suffix")
endif()
get_filename_component(source_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
if(NOT DEFINED OUTPUT_DIR)
  set(OUTPUT_DIR "${source_root}/dist/native")
endif()
get_filename_component(OUTPUT_DIR "${OUTPUT_DIR}" ABSOLUTE)

set(package "dsbo-native-${VERSION}")
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef staging_id)
set(staging "${OUTPUT_DIR}/.stage-${package}-${staging_id}")
set(root "${staging}/${package}")
file(MAKE_DIRECTORY "${root}")

# Only native source inputs: never include checkout build outputs or bindings.
set(inputs
  cpp/CMakeLists.txt cpp/README.md cpp/vendor/README.md
  cpp/cmake/BundledHDF5.cmake cpp/cmake/PackageSource.cmake
  cpp/examples/CMakeLists.txt cpp/examples/workflow_step.cpp
  cpp/tests/CMakeLists.txt cpp/tests/doctest.h cpp/tests/environment.cpp cpp/tests/hdf5.cpp
  cpp/tests/header_linkage_check.cpp cpp/tests/test_support.hpp cpp/tests/workflow_step.cpp
  docs/guide/native_io.md)
file(GLOB_RECURSE headers RELATIVE "${source_root}" "${source_root}/cpp/include/*.hpp")
file(GLOB archives LIST_DIRECTORIES FALSE RELATIVE "${source_root}" "${source_root}/cpp/vendor/*.tar.gz")
file(GLOB notices LIST_DIRECTORIES FALSE RELATIVE "${source_root}" "${source_root}/LICENSES/*")
if(NOT archives)
  message(FATAL_ERROR "No vendored native dependency archives found in cpp/vendor/")
endif()
foreach(input IN LISTS inputs headers archives notices)
  get_filename_component(directory "${root}/${input}" DIRECTORY)
  file(MAKE_DIRECTORY "${directory}")
  file(COPY_FILE "${source_root}/${input}" "${root}/${input}")
endforeach()

if(NOT DEFINED SOURCE_REVISION)
  find_package(Git QUIET)
  if(Git_FOUND AND EXISTS "${source_root}/.git")
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${source_root}" rev-parse HEAD
      OUTPUT_VARIABLE SOURCE_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  endif()
endif()
if(NOT SOURCE_REVISION)
  set(SOURCE_REVISION unknown)
endif()
file(WRITE "${root}/RELEASE.txt" "Version: ${VERSION}\nSource revision: ${SOURCE_REVISION}\n")
file(WRITE "${root}/README.md" "# DSBO native ${VERSION}\n\n"
  "Self-contained C++ sources, including HDF5/zlib archives, the JSON header, and the test-only doctest header.\n"
  "Requires CMake >=3.26, C/C++17 compilers, and Make or Ninja; no Python or network access.\n"
  "From this directory:\n\n```sh\n"
  "cmake -S cpp -B build/native -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON\n"
  "cmake --build build/native --target check --parallel 4\n```\n\n"
  "See [native development](cpp/README.md), the [I/O guide](docs/guide/native_io.md),\n"
  "and [license notices](LICENSES/README.md). Python integration examples require the full repository.\n"
  "No Python bindings or precompiled libraries are included.\n")

execute_process(COMMAND "${CMAKE_COMMAND}" -E tar czf "${staging}/${package}.tar.gz" "${package}"
  WORKING_DIRECTORY "${staging}" COMMAND_ERROR_IS_FATAL ANY)
file(RENAME "${staging}/${package}.tar.gz" "${OUTPUT_DIR}/${package}.tar.gz")
file(SHA256 "${OUTPUT_DIR}/${package}.tar.gz" checksum)
file(WRITE "${OUTPUT_DIR}/${package}.tar.gz.sha256" "${checksum}  ${package}.tar.gz\n")
file(REMOVE_RECURSE "${staging}")
message(STATUS "Created ${OUTPUT_DIR}/${package}.tar.gz and checksum")
