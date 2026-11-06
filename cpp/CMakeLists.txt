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

add_executable(dsbo_native_step examples/workflow_step.cpp)
target_link_libraries(dsbo_native_step PRIVATE dsbo::io)

if(BUILD_TESTING)
  # Compile each first-party header before any other include, then mix it with
  # the umbrella and include it again. Each source is a separate translation unit.
  file(GLOB_RECURSE sdk_headers CONFIGURE_DEPENDS
    RELATIVE "${DSBO_SDK_INCLUDE_DIR}" "${DSBO_SDK_INCLUDE_DIR}/dsbo/*.hpp")
  set(sdk_files ${sdk_headers})
  list(TRANSFORM sdk_files PREPEND "${DSBO_SDK_INCLUDE_DIR}/")
  # A manifest invalidates the copied tree when headers are added or removed.
  set(sdk_manifest "${CMAKE_CURRENT_BINARY_DIR}/sdk-headers.txt")
  file(GENERATE OUTPUT "${sdk_manifest}" CONTENT "${sdk_headers}\n")
  list(FILTER sdk_headers EXCLUDE REGEX "^dsbo/third_party/")
  set(header_checks)
  foreach(header IN LISTS sdk_headers)
    string(MAKE_C_IDENTIFIER "${header}" identifier)
    set(source "${CMAKE_CURRENT_BINARY_DIR}/header-checks/${identifier}.cpp")
    file(GENERATE OUTPUT "${source}" CONTENT
      "#include <${header}>\n#include <dsbo/hdf5.hpp>\n#include <${header}>\n")
    list(APPEND header_checks "${source}")
  endforeach()
  add_library(dsbo_header_checks OBJECT ${header_checks})
  target_link_libraries(dsbo_header_checks PRIVATE dsbo::io)

  add_executable(dsbo_native_tests tests/hdf5.cpp tests/second_translation_unit.cpp)
  target_link_libraries(dsbo_native_tests PRIVATE dsbo::io)
  add_test(NAME native_hdf5 COMMAND dsbo_native_tests "${CMAKE_CURRENT_BINARY_DIR}/test-files")
  set_tests_properties(native_hdf5 PROPERTIES TIMEOUT 60)

  # Compile and run the same public SDK tests with only a copied include tree.
  # Do not link dsbo::io here: its source-tree include path could hide omissions.
  set(copied_include "${CMAKE_CURRENT_BINARY_DIR}/copied-sdk/include")
  set(copied_stamp "${CMAKE_CURRENT_BINARY_DIR}/copied-sdk.stamp")
  add_custom_command(OUTPUT "${copied_stamp}"
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${copied_include}"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${DSBO_SDK_INCLUDE_DIR}" "${copied_include}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${copied_stamp}"
    DEPENDS ${sdk_files} "${sdk_manifest}"
    VERBATIM)
  add_custom_target(dsbo_copy_headers DEPENDS "${copied_stamp}")
  add_executable(dsbo_copied_sdk_tests tests/hdf5.cpp tests/second_translation_unit.cpp)
  add_dependencies(dsbo_copied_sdk_tests dsbo_copy_headers)
  target_compile_features(dsbo_copied_sdk_tests PRIVATE cxx_std_17)
  target_include_directories(dsbo_copied_sdk_tests PRIVATE "${copied_include}")
  target_link_libraries(dsbo_copied_sdk_tests PRIVATE hdf5-static Threads::Threads)
  add_test(NAME native_copied_sdk COMMAND dsbo_copied_sdk_tests "${CMAKE_CURRENT_BINARY_DIR}/copied-sdk-files")
  set_tests_properties(native_copied_sdk PROPERTIES TIMEOUT 60)

  add_executable(dsbo_environment_tests tests/environment.cpp)
  target_link_libraries(dsbo_environment_tests PRIVATE dsbo::io)
  foreach(scenario IN ITEMS supported old_version version_query_failure non_threadsafe threadsafe_query_failure)
    add_test(NAME native_environment_${scenario} COMMAND dsbo_environment_tests "${scenario}")
  endforeach()

  add_executable(dsbo_workflow_step_tests tests/workflow_step.cpp)
  target_link_libraries(dsbo_workflow_step_tests PRIVATE dsbo::io)
  foreach(scenario IN ITEMS reordered partial matching unnamed)
    add_test(NAME native_workflow_backward_${scenario}
      COMMAND "${CMAKE_COMMAND}"
        "-DTEST_HELPER=$<TARGET_FILE:dsbo_workflow_step_tests>"
        "-DNATIVE_STEP=$<TARGET_FILE:dsbo_native_step>"
        "-DSCENARIO=${scenario}"
        "-DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/workflow-step/${scenario}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/workflow_step.cmake")
  endforeach()

  add_custom_target(check
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure -C "$<CONFIG>"
    DEPENDS dsbo_header_checks dsbo_native_tests dsbo_copied_sdk_tests
            dsbo_environment_tests dsbo_workflow_step_tests dsbo_native_step
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    USES_TERMINAL
    VERBATIM)
endif()
