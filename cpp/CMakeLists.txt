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
  add_executable(dsbo_native_tests tests/hdf5.cpp tests/second_translation_unit.cpp)
  target_link_libraries(dsbo_native_tests PRIVATE dsbo::io)
  add_test(NAME native_hdf5 COMMAND dsbo_native_tests "${CMAKE_CURRENT_BINARY_DIR}/test-files")

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
    DEPENDS dsbo_native_tests dsbo_environment_tests dsbo_workflow_step_tests dsbo_native_step
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    USES_TERMINAL
    VERBATIM)
endif()
