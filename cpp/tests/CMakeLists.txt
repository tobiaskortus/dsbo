# configure_file tracks header edits; CONFIGURE_DEPENDS tracks additions/removals.
set(copied_include "${CMAKE_CURRENT_BINARY_DIR}/copied-sdk/include")
file(REMOVE_RECURSE "${copied_include}")
file(GLOB_RECURSE headers CONFIGURE_DEPENDS
  RELATIVE "${DSBO_SDK_INCLUDE_DIR}" "${DSBO_SDK_INCLUDE_DIR}/dsbo/*.hpp")
set(header_checks)
foreach(header IN LISTS headers)
  configure_file("${DSBO_SDK_INCLUDE_DIR}/${header}" "${copied_include}/${header}" COPYONLY)
  if(NOT header MATCHES "^dsbo/third_party/")
    # Each source checks one header on its own, include order, and repeated inclusion.
    set(source "${CMAKE_CURRENT_BINARY_DIR}/header-checks/${header}.cpp")
    file(GENERATE OUTPUT "${source}" CONTENT
      "#include <${header}>\n#include <dsbo/hdf5.hpp>\n#include <${header}>\n")
    list(APPEND header_checks "${source}")
  endif()
endforeach()
add_library(dsbo_header_checks OBJECT ${header_checks})
target_link_libraries(dsbo_header_checks PRIVATE dsbo::io)

add_executable(dsbo_native_tests hdf5.cpp header_linkage_check.cpp)
target_link_libraries(dsbo_native_tests PRIVATE dsbo::io)
add_test(NAME native_hdf5 COMMAND dsbo_native_tests)

# Avoid dsbo::io so its source-tree include path cannot hide missing copied headers.
add_executable(dsbo_copied_sdk_tests hdf5.cpp header_linkage_check.cpp)
target_compile_features(dsbo_copied_sdk_tests PRIVATE cxx_std_17)
target_include_directories(dsbo_copied_sdk_tests PRIVATE "${copied_include}")
target_link_libraries(dsbo_copied_sdk_tests PRIVATE hdf5-static Threads::Threads)
add_test(NAME native_copied_sdk COMMAND dsbo_copied_sdk_tests)

add_executable(dsbo_environment_tests environment.cpp)
target_link_libraries(dsbo_environment_tests PRIVATE dsbo::io)
add_test(NAME native_environment COMMAND dsbo_environment_tests)
set_tests_properties(native_environment PROPERTIES TIMEOUT 30)

add_executable(dsbo_workflow_step_tests workflow_step.cpp)
target_link_libraries(dsbo_workflow_step_tests PRIVATE dsbo::io)
add_dependencies(dsbo_workflow_step_tests dsbo_native_step)
add_test(NAME native_workflow_backward COMMAND dsbo_workflow_step_tests)
set_tests_properties(native_hdf5 native_copied_sdk native_workflow_backward PROPERTIES TIMEOUT 60)

add_custom_target(check
  COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure -C "$<CONFIG>"
  DEPENDS dsbo_header_checks dsbo_native_tests dsbo_copied_sdk_tests
          dsbo_environment_tests dsbo_workflow_step_tests dsbo_native_step
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  USES_TERMINAL
  VERBATIM)
