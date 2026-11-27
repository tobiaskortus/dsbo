add_executable(dsbo_native_step workflow_step.cpp)
target_link_libraries(dsbo_native_step PRIVATE dsbo::io)
