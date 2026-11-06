execute_process(
  COMMAND "${TEST_HELPER}" setup "${SCENARIO}" "${TEST_DIRECTORY}"
  COMMAND_ERROR_IS_FATAL ANY)

execute_process(
  COMMAND "${NATIVE_STEP}" backward
    "${TEST_DIRECTORY}/input_gradient.h5" "${TEST_DIRECTORY}/output_gradient.h5"
  COMMAND_ERROR_IS_FATAL ANY)

execute_process(
  COMMAND "${TEST_HELPER}" verify "${SCENARIO}" "${TEST_DIRECTORY}"
  COMMAND_ERROR_IS_FATAL ANY)
