include_guard(GLOBAL)
include(FetchContent)

function(dsbo_add_bundled_hdf5)
  # Function-local options keep the dependency builds from changing the parent
  # project's BUILD_TESTING/shared-library settings. Only local, checksum-pinned
  # archives are used; no HDF5/zlib discovery, downloads, or installation occurs.
  set(ZLIB_BUILD_SHARED OFF)
  set(ZLIB_BUILD_STATIC ON)
  set(ZLIB_BUILD_TESTING OFF)
  set(ZLIB_INSTALL OFF)
  FetchContent_Declare(zlib
    URL "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../vendor/zlib-1.3.2.tar.gz"
    URL_HASH SHA256=bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
  FetchContent_MakeAvailable(zlib)

  set(BUILD_SHARED_LIBS OFF)
  set(BUILD_STATIC_LIBS ON)
  set(BUILD_TESTING OFF)
  set(HDF5_ENABLE_THREADSAFE ON)
  set(HDF5_ENABLE_CONCURRENCY OFF)
  set(HDF5_ENABLE_PARALLEL OFF)
  set(HDF5_BUILD_CPP_LIB OFF)
  set(HDF5_BUILD_FORTRAN OFF)
  set(HDF5_BUILD_HL_LIB OFF)
  set(HDF5_BUILD_JAVA OFF)
  set(HDF5_BUILD_TOOLS OFF)
  set(HDF5_BUILD_EXAMPLES OFF)
  set(HDF5_ENABLE_SZIP_SUPPORT OFF)
  set(HDF5_ENABLE_ZLIB_SUPPORT ON)
  set(HDF5_ALLOW_EXTERNAL_SUPPORT NO)
  set(HDF5_ENABLE_EMBEDDED_LIBINFO OFF)
  set(HDF5_NO_PACKAGES ON)
  # HDF5 supports parent-provided zlib through these variables.
  set(H5_ZLIB_HEADER "zlib.h")
  set(H5_ZLIB_INCLUDE_DIRS "${zlib_SOURCE_DIR};${zlib_BINARY_DIR}")
  FetchContent_Declare(hdf5
    URL "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../vendor/hdf5-2.0.0.tar.gz"
    URL_HASH SHA256=f4c2edc5668fb846627182708dbe1e16c60c467e63177a75b0b9f12c19d7efed
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
  FetchContent_MakeAvailable(hdf5)
  target_link_libraries(hdf5-static PRIVATE zlibstatic)
  set_target_properties(hdf5-static zlibstatic PROPERTIES
    C_VISIBILITY_PRESET hidden POSITION_INDEPENDENT_CODE ON)
endfunction()
