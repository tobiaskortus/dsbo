#pragma once

// DSBO's header-only native storage SDK. Requires C++17 and thread-safe HDF5 >= 2.0.
//
// Public entry point: array/value types, numeric conversion and statistics,
// synchronized Reader, and stateless storage operations. Implementation layers
// under hdf5/detail/ are internal; distribute the complete dsbo/ header tree.
#include "hdf5/types.hpp"
#include "hdf5/detail/arrays.hpp"
#include "hdf5/statistics.hpp"
#include "hdf5/reader.hpp"
#include "hdf5/storage.hpp"
