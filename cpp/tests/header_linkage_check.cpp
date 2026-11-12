#include <dsbo/hdf5/statistics.hpp>
#include <dsbo/hdf5/storage.hpp>
#include <dsbo/hdf5.hpp>

dsbo::io::DType double_dtype_from_separate_cpp_file() {
    dsbo::io::check_environment();
    return dsbo::io::dtype<double>();
}

dsbo::io::Statistics double_statistics_from_separate_cpp_file(dsbo::io::ArrayView value) {
    const auto converted = dsbo::io::convert(value, dsbo::io::dtype<double>());
    return dsbo::io::statistics(converted.view()).value();
}
