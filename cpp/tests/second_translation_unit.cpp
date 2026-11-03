#include <dsbo/hdf5/statistics.hpp>
#include <dsbo/hdf5/storage.hpp>
#include <dsbo/hdf5.hpp>

dsbo::io::DType second_translation_unit() {
    dsbo::io::check_environment();
    return dsbo::io::dtype<double>();
}

dsbo::io::Statistics second_translation_unit_statistics(dsbo::io::ArrayView value) {
    const auto converted = dsbo::io::convert(value, dsbo::io::dtype<double>());
    return dsbo::io::statistics(converted.view()).value();
}
