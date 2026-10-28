#include <dsbo/hdf5.hpp>

dsbo::io::DType second_translation_unit() {
    dsbo::io::check_environment();
    return dsbo::io::dtype<double>();
}
