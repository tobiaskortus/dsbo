#include <hdf5.h>
#if H5_VERS_MAJOR < 2
#error "DSBO requires HDF5 >=2.0"
#endif
#if !defined(H5_HAVE_THREADSAFE) || !H5_HAVE_THREADSAFE
#error "DSBO requires HDF5_ENABLE_THREADSAFE=ON"
#endif
#if !defined(H5_HAVE_FILTER_DEFLATE) || !H5_HAVE_FILTER_DEFLATE
#error "DSBO's bundled HDF5 must support DEFLATE compression"
#endif

int main() {
    unsigned major = 0, minor = 0, patch = 0;
    bool threadsafe = false;
    return H5get_libversion(&major, &minor, &patch) < 0 || major < 2 ||
           H5is_library_threadsafe(&threadsafe) < 0 || !threadsafe ||
           H5Zfilter_avail(H5Z_FILTER_DEFLATE) <= 0;
}
