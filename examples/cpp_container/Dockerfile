FROM ubuntu:24.04 AS build
ARG HDF5_VERSION=2.0.0
ARG HDF5_SHA256=f4c2edc5668fb846627182708dbe1e16c60c467e63177a75b0b9f12c19d7efed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake curl ca-certificates pkg-config zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
RUN curl -fL "https://github.com/HDFGroup/hdf5/releases/download/${HDF5_VERSION}/hdf5-${HDF5_VERSION}.tar.gz" -o hdf5.tar.gz \
    && echo "${HDF5_SHA256}  hdf5.tar.gz" | sha256sum -c - \
    && tar -xzf hdf5.tar.gz \
    && cmake -S "hdf5-${HDF5_VERSION}" -B hdf5-build \
       -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/hdf5 \
       -DHDF5_ENABLE_THREADSAFE=ON -DHDF5_BUILD_CPP_LIB=OFF \
       -DHDF5_BUILD_FORTRAN=OFF -DHDF5_BUILD_HL_LIB=OFF \
       -DHDF5_ENABLE_PARALLEL=OFF -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF \
       -DBUILD_TESTING=OFF -DHDF5_BUILD_EXAMPLES=OFF -DHDF5_BUILD_TOOLS=OFF \
       -DHDF5_ENABLE_ZLIB_SUPPORT=ON -DHDF5_ENABLE_SZIP_SUPPORT=OFF \
    && cmake --build hdf5-build --parallel 2 --target install
COPY dsbo-sdk /opt/dsbo-sdk
COPY workflow_step.cpp /build/workflow_step.cpp
RUN c++ -std=c++17 -O2 -pthread \
    -I/opt/dsbo-sdk/include -I/opt/hdf5/include workflow_step.cpp \
    -L/opt/hdf5/lib -Wl,-rpath,/opt/hdf5/lib -lhdf5 -o /build/dsbo_native_step

FROM ubuntu:24.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 zlib1g \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/hdf5/lib /opt/hdf5/lib
COPY --from=build /opt/hdf5/share/LICENSE /usr/share/licenses/hdf5/LICENSE
COPY --from=build /opt/dsbo-sdk/licenses /usr/share/licenses/dsbo-sdk
COPY --from=build /build/dsbo_native_step /usr/local/bin/dsbo_native_step
ENTRYPOINT ["/usr/local/bin/dsbo_native_step"]
