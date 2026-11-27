# syntax=docker/dockerfile:1
FROM ubuntu:24.04 AS build

ARG HDF5_VERSION=2.0.0
ARG HDF5_SHA256=f4c2edc5668fb846627182708dbe1e16c60c467e63177a75b0b9f12c19d7efed
ARG PYHEPEMSHOW_REPO=https://github.com/tobiaskortus/pyHepEmShow.git
# TODO: Pin PYHEPEMSHOW_REF to a commit or tag for reproducible builds.
ARG PYHEPEMSHOW_REF=example/dsbo-native-cli
ARG BUILD_TYPE=Release

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates curl pkg-config zlib1g-dev \
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

RUN git clone "${PYHEPEMSHOW_REPO}" /opt/pyHepEmShow \
    && git -C /opt/pyHepEmShow checkout "${PYHEPEMSHOW_REF}" \
    && git -C /opt/pyHepEmShow submodule update --init --recursive

COPY --from=dsbo-include / /opt/dsbo/include/
COPY --from=dsbo-licenses / /opt/dsbo/licenses/

RUN bash /opt/pyHepEmShow/externals/configure_native.sh \
        forward /opt/dsbo/include /opt/hdf5 "${BUILD_TYPE}" \
    && bash /opt/pyHepEmShow/externals/configure_native.sh \
        backward /opt/dsbo/include /opt/hdf5 "${BUILD_TYPE}" \
    && install -m 0755 /opt/pyHepEmShow/hepemshow/build_native_forward/HepEmShowNative /usr/local/bin/hepemshow-forward \
    && install -m 0755 /opt/pyHepEmShow/hepemshow/build_native_backward/HepEmShowNative /usr/local/bin/hepemshow-backward

COPY --from=hepemshow-data /hepem_data_fe_pbwo4.json /opt/hepemshow/data/hepem_data_fe_pbwo4.json

FROM ubuntu:24.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 zlib1g \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/opt/hdf5/lib

COPY --from=build /opt/hdf5/lib /opt/hdf5/lib
COPY --from=build /opt/hdf5/share/LICENSE /usr/share/licenses/hdf5/LICENSE
COPY --from=build /opt/dsbo/licenses /usr/share/licenses/dsbo
COPY --from=build /opt/pyHepEmShow/externals/g4hepem/install_forward/lib /opt/pyHepEmShow/externals/g4hepem/install_forward/lib
COPY --from=build /opt/pyHepEmShow/externals/g4hepem/install_backward/lib /opt/pyHepEmShow/externals/g4hepem/install_backward/lib
COPY --from=build /usr/local/bin/hepemshow-forward /usr/local/bin/hepemshow-forward
COPY --from=build /usr/local/bin/hepemshow-backward /usr/local/bin/hepemshow-backward
COPY --from=build /opt/hepemshow/data/hepem_data_fe_pbwo4.json /opt/hepemshow/data/hepem_data_fe_pbwo4.json
COPY runscript.sh /usr/local/bin/hepemshow-native
RUN chmod +x /usr/local/bin/hepemshow-native

ENTRYPOINT ["/usr/local/bin/hepemshow-native"]
