# Build container
FROM viros/is-cpp:1-openpose-cuda-9.0-cudnn7
ENV LD_LIBRARY_PATH /is-sdk/lib/x86_64-linux-gnu
ADD . /is-sdk/src/myservice
RUN b /is-sdk/src/myservice                                             \
 && mkdir -v -p /tmp/is/lib /tmp/is/bin                                 \
 && libs=`find /is-sdk/bin/ -type f -name '*.bin' -exec ldd {} \;       \
    | cut -d '(' -f 1 | cut -d '>' -f 2 | sort | uniq`                  \
 && for lib in $libs;                                                   \
    do                                                                  \
      dir="/tmp/is/lib`dirname $lib`";                                  \
      mkdir -v -p  $dir;                                                \
      cp --verbose $lib $dir;                                           \
    done                                                                \
&& cp --verbose `find /is-sdk/bin/ -type f -name '*.bin'` /tmp/is/bin/

# Deployment container
FROM scratch
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/is-sdk/lib:/is-sdk/lib/x86_64-linux-gnu \
    PATH=/is-sdk/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
COPY --from=0 /tmp/is/lib/ /
COPY --from=0 /tmp/is/bin/ /