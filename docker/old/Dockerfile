FROM viros/is-cpp:1-cuda-8.0-cudnn5

RUN apt-get update && apt-get install -y --no-install-recommends \
  libatlas-base-dev                                              \
  libgflags-dev                                                  \
  libgoogle-glog-dev                                             \
  libhdf5-serial-dev                                             \
  libleveldb-dev                                                 \
  liblmdb-dev                                                    \
  libsnappy-dev                                                  \
  python-dev                                                     \
  python-numpy                                                   \
  python-pip                                                     \
  python-setuptools                                              \
  python-scipy                                                   \
  lsb-release
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose/ \
 && cd openpose                                                  \
 && mkdir build                                                  \
 && cd build                                                     \
 && cmake ..                                                     \
     -DBUILD_EXAMPLES=OFF                                        \
     -DBUILD_DOCS=OFF                                            \
     -DDOWNLOAD_COCO_MODEL=ON                                    \
     -DDOWNLOAD_FACE_MODEL=ON                                    \
     -DDOWNLOAD_HAND_MODEL=ON                                    \
     -DDOWNLOAD_MPI_MODEL=ON                                     \
 && make -j`nproc`                                               \
 && make install                                                 \
 && ldconfig                                                     \
 && cp -r /openpose/models /models                               \
 && rm -rf /openpose