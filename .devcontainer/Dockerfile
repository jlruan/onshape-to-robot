FROM ubuntu:latest

# Avoid prompts during install
ENV DEBIAN_FRONTEND=noninteractive
# Install Python and pip
RUN apt-get update \
&& apt-get install -y --no-install-recommends python3 \
python3-pip \
build-essential \
git \
libeigen3-dev \
python3-dev \
cmake \
ninja-build \
llvm \
pkg-config \
libcgal-dev \
nlohmann-json3-dev \
libfmt-dev \
lsb-release \
curl \
&& rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/apt/keyrings && curl http://robotpkg.openrobots.org/packages/debian/robotpkg.asc \
| tee /etc/apt/keyrings/robotpkg.asc
RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" \
     | tee /etc/apt/sources.list.d/robotpkg.list

RUN python3 -m pip config set global.break-system-packages true
RUN pip install tyro viser

WORKDIR /opt
RUN git clone https://github.com/valstad-shipworks/eaik.git --recursive \
&& cd /opt/eaik \
&& pip install . 

WORKDIR /opt
RUN git clone https://github.com/CoMMALab/foam.git --recursive \
&& cd /opt/foam \
&& cmake -Bbuild -GNinja . \
&& cmake --build build/ \
&& pip install -r requirements.txt \
&& pip install -e .

# Install CppAD
RUN git clone https://github.com/coin-or/CppAD.git /tmp/CppAD && \
    cd /tmp/CppAD && \
    mkdir build && cd build && \
    cmake .. -Dcppad_testvector=cppad -Dcppad_prefix=/usr/local && \
    make -j$(nproc) && make install && \
    rm -rf /tmp/CppAD

# Install Inja (header-only but packaged through CMake)
RUN git clone https://github.com/pantor/inja.git /tmp/inja && \
    cd /tmp/inja && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && make install && \
    rm -rf /tmp/inja

# Install Pinocchio
RUN apt update && apt install -qqy robotpkg-py3*-pinocchio
ENV PATH=/opt/openrobots/bin:$PATH
ENV PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
ENV LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/opt/openrobots/lib/python3.12/site-packages:$PYTHONPATH
ENV CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH

# RUN git clone --recursive https://github.com/stack-of-tasks/pinocchio.git /tmp/pinocchio && \
#     cd /tmp/pinocchio && \
#     mkdir build && cd build && \
#     cmake .. \
#       -DCMAKE_BUILD_TYPE=Release \
#       -DBUILD_PYTHON_INTERFACE=OFF \
#       -DPYTHON_EXECUTABLE="" \
#       -DPINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION=OFF && \
#     ninja && ninja install && \
#     rm -rf /tmp/pinocchio

WORKDIR /opt
RUN git clone https://github.com/valstad-shipworks/cricket.git --recursive \
&& cd /opt/cricket \
&& cmake -GNinja -Bbuild -DCPM_SOURCE_CACHE=.cpm_cache cricket && cmake --build build