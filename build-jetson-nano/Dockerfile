FROM nvcr.io/nvidia/l4t-base:r32.7.1

# Install required packages
RUN apt update && apt install -y build-essential curl libfontconfig-dev && apt clean

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
ENV PATH="/root/.cargo/bin:${PATH}"

# Add NVIDIA apt key
RUN wget -O apt.key https://gitlab.com/nvidia/container-images/l4t-base/-/raw/master/jetson-ota-public.key?inline=false && apt-key add apt.key && rm apt.key
RUN echo "deb https://repo.download.nvidia.com/jetson/common r32 main" >> /etc/apt/sources.list

# Install cuda toolkit
RUN apt update && apt install -y cuda-toolkit-10-0 && apt clean

CMD cd /proj/cuda-viewer && cargo b --release --features zero-copy-mem -Z sparse-registry && \
    cd /proj/app && nvcc main.cu -o app -lcuda_viewer -I../cuda-viewer -L../cuda-viewer/target/release -DUSE_ZERO_COPY_MEMORY
