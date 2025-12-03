FROM ubuntu:latest

# Avoid prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and pip
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip build-essential git libeigen3-dev python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install tyro viser --break-system-packages

RUN git clone https://github.com/valstad-shipworks/eaik.git --recursive && cd eaik && pip install . --break-system-packages