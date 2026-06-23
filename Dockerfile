# Build with the command:
# docker build --platform=linux/amd64 -f Dockerfile --tag thombadings/lograsm:v1 .

FROM ubuntu:latest
WORKDIR /home

USER root
RUN apt-get update && apt-get install -y wget git

# Miniconda
ENV PATH="/home/miniconda3/bin:$PATH"
ARG PATH="/home/miniconda3/bin:$PATH"
RUN mkdir -p miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
            -O /home/miniconda3/miniconda.sh \
    && bash /home/miniconda3/miniconda.sh -b -u -p /home/miniconda3 \
    && rm -f /home/miniconda3/miniconda.sh \
    && /home/miniconda3/bin/conda init bash \
    && /home/miniconda3/bin/pip install --upgrade pip
RUN /home/miniconda3/bin/conda install -y ipython

# JAX
RUN /home/miniconda3/bin/pip install --upgrade "jax[cuda12]"

# Build artifact dependencies
#############
RUN mkdir /home/lograsm
WORKDIR /home/lograsm

# Obtain requirements and install them
COPY requirements.txt requirements.txt
RUN /home/miniconda3/bin/pip install --upgrade pip
RUN /home/miniconda3/bin/pip install --no-cache-dir -r requirements.txt

# Copy all files
WORKDIR /home/lograsm
COPY . .