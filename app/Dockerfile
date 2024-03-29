# TensorFlow image base written by TensorFlow authors.
# Source: https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/tools/dockerfiles/partials/ubuntu/nvidia.partial.Dockerfile
# -------------------------------------------------------------------------
# ARG ARCH=
# ARG CUDA=10.1
# FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
# # ARCH and CUDA are specified again because the FROM directive resets ARGs
# # (but their default value is retained if set previously)
# ARG ARCH
# ARG CUDA
# ARG CUDNN=7.6.4.38-1
# ARG CUDNN_MAJOR_VERSION=7
# ARG LIB_DIR_PREFIX=x86_64
# ARG LIBNVINFER=6.0.1-1
# ARG LIBNVINFER_MAJOR_VERSION=6

# # Needed for string substitution
# SHELL ["/bin/bash", "-c"]
# # Pick up some TF dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         build-essential \
#         cuda-command-line-tools-${CUDA/./-} \
#         # There appears to be a regression in libcublas10=10.2.2.89-1 which
#         # prevents cublas from initializing in TF. See
#         # https://github.com/tensorflow/tensorflow/issues/9489#issuecomment-562394257
#         libcublas10=10.2.1.243-1 \ 
#         cuda-nvrtc-${CUDA/./-} \
#         cuda-cufft-${CUDA/./-} \
#         cuda-curand-${CUDA/./-} \
#         cuda-cusolver-${CUDA/./-} \
#         cuda-cusparse-${CUDA/./-} \
#         curl \
#         libcudnn7=${CUDNN}+cuda${CUDA} \
#         libfreetype6-dev \
#         libhdf5-serial-dev \
#         libzmq3-dev \
#         pkg-config \
#         software-properties-common \
#         unzip

# # Install TensorRT if not building for PowerPC
# RUN [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
#         apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
#         libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
#         && apt-get clean \
#         && rm -rf /var/lib/apt/lists/*; }

# # For CUDA profiling, TensorFlow requires CUPTI.
# ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# # Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# # dynamic linker run-time bindings
# RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
#     && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
#     && ldconfig
# # -------------------------------------------------------------------------
# #
# # Custom part
# FROM base
# ARG PYTHON_VERSION=3.7

# RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
#           python${PYTHON_VERSION} \
#           python3-pip \
#           python${PYTHON_VERSION}-dev \
# # Change default python
#     && cd /usr/bin \
#     && ln -sf python${PYTHON_VERSION}         python3 \
#     && ln -sf python${PYTHON_VERSION}m        python3m \
#     && ln -sf python${PYTHON_VERSION}-config  python3-config \
#     && ln -sf python${PYTHON_VERSION}m-config python3m-config \
#     && ln -sf python3                         /usr/bin/python \
# # Update pip and add common packages
#     && python -m pip install --upgrade pip \
#     && python -m pip install --upgrade \
#         setuptools \
#         wheel \
#         six \
# # Cleanup
#     && apt-get clean \
#     && rm -rf $HOME/.cache/pip



FROM python:3.8-slim
ENV PYTHONUNBUFFERED 1
RUN mkdir /app
WORKDIR /app
COPY ./requirements.txt /app/
EXPOSE 8000
RUN pip install opencv-python 
RUN apt update && apt install -y libsm6 libxext6 && apt-get install -y libxrender-dev
RUN apt-get -y update && apt-get install build-essential -y libpoppler-cpp-dev pkg-config python3-dev 
RUN apt-get install poppler-utils -y
RUN apt -y install tesseract-ocr tesseract-ocr-jpn libtesseract-dev libleptonica-dev tesseract-ocr-script-jpan tesseract-ocr-script-jpan-vert
RUN apt install git -y
RUN apt-get -y upgrade
RUN pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git' 
#RUN python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
# RUN git clone https://github.com/PaddlePaddle/PaddleOCR ##executed inside the cntainer 
COPY . /app/
