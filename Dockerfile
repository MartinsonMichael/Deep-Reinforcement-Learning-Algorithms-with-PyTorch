FROM python:3.6.9

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       apt-utils \
       build-essential \
       curl \
       xvfb \
       ffmpeg \
       xorg-dev \
       libsdl2-dev \
       swig \
       cmake \
       python-opengl

RUN pip3 install --upgrade pip
COPY pip.packages /tmp/
# RUN cat /tmp/pip.packages | xargs -n 1 pip3 install --trusted-host pypi.python.org

RUN pip3 install --trusted-host pypi.python.org -r /tmp/pip.packages
