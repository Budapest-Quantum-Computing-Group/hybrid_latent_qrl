ARG TENSORFLOW_VERSION=2.15.0

FROM tensorflow/tensorflow:${TENSORFLOW_VERSION}

# User and group variables
ARG USRNM
ARG USRUID
ARG USRGID

# use bash instead sh
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -yq git llvm
# install opencv dependencies
RUN apt-get install -yq ffmpeg libsm6 libxext6

# Create user and add to the specified group
RUN groupadd -g ${USRGID} ${USRNM} || echo "Could not create group with GID ${USERGID}"
RUN useradd -g ${USRGID} -u ${USRUID} -m ${USRNM}
USER $USRNM


# Set the workdir to tmp to copy the requirements.txt
# to install the dependencies
WORKDIR /tmp
COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /

ENTRYPOINT ["/bin/bash"]
