 #!/bin/sh
 tensorflow_version=$1
docker buildx build \
    --build-arg TENSORFLOW_VERSION=$1 \
    --build-arg USRNM="$(whoami)" \
    --build-arg USRUID="$(id -u)" \
    --build-arg USRGID="$(id -g)" \
    -t qml/latent-qrl:latest .
