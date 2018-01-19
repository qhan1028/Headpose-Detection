nvidia-docker run -it --rm \
    -v /data/qhan/Headpose-Detection:/app \
    -p 1028:1028 \
    --name headpose-detection-dev \
    qhan1028/headpose_detection $@
    bash
