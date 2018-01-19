if [ "${1}" == "" ]; then
    nvidia-docker build -t qhan1028/headpose_detection .
else
    nvidia-docker build -t qhan1028/headpose_detection:${1} .
fi
