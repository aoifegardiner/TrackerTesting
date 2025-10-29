#!/bin/bash
# Usage: ./rundocker.sh /path/to/dataset /path/to/output

DATASETLOCATION=$1   # host dataset directory
OUTPUTLOCATION=$2    # host output directory
TAG=stir_agardiner   # tag for your built image

# Detect GPU runtime
if docker info 2>/dev/null | grep -q 'nvidia'; then
    GPU_RUNTIME="--runtime=nvidia"
else
    GPU_RUNTIME="--gpus all"
fi

# === For submission / running model ===
docker run --rm $GPU_RUNTIME --net host --ipc=host \
   --cap-add=CAP_SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 \
   -v /var/run/docker.sock:/var/run/docker.sock \
   --mount src=$DATASETLOCATION,target=/workspace/data,type=bind \
   --mount src=$OUTPUTLOCATION,target=/workspace/output,type=bind \
   $TAG /bin/bash -c "python datatest/flow2d.py --num_data 1 --showvis 0 --jsonsuffix test --modeltype MFTWAFT --ontestingset 1"

## For Submission -- 3D
#docker run --rm $GPU_RUNTIME --net host --ipc=host \
#   --cap-add=CAP_SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 \
#   -v /var/run/docker.sock:/var/run/docker.sock \
#   --mount src=$DATASETLOCATION,target=/workspace/data,type=bind \
#   --mount src=$OUTPUTLOCATION,target=/workspace/output,type=bind \
#   $TAG /bin/bash -c "cd STIRMetrics/src && python datatest/flow3d.py --num_data -1 --showvis 0 --jsonsuffix test --modeltype RAFT_Stereo_RAFT --ontestingset 1"
