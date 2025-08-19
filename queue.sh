#!/bin/bash

# Loop indefinitely
while true; do
    # Check if any Python program is using the GPU
    if nvidia-smi -i 0 | grep 'python' > /dev/null; then
        echo "GPU is being used by a Python program. Waiting..."
    else
        echo "GPU is free. Launching the training script."

        bash -x train.sh

        # Break the loop if you want to run the script only once when GPU is free
        break
    fi

    # Wait for a minute before checking again
    sleep 30
done
