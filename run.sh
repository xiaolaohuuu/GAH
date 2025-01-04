#!/bin/bash

# Define arrays for datasets, bit values, and GPUs
datasets=("nus")  # Add your datasets here
bits=(16 32 64 128)  # Add your bit values here
gpus=(2 3)  # GPU IDs

# Initialize a counter for GPU assignment
gpu_counter=0

# Loop through each dataset and bit value
for dataset in "${datasets[@]}"; do
    for bit in "${bits[@]}"; do
        log_file="logs/${dataset}/stomass_new/${dataset}_1_last20_${bit}.log"
        gpu_id="${gpus[$gpu_counter]}"

        echo "Running with dataset=${dataset}, bit=${bit} on GPU=${gpu_id}, logging to ${log_file}"
        
        CUDA_VISIBLE_DEVICES="${gpu_id}" nohup python main.py  --bit=${bit} --flag=${dataset} > "${log_file}" 2>&1 &

        # Increment the GPU counter
        gpu_counter=$((gpu_counter + 1))

        # Reset the GPU counter if it exceeds the number of GPUs 
        if [ $gpu_counter -ge ${#gpus[@]} ]; then
        gpu_counter=0
        fi
    done
done

echo "All processes started."
