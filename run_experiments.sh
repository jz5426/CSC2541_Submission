#!/bin/bash

# Function to run experiments
run_experiment() {
    local sampler=$1
    local with_cfg=$2

    # Define the values for `ita` and `sampling step`
    if [[ "$sampler" == "DDPM" ]]; then
        ita_values=(1)
    else
        ita_values=(0, 0.2, 0.5, 1.)
    fi

    sampling_steps=(10, 20, 50, 100)

    # Iterate over each combination of `ita` and `sampling step`
    for ita in "${ita_values[@]}"; do
        for step in "${sampling_steps[@]}"; do
            cfg="2.9"
            temperature="1.0"
            if [[ "$with_cfg" == "false" ]]; then
                cfg="1"
                temperature="0.95"
            fi

            echo "Running command with sampler=$sampler, ita=$ita, sampling_steps=$step, with_cfg=$with_cfg..."

            # Run the torchrun command
            torchrun \
                --nproc_per_node=1 \
                --nnodes=1 \
                --node_rank=0 \
                main_mar.py \
                --sampler "$sampler" \
                --model mar_base \
                --diffloss_d 6 \
                --diffloss_w 1024 \
                --eval_bsz 64 \
                --num_images 2000 \
                --num_iter 64 \
                --cfg "$cfg" \
                --cfg_schedule linear \
                --temperature "$temperature" \
                --output_dir pretrained_models/mar/mar_base \
                --resume pretrained_models/mar/mar_base \
                --data_path ./ \
                --evaluate \
                --ita "$ita" \
                --num_sampling_steps "$step"

            if [[ $? -eq 0 ]]; then
                echo "Command executed successfully!"
            else
                echo "Command failed for sampler=$sampler, ita=$ita, sampling_steps=$step, with_cfg=$with_cfg"
            fi
        done
    done
}

# Run experiments

# Experiment 2: DDIM
run_experiment "DDIM" true  # With classifier-free guidance
run_experiment "DDIM" false # Without classifier-free guidance

# # # Experiment 1: DDPM
run_experiment "DDPM" true  # With classifier-free guidance
# run_experiment "DDPM" false # Without classifier-free guidance
