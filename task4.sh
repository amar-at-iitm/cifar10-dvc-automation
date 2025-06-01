#!/bin/bash

set -e  # Exit on any error

echo "Starting Task 4: Experiment Pipeline Runs..."

# Array of dataset combinations
dataset_combinations=("v1" "v2" "v3" "v1+v2" "v1+v2+v3")

# Repeat runs for 3 different random seed values
for seed in 42 100 2025; do
    echo "Running experiments with random seed: $seed"

    for combination in "${dataset_combinations[@]}"; do
        echo "Running pipeline for dataset combination: $combination"

        # Update the config file with the current seed
        sed -i "s/^seed: .*/seed: $seed/" config.yaml
        
        # Update the dataset version in the config file
        combination_escaped=$(echo "$combination" | sed 's/+/\\+/g')
        sed -i "s|^  version: .*|  version: \"$combination_escaped\"|" config.yaml


        # Store the combination name in a temporary file
        echo "$combination" > version.txt

        # Reproduce the pipeline with the new configuration
        dvc repro

        # Push the results to remote storage
        dvc push
    done
done

# Display all experiments
echo "Displaying all experiment results..."
dvc exp show

echo "Task 4 completed successfully!"
