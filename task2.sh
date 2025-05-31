#!/bin/bash

set -e  # Exit immediately if any command fails

echo "Starting Task 2: Creating and versioning dataset partitions..."

# Step 1: Run the Python script to create dataset partitions
echo "Generating dataset partitions..."
python create_partitions.py

# Ensure partitions were created
for version in v1 v2 v3; do
    if [ ! -d "data/partitions/$version" ]; then
        echo "Error: Dataset partition $version was not created correctly. Exiting..."
        exit 1
    fi
done

# Step 2: Untrack partitions from Git if already tracked
if git ls-files --error-unmatch data/partitions &> /dev/null; then
    echo "Removing data/partitions from Git tracking..."
    git rm -r --cached data/partitions
    git commit -m "Stop tracking data/partitions with Git"
fi

# Step 3: Add or commit dataset partitions to DVC
echo "Adding dataset partitions to DVC..."
if ! dvc add data/partitions; then
    echo "DVC add failed. Attempting to commit changes instead..."
    dvc commit data/partitions
fi


# Step 4: Configure local DVC remote (if not already configured)
if ! dvc remote list | grep -q '^myremote'; then
    read -p "Enter the remote location path for DVC (e.g., /home/amar/remote_location): " remote_path
    echo "Configuring local DVC remote at $remote_path..."
    dvc remote add -d myremote "$remote_path"
else
    echo "DVC remote 'myremote' is already configured."
fi

# Ensure .dvc files were created
for version in v1 v2 v3; do
    if [ ! -f "data/partitions/$version.dvc" ]; then
        echo "Error: DVC tracking failed for dataset partition $version. Exiting..."
        exit 1
    fi
done

# Step 5: Commit dataset partitions to Git & DVC
echo "Committing partitions to Git & DVC..."
git add data/partitions/.gitignore data/partitions/v1.dvc data/partitions/v2.dvc data/partitions/v3.dvc
git commit -m "Added dataset partitions v1, v2, and v3"

# Step 6: Push to local DVC storage
echo "Pushing partitions to local DVC storage..."
dvc push

echo "Task 2 completed successfully!"
