#!/bin/bash

# Loop through each line in the requirements.txt
while read -r package; do
    # Try to install the package
    pip install "$package"
    # Check if the installation failed
    if [ $? -ne 0 ]; then
        # Print an error message but continue with the next package
        echo "Failed to install $package. Continuing..."
    fi
done < requirements.txt
