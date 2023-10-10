#!/bin/bash

# Navigate to the data_engineering directory from the root directory
cd data_engineering

# Run the Python files in sequence
python3 1_bronze_layer.py
python3 2_silver_layer.py
python3 3_gold_layer.py

# Navigate back to the root directory
cd ..

echo "All scripts executed successfully!"