#!/bin/bash
# This script can be run in terminal whenever the venv fails

# Remove existing virtual environment
rm -rf .venv
echo "venv removed"

# Deactivate virtual environment if active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    deactivate
fi

# Create a new virtual environment
python3.10 -m venv .venv
echo "New venv created"

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
echo "Pip upgraded"

# Restore packages
pip install -r requirements.txt

# Indicate when process is finished
echo "Virtual environment reset and activated."

# Run: ./reset_venv.sh