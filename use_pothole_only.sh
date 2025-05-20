#!/bin/bash

# Backup the original architecture.py
cp -f src/architecture.py src/architecture.py.original

# Copy our pothole-only version to be the main architecture.py
cp -f src/architecture_pothole_only.py src/architecture.py

echo "Architecture files swapped. Now the system will only use pothole detection."
echo "To restore the original setup, run './restore_architecture.sh'"
