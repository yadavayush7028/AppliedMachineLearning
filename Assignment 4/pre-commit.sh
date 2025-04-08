#!/bin/bash

echo "Running tests before commit..."

python3 test.py

if [ $? -ne 0 ]; then
  echo "Tests failed. Commit aborted."
  exit 1
fi

echo "All tests passed. Proceeding with commit."
