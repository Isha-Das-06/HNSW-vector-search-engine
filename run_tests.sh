#!/bin/bash
# Run the test suite

cd "$(dirname "$0")"

echo "Running HNSW test suite..."
echo "========================================="

python -m pytest tests/test_hnsw.py -v --tb=short

echo ""
echo "Test run complete."
