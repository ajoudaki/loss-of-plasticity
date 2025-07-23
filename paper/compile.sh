#!/bin/bash

# LaTeX compilation script for the paper
# This script compiles main.tex and outputs all files to the out/ directory

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Compiling paper in: $SCRIPT_DIR"

# Create output directory if it doesn't exist
mkdir -p out

# Copy necessary files to out directory
echo "Copying style and bibliography files..."
cp *.sty *.bib out/ 2>/dev/null || true

# Set TEXINPUTS to find files in parent directories
export TEXINPUTS="..:..//:"

echo "Running pdflatex (1st pass)..."
pdflatex -output-directory=out -interaction=nonstopmode main.tex

echo "Running bibtex..."
cd out
bibtex main || echo "Warning: bibtex failed, continuing..."
cd ..

echo "Running pdflatex (2nd pass)..."
pdflatex -output-directory=out -interaction=nonstopmode main.tex

echo "Running pdflatex (3rd pass)..."
pdflatex -output-directory=out -interaction=nonstopmode main.tex

echo ""
echo "Compilation complete!"
echo "PDF output: $SCRIPT_DIR/out/main.pdf"

# Check if PDF was created successfully
if [ -f "out/main.pdf" ]; then
    echo "✓ PDF successfully created"
    # Get PDF size
    PDF_SIZE=$(du -h "out/main.pdf" | cut -f1)
    echo "  File size: $PDF_SIZE"
else
    echo "✗ PDF creation failed"
    exit 1
fi