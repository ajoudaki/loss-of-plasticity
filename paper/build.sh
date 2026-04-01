#!/bin/bash
# Build LaTeX documents with all output in out/ directory
# Usage: ./build.sh [slides|main|poster|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/out"
mkdir -p "$OUT_DIR"

build_doc() {
    local texfile="$1"
    local basename="${texfile%.tex}"
    echo "Building $texfile ..."
    pdflatex -interaction=nonstopmode -output-directory="$OUT_DIR" "$SCRIPT_DIR/$texfile"
    # Run twice for references/toc
    pdflatex -interaction=nonstopmode -output-directory="$OUT_DIR" "$SCRIPT_DIR/$texfile"
    echo "Output: $OUT_DIR/$basename.pdf"
}

case "${1:-slides}" in
    slides)  build_doc slides.tex ;;
    main)    build_doc main.tex ;;
    poster)  build_doc poster.tex ;;
    all)
        build_doc slides.tex
        build_doc main.tex
        build_doc poster.tex
        ;;
    *)
        echo "Usage: $0 [slides|main|poster|all]"
        exit 1
        ;;
esac
