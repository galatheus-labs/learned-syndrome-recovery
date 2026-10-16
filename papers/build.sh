#!/usr/bin/env bash
# Build the four public PDFs from their checked-in TeX and figure inputs.
set -euo pipefail
paper_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
paper_build_dir="$(mktemp -d)"
trap 'rm -rf -- "$paper_build_dir"' EXIT
for paper_name in compression gauge qec fracton; do
  mkdir -p "$paper_build_dir/$paper_name"
  (
    cd "$paper_root/src/$paper_name"
    latexmk -pdf -interaction=nonstopmode -halt-on-error \
      -outdir="$paper_build_dir/$paper_name" main.tex
  )
  cp "$paper_build_dir/$paper_name/main.pdf" "$paper_root/$paper_name.pdf"
done
