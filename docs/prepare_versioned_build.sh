#!/bin/bash

print_help(){
  echo "Usage: bash" "$0"
  echo ""
  echo "  Small bash script to prepare the docs for a _versioned_ build."
  echo ""
  echo "  The environment variable BIOME_TEXT_DOC_VERSION must be set!"
  echo "  This env variable must match the release tag (for a new release) or 'master'."
}

if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    print_help
    exit 0
fi

if [ -z "$BIOME_TEXT_DOC_VERSION" ]; then
  echo "ERROR: BIOME_TEXT_DOC_VERSION not set!"
  print_help
  exit 1
fi


echo " - Modifying font urls  ..."

if ! sed -i "s|/biome-text/|/biome-text/$BIOME_TEXT_DOC_VERSION/|g" ./docs/.vuepress/theme/styles/fonts.styl; then
  echo "ERROR: Could not modify 'fonts.styl'!"
  exit 1
fi


if [ "$BIOME_TEXT_DOC_VERSION" != "master" ]; then
  echo " - Modifying tutorials ..."

  modified=$(find ./docs/documentation/tutorials -maxdepth 1 -name "*.ipynb" \
    -exec sed -i -e "s|pip install -U git+https://github.com/recognai/biome-text.git|pip install -U biome-text|g" \
      -e "s|/biome-text/master/|/biome-text/$BIOME_TEXT_DOC_VERSION/|g" \
      -e "s|/biome-text/blob/master/|/biome-text/blob/$BIOME_TEXT_DOC_VERSION/|g" {} \; \
    -exec echo {} \; | wc -l)
  if [ "$modified" -eq 0 ]; then
    echo "ERROR: No tutorials modified!"
    exit 1
  fi
fi


echo " - Done!"

exit 0
