#!/bin/bash

# First `uv run --all-extras mkdocs build`
# Then run this script.

rm -rf build;
mkdir build;
cd build;
git init -b gh-pages;
git remote add origin git@github.com:WiseLabCMU/red-rover.git;
cd ..;
cp -r site/* build/;
cd build;
touch .nojekyll;
git add --all;
git commit -m "Update gh-pages";
git push -f origin gh-pages;