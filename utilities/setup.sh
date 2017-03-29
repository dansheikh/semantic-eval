#!/usr/bin/env bash

command -v curl >/dev/null 2>&1 || { echo "Program 'curl' required. Install program and retry." >&2; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo "Program 'unzip' required. Install program and retry." >&2; exit 1; }

echo 'Setup initiated.'

embeds_zip="https://drive.google.com/uc?export=download&id=0B7r5RtYVraYyMFQtd24tVHJ3blE"
semeval_zip="https://drive.google.com/uc?export=download&id=0B7r5RtYVraYyd3pFX2lIeE4zcms"
urls=($embeds_zip $semeval_zip)
zipfiles=('embeds.zip' 'semeval.zip')
dirs=('embeds' 'data' 'checkpoints' 'logs/tensorflow')

# Download zip files.
((url_cnt=${#urls[@]} - 1))
for idx in $(seq 0 $url_cnt); do
    echo "Fetching ${names[$idx]} from ${urls[$idx]}"
    curl -o ${zipfiles[$idx]} -Ls ${urls[$idx]} &
done
wait

# Make target directories.
((dir_cnt=${#dirs[@]} - 1))
for idx in $(seq 0 $dir_cnt); do
    echo "Making directory ${dirs[$idx]}"
    mkdir -p ${dirs[$idx]}
done

# Unzip files.
((zip_cnt=${#zipfiles[@]} - 1))
for idx in $(seq 0 $zip_cnt); do
    echo "Unzipping ${zipfiles[$idx]}"
    unzip ${zipfiles[$idx]} -d ${dirs[$idx]} &
done
wait

echo 'Setup complete.'
