#!/usr/bin/env bash
# Download NIH ChestX-ray14 dataset.
#
# Usage:
#   bash scripts/download_data.sh [TARGET_DIR]
#
# TARGET_DIR defaults to data/NIH-ChestX-ray14.
# Requires: wget, md5sum (standard on Linux/macOS)
#
# NIH ChestX-ray14 is publicly available under a CC0 license.
# Direct download from the NIH Clinical Center:
#   https://nihcc.app.box.com/v/ChestXray-NIHCC

set -euo pipefail

TARGET="${1:-data/NIH-ChestX-ray14}"
mkdir -p "$TARGET/images"

echo "Downloading NIH ChestX-ray14 to $TARGET ..."

# Metadata files
BASE="https://nihcc.app.box.com/v/ChestXray-NIHCC/file"
wget -q --show-progress -O "$TARGET/Data_Entry_2017.csv" \
    "${BASE}/220040421496"
wget -q --show-progress -O "$TARGET/BBox_List_2017.csv" \
    "${BASE}/220040406914"
wget -q --show-progress -O "$TARGET/train_val_list.txt" \
    "${BASE}/220040387367"
wget -q --show-progress -O "$TARGET/test_list.txt" \
    "${BASE}/220040369671"

# Image archives (images_001.tar.gz … images_012.tar.gz)
IMAGE_BASE="https://nihcc.app.box.com/v/ChestXray-NIHCC/file"
declare -A ARCHIVES=(
    ["images_001.tar.gz"]="220040415065"
    ["images_002.tar.gz"]="220040479822"
    ["images_003.tar.gz"]="220040501367"
    ["images_004.tar.gz"]="220040573933"
    ["images_005.tar.gz"]="220040598814"
    ["images_006.tar.gz"]="220040620513"
    ["images_007.tar.gz"]="220040634571"
    ["images_008.tar.gz"]="220040646332"
    ["images_009.tar.gz"]="220040656504"
    ["images_010.tar.gz"]="220040660214"
    ["images_011.tar.gz"]="220040665426"
    ["images_012.tar.gz"]="220040681112"
)

for ARCHIVE in "${!ARCHIVES[@]}"; do
    FILE_ID="${ARCHIVES[$ARCHIVE]}"
    echo "Downloading $ARCHIVE ..."
    wget -q --show-progress -O "/tmp/$ARCHIVE" "${IMAGE_BASE}/${FILE_ID}"
    echo "Extracting $ARCHIVE ..."
    tar -xzf "/tmp/$ARCHIVE" -C "$TARGET/images" --strip-components=1
    rm "/tmp/$ARCHIVE"
done

echo "Done. Dataset at $TARGET"
echo "Image count: $(ls "$TARGET/images" | wc -l)"
