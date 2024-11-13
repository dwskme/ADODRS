#!/bin/zsh

# =============================================================================
# Dataset downlader & extractor for bdd100k
# Author: dwskme
# Last Updated: 2024
# =============================================================================
#
# Download train set
curl https://dl.cv.ethz.ch/bdd100k/data/100k_images_train.zip -o 100k_images_train.zip
# Download val set
curl https://dl.cv.ethz.ch/bdd100k/data/100k_images_val.zip -o 100k_images_val.zip
# Download label set
curl https://dl.cv.ethz.ch/bdd100k/data/bdd100k_det_20_labels_trainval.zip -o bdd100k_det_20_labels_trainval.zip

# Unzip Trainset
unzip -q 100k_images_train.zip
# Unzip Valset
unzip -q 100k_images_val.zip

# Unzip Labels
unzip -q bdd100k_det_20_labels_trainval.zip

# Move dirs
mv bdd100k/labels/det_20/det_train.json bdd100k/labels/bdd100k_labels_images_train.json
mv bdd100k/labels/det_20/det_val.json bdd100k/labels/bdd100k_labels_images_val.json
rmdir bdd100k/labels/det_20
mkdir COCO
mkdir COCO/images
mkdir COCO/annotations
