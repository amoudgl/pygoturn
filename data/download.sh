#!/bin/bash
echo 'Downloading ImageNet training images...'
wget http://image-net.org/image/ilsvrc2014/ILSVRC2014_DET_train.tar
echo 'Downloading ImageNet training bounding boxes...'
wget http://image-net.org/image/ilsvrc2014/ILSVRC2014_DET_bbox_train.tgz
echo 'Downloading ALOV dataset...'
wget http://isis-data.science.uva.nl/alov/alov300++_frames.zip
wget http://isis-data.science.uva.nl/alov/alov300++GT_txtFiles.zip
echo 'Done'