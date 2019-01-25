#!/bin/bash

# setup imagenet images 
tar -xvf ILSVRC2014_DET_train.tgz
cp ./untar.sh ./ILSVRC2014_DET_train/ && cd ./ILSVRC2014_DET_train/
# delete ILSVRC2013* data to exactly match GOTURN training images 
rm -rf ILSVRC2013*
./untar.sh
rm -rf *.tar
rm untar.sh && cd ..

# setup imagenet bounding boxes
tar -xvf ILSVRC2014_DET_bbox_train.tgz

# setup alov images + bounding boxes
unzip alov300++_frames.zip
unzip alov300++GT_txtFiles.zip
