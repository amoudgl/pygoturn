# PyTorch GOTURN tracker
This is the PyTorch implementation of GOTURN visual tracker. 

![](images/pygoturn_man.gif) 

If you find this code useful in your research, please cite:
```
@inproceedings{held2016learning,
  title={Learning to Track at 100 FPS with Deep Regression Networks},
  author={Held, David and Thrun, Sebastian and Savarese, Silvio},
  booktitle={European Conference Computer Vision (ECCV)},
  year={2016}
}
```

### Why PyTorch implementation? 
GOTURN is one of the key trackers which proposed an alternative deep learning approach to object tracking by learning a comparator function. Although author's original C++ Caffe implementation and this Python Caffe implementation are well-documented, I feel a PyTorch implementation would be more readable and much easier to adapt for further research. Hence, this is my humble attempt to reproduce GOTURN from scratch in PyTorch which includes data loading, training and inference. I hope this is a useful contribution to the vision community.

## Dependencies

PyTorch 1.0 and Python3 recommended. 

Following python packages are required to run the code.
```
scikit_image
numpy
torch
torchvision
matplotlib
Pillow
```
To install all the packages, do `pip3 install -r requirements.txt`.

## Inference 
You can test pretrained PyTorch GOTURN on a new sequence by following the steps below: 

- [Download]() PyTorch pretrained model.
- Run `src/test.py` and provide path to sequence as argument 
 
You can test pre-trained

## Training 

- Download ImageNet and ALOV datasets in `data/` directory. 
- Run `src/train.py`. 

