# PyTorch GOTURN tracker

This is the PyTorch implementation of GOTURN visual tracker (Held. et. al, ECCV 2016). GOTURN is one of the key trackers which proposed an alternative deep learning approach to object tracking by learning a comparator function.

![](images/goturn.png)

### Why PyTorch implementation?
Although author's original [C++ Caffe implementation](https://github.com/davheld/GOTURN) and [this](https://github.com/nrupatunga/PY-GOTURN) Python Caffe implementation are well-documented, I feel a PyTorch implementation would be more readable and much easier to adapt for further research. Hence, this is my humble attempt to reproduce GOTURN from scratch in PyTorch which includes data loading, training and inference. I hope this is a useful contribution to the vision community.

## Highlights

* Supports **PyTorch 1.0 and Python3**.
* **Reproduces GOTURN** end to end in PyTorch including training and inference.
* Provides **pretrained PyTorch GOTURN** model.
* **Fast:** Tracks target objects at 100+ fps.
* **Benchmark:** Evaluation on OTB50 and OTB100.

## Output

![](images/pygoturn_output.gif)

## Environment

PyTorch 1.0 and Python3 recommended.

```
numpy==1.14.5
torch==1.0.0
opencv-python==4.0.0.21
torchvision==0.2.1
tensorboardX==1.6
```
To install all the packages, do `pip3 install -r requirements.txt`.

## Demo

### [Download pretrained model](https://drive.google.com/file/d/1szpx3J-hfSrBEi_bze3d0PjSfQwNij7X/view?usp=sharing)

Navigate to `pygoturn/src` and do:

```
python3 demo.py -w /path/to/pretrained/model
```

Images with bounding box predictions will be saved in `pygoturn/result` directory.

Arguments:

`-w / --model-weights`: Path to a PyTorch pretrained model checkpoint.   
`-d / --data-directory`: Path to a tracking sequence which follows [OTB format](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html).   
`-s / --save-directory`: Directory to save sequence images with predicted bounding boxes.   

## Benchmark

To evaluate PyTorchGOTURN on OTB50 and OTB100, follow the steps below:

- Install [got10k toolkit](https://github.com/got-10k/toolkit).
     ```
     pip install --upgrade got10k
     ```
- [Download pretrained model](https://drive.google.com/file/d/1szpx3J-hfSrBEi_bze3d0PjSfQwNij7X/view?usp=sharing).
- Edit OTB dataset path and model path appropriately in `src/evaluate.py`. The script will automatically download OTB dataset at the path provided.
- Run evaluation script:
    ```
    python3 evaluate.py
    ```

## Performance

| Dataset | AUC | Precision  |
| -------|:--------:| -----:|
| OTB50  | 0.401 | 0.548 |
| OTB100 | 0.405 |  0.550 |

As per [foolwood/benchmark_results](https://github.com/foolwood/benchmark_results), the original Caffe GOTURN yields AUC: 0.427 and Precision: 0.572 on OTB100. I feel this minor difference in performance is due to difference in the way ImageNet models are trained in Caffe and PyTorch like input normalization, layer specific learning rates etc. In this repository, I followed exact GOTURN hyperparameters which may not be the best for PyTorch. I feel with some hyperparameter tuning, GOTURN performance can be reproduced with an end-to-end PyTorch model.

Feel free to contribute to this project, if you have any improvements!


## Fast inference

In order to benchmark results for a tracking sequence or do fast inference, run the following command:
```
python3 test.py -w ../checkpoints/pytorch_goturn.pth.tar -d ../data/OTB/Man
```

**Arguments:**

`-w / --model-weights`: Path to a PyTorch pretrained model checkpoint.  
`-d / --data-directory`: Path to a tracking sequence which follows [OTB format](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html).       

## Training

Please follow the steps below for data preparation and training a pygoturn model from scratch.

![](images/pygoturn_loss.jpg)

### Prepare training data
Navidate to `pygoturn/data`.

Either use `download.sh` script to automatically download all datasets or manually download them from the links below in `pygoturn/data`:
- [ILSVRC2014_DET_train.tar](http://image-net.org/image/ilsvrc2014/ILSVRC2014_DET_train.tar) (47GB)
- [ILSVRC2014_DET_bbox_train.tgz](http://image-net.org/image/ilsvrc2014/ILSVRC2014_DET_bbox_train.tgz) (15MB)
- [alov300++_frames.zip](http://isis-data.science.uva.nl/alov/alov300++_frames.zip) (10GB)
- [alov300++GT_txtFiles.zip](http://isis-data.science.uva.nl/alov/alov300++GT_txtFiles.zip) (300KB)

Once you have all the above files in `pygoturn/data`, use `pygoturn/data/setup.sh` script to setup datasets in the way pygoturn training script `/src/train.py` expects OR follow the manual steps below:

- Untar `ILSVRC2014_DET_train.tar`. You'll have a directory `ILSVRC2014_DET_train` containing multiple tar files.
- First, delete all the tar files in `ILSVRC2014_DET_train` directory which start with name `ILSVRC2013`. This is an important step to reproduce the *exact* same number of ImageNet training samples (239283) as described in GOTURN paper.
- Untar all the remaining tar files in `ILSVRC2014_DET_train`. When done, *delete* all `*.tar` files. Since there are several `tar` files to untar, you can use `data/untar.sh` script. Just copy `untar.sh` to `ILSVRC2014_DET_train` directory and do: `./untar.sh`. Delete `untar.sh` from `data/ILSVRC2014_DET_train` when you are done.
- Untar `ILSVRC2014_DET_bbox_train.tgz`.
- Unzip `alov300++_frames.zip` and `alov300++GT_txtFiles.zip`.

Once you finish data preparation, make sure that you have the following directories:

```
data/ILSVRC2014_DET_train
data/ILSVRC2014_DET_bbox_train
data/imagedata++
data/alov300++_rectangleAnnotation_full
```

### Kick off training!
Navigate to `pygoturn/src` and run the following command to train GOTURN with default parameters:

```
python3 train.py
```
All the parameters for GOTURN training can be passed as arguments. View `pygoturn/src/train.py` for more details regarding arguments.


## Citation

If you find this code useful in your research, please cite:

```
@inproceedings{held2016learning,
  title={Learning to Track at 100 FPS with Deep Regression Networks},
  author={Held, David and Thrun, Sebastian and Savarese, Silvio},
  booktitle={European Conference Computer Vision (ECCV)},
  year={2016}
}
```

## Acknowledgements

- I'd like to thank the original authors for releasing a clean C++ implementation [[davheld/GOTURN](https://github.com/davheld/GOTURN)] and it was heavily referenced to tune hyperparameters appropriately.   
- This python caffe implementation [[nrupatunga/PY-GOTURN](https://github.com/nrupatunga/PY-GOTURN)] was pretty useful to understand GOTURN batch formation procedure. I borrowed some of its parts and adapted it to Pytorch.


## License

MIT
