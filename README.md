
# Image Classifier for Flowers Species

In this Deep Learning project I have trained a Classifier to recognize 102 different flower categories.

The dataset used to train this model can be find [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

Here are some examples of the Flower images:

### Flower Images

![Flowers](assets/Flowers.png)

### Inference

![Inference](assets/inference_example.png)
## Tech Stack

**_Language:_** Python

**_Modelling:_** Pytorch

**_Other Libraries:_** PIL, Numpy, json, argparse
## Installation

Clone the project

```bash
  git clone https://github.com/Dua-Jan-Muhammad/Image-Classifier-using-Pytorch.git

```
Install dependencies

```bash
  pip install -r requirements.txt
```
OR

make environment from .yaml file
```
conda env create -f environment.yaml
```

## How to Run

`train.py` need to be run first and then `predict.py` which predicts the image.

To run `train.py` execute the following command
```bash
  python train.py --image_data_dir <path to the dataset file>
```
Other optional agruments can be passed, run` python train.py --help` to get arguments




To run `predict.py` execute the following command
```bash
  python predict.py --image_path <path to image for prediction>
```
other optional agruments can be passed, run` python predict.py --help` to get arguments




