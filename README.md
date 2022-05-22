# Photomosaic Generator

This code can be used to create mosaics of images, where each subimage in the mosaic is a 32x32 pixel image from the 
Cifar-100 dataset. The included `/data` folder contains a `preprocessing.py` script used to generate the average RGB 
values for each Cifar-100 image. This is run on the fully extracted `cifar-100-python` folder generated from the python 
download at <https://www.cs.toronto.edu/~kriz/cifar.html>. A sample low resolution (256x256) `mario.jpg` image is 
included in the `/data` folder. This should be able to run in a few minutes on most laptops. Higher resolution images 
take considerably more time.

You can learn more about Cifar-100 at 
<https://www.cs.toronto.edu/~kriz/cifar.html>

# Installation and Use

1. Clone the repo
1. Install the repo requirements into your environment using `pip install -r requirements.txt` from the repo root
1. Download the Cifar-100 data from <https://www.cs.toronto.edu/~kriz/cifar.html>
1. Extract the Cifar-100 data into `/data`. This should look like a folder named `cifar-100-python` with a `train` file 
inside of it
1. Run `/data/preprocessing.py` to generate the average RGB csv file
1. Change the input and output file paths at the bottom of `create_mosaic.py`, or load CreateMosaic into another file
1. Run `create_mosaic.py` to generate mosaics
1. If you only want to use one of the coarse Cifar-100 classes in the mosaic, add the optional `coarse_class` argument 
to `CreateMosaic.create_mosaic()`. Valid classes are `"all"` or a single Superclass from 
<https://www.cs.toronto.edu/~kriz/cifar.html> 

# Running tests

Tests can be run by calling `pytest` at the project root.