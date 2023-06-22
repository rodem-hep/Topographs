# Topographs

Public repository for a minimum working example of the topographs project.
* Associated paper: [arxiv](https://arxiv.org/abs/2303.13937)

## Installation
1. Setup the environment.
    * You can setup your own environment using the requirements.txt file. The project was tested with python3.8
    * Alternatively you can build a docker image with the provided Dockerfile
    * You can also use the already built docker image at ???
2. Download the data from zenodo [data](https://zenodo.org/record/7737248)
    * Make sure to adjust the paths in the configuration files to point to the data.

## Usage
To train a model, simply run ```python train.py --config configs/config_full.yaml --log_dir log_dir```. This will train a model on complete events only saving all outputs in the directory ```log_dir``` which will be created.
Alternatively you can use ```configs/config_partial.yaml``` to train including partial events.

