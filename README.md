# Poetry_Music_Transformer

## Installation

We use a single GPU (RTX 2080 Ti) to develop this system with:
- anaconda 3 (python 3.7)
- tensorflow-gpu 2.4.1
- tensorboardX 2.1
- tf-slim 1.1.0

If you have a GPU, have installed the Display Driver, and have installed Anaconda 3, you can use the following bash code to create conda environment and install all the packages needed in this system:
```bash
conda create -n music_poetry_transformer python=3.7
source activate music_poetry_transformer
conda install tensorflow-gpu 2.4.1
pip install tf_slim 1.1.0
pip install tensorboardX 2.1
```

## Data Preparation

## Code Layout

* `pretrain.py`: Using data to pretrain the model. 
* `fintune.py`: Using data to finetune the model.
* `test.py`: Computing perplexity values to evaluate the model.
* `data_pipeline.py`: Building data class and preparating data.
* `GPT_Model.py`: Model used in this system.
* `config.py`: Hyperparameters setting.

## Usage



## Results
