# Poetry_Music_Transformer

## Install

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
