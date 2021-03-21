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

## Code Layout

* `pretrain.py`: Using data to pretrain the model. 
* `fintune.py`: Using data to finetune the model.
* `test.py`: Computing perplexity values to evaluate the model.
* `data_pipeline.py`: Building data class and preparating data.
* `GPT_Model.py`: Model used in this system.
* `config.py`: Hyperparameters setting.

## Data Preparation

## Usage

- Pretrain the model on different data. A simple run would be something like:

```bash
python pretrain.py \
  -t poetry \  
  -n 1
```

In the above bash code, `-t` represents the data type of pretraining, `-t peotry` represents pretraining the model on the Classical Chinese poetry data. `-n` represents the number of experiment, `-n 1` represents the number of experiment is 1. After running the above bash code, pretrained models would be stored in `../Exp_1/poetry_pretrain`. All the possible options about `-t` and `-n` can be seen in the following python code:

```python
parser.add_argument("-t", "--type", choices = ['random_poetry', 'poetry', 
                                               'poetry_pos', 'poetry_tone',
                                                'random_music', 'music'])
parser.add_argument("-n", "--exp_number", choices = ['0', '1', '2', '3', '4'])
```

### Finetune

### Testing

## Results
