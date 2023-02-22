<div align="center">

# Robot Motion Planning using Deep Reinforcement Learning in Dynamic Swarm-Based Environments

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://gymnasium.farama.org/#"><img alt="Gymnasium" src="https://img.shields.io/badge/Gymnasium-ffffff?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAGdElEQVRo3u3aa4xdVRUH8N%2B982BmatvpTK1khhYq0FYgZtCUmpZIgiBJqyIhQUuCRUhULErSojESSWxA1IgiCdEPftCYAj6wRtHo2GhKolZNxapQbdFShxa0AYyiIo%2FWD2tdeu%2Bde%2B49dzrIF%2F7Jycycvfba%2F%2F1Yr32Gl%2FHSonKcfU%2FEsvx5BFPYiycK%2Boyk%2FGJU8VjKP4aj%2F68JDGADrsMgfp%2FEe3AqVuAgbsG92eet%2BAjG8Qf8Cc%2FnRM7C07gNX8nfXzSszsHvTKLVFjJ9eAMm8WA%2Bk%2Fmur4V8NXXdmbrXvFjkr8YBrCwhW8G1OJTPtcrt9soc4%2BrZJn%2BpOKuLSsp%2FFLswms%2BufFcGi3KsS2eL%2FBgexckl5d%2BBBzC37t3cfHdZSR0n55hjszGBu7Gxy8kumQVSG3Ps48Ko8CiDJeW34X1t2q9JmTIYFPYzejwTuBh3lZRdJrxIXxuZvpRZVlLnXcmhENUOClZhR8nBNuNWPNtG5ll8FptK6tyRHGY8gQWKo2o9%2BrAW3ygh%2B3WsQ28J2SeSQyHaKani33iTOI%2F%2FEO7tIfy3SfYU%2FB2HS5A6nLJLsa%2Bp7QScJo7YPBE4n0ouR8pOoAfrcWMqHEwlC%2FBa4UUmcXMdgbOxswT5GnZmn1r%2F03ED3iwM97d4EpfgPyIV2SJs4vl2iufjJ0lwBeaY7gmG8R48LHKXYdwkPExZvD%2F7DKeOh1PncJ3MaI49J7lM4sfJsSXmYLdIuupt4ybc0UJ%2BAB8TidwvdPAWTXh79plKHQMtZO7IsWuoJrfdybUBFWFcN5ueswyJrb6ogMwE%2FiqMuCzWZp%2BJgvaLcsyhFjw%2FkVwbeK7B7xT78OUiihZlil%2FUmCb0CCN8VT5zNe7qZfhCga41Odbygva%2B5PoCl4rwtxd2WLUJEZU3mL5LHxZbvkUY6CP5PCQC10FxXHak7G35s3l1N6Rs0c7UcGHqqhAeYEr7CFrDWBL8rihOhkSqPIXH8SHhqeblLtTvyAKRLm8RbnQf3pnjjqXOncrlSn25QKfDx5XPTwjXu1HEhUPCi6zBfp0Do5TZn6t4r%2FBATwrPVCa41bAtuXtQ%2BVydOM%2Ffxs9qKyC2cq8ITp2wNGVrx3Al9ogj2N8FjxvwQDVJlImgRPH%2BS%2BHK3uhYIDoqgsyVJXRcKcrHWhH%2FKxHUhvAjvKIkl8O1BTyA60t0WCiMckNB%2B7iwhYE2OgZSZrxFW0UciZ3Kpe%2FXi%2BPndmztINyfK9Up2n6tg8w1KVOEishWv6mzPW3F5%2BF1wqh62gh%2FEl%2FWuTBfLAx7pEXbSLYt7qCjF%2Ffh3W1kepLz2XKmuxTn3aeKbS97NjeLvKV%2BQXrz3eaSOk7KyQ4XtK9Kzi%2Fs0lr8XOttu0c546yhmn1uFztWyd%2FvUc7N1vApjblQDT3JtSF1qeYKNVdKC0VkbGeYrTAgjsGt%2BeyYgY5FIlid0PR%2BU3KdthgjIuxfXvfuXfhSlwPXT%2F7xfBbOUMd9oqip4fLkOFLUYVwEmRvFud0mioxuUMF5%2BLO4H70lfz9P93exHxDZZ69Iu%2Fdq7YIbMA%2Ffwv2K73haoSqi6nb8E%2BfUta0WZej2lClrC6vEhdj9yWleN6t4gaiJ94hUY0IkZP25IgMiVT5XGNw%2B%2FBrvFQGvHn25GOuTzL7sc27qGEid%2FTnGRI75R5H4na9g99ptaQ%2F%2BgrfhLaLIOEljwvV0kv0eviMi45L8%2B6wmXQfwGlFfn5J614kivt7AnxPG%2B0P8VHiwM8qufPPk9uOVde%2BqjhX6%2FQULsDh3rR69wpsNtZCvpK7B1F1%2FvM4QpWch2qWvR0UAW%2BJYsnfE9CuVZjwlatb6q5DaCj9TMM4zBbpWiGNUiE7GNKm7Wpcw4KrG2%2BnxXITnutR1Mb7fZZ8GnCYMrqfLfttFul3DRpGkdYNZudytiHuibj82rHcs66yK%2BqFTnduM63TOkkthuTDAsl9nCGN9RHibCzQlXiWwVLjdE2djAsSt2W8UZ4etcIWoIQ6IlL0sFolje8lskSeO0ibh88sehTFxu7xba%2FfZCufkhK%2BaTfL1OF8Eq604U2vjno8PiiNwFT6TE1%2BnddFeFf7%2BbpGordYFZvqh%2B4okOT%2FJPSqC0KvFuf2BSL4OZp%2FX49PCr%2B%2FJPkfEd7Mz8S98Dl%2FVOc4c9wTq%2B44Kg1sovr5MiSNQ9LV9gXAKtX81OCQyzL%2BZ4b8avIyXGv8D4JVZRvGBQW4AAAAASUVORK5CYII%3D"></a>
<a href="https://stable-baselines3.readthedocs.io/en/master/#"><img alt="StableBaselines3" src="https://img.shields.io/badge/StableBaselines3-b0b3e6"></a>
</div>

## Description
This is the code to implement dynamic swarm-based environments and enable robot motion planning using deep reinforcement learning in various such scenarios.

Authors: Jakob Taube (concept, programming), Leon Blumenthal (concept, programming), Lennart Röstel (supervision)

## Setup
Install dependencies:
```yaml
# clone project
git clone git@github.com:leonblumenthal/tum-adlr-ws22-04.git
cd tum-adlr-ws22-04.git

# [OPTIONAL] create virtual environment using conda or pyenv

# install requirements
pip install -r requirements.txt
```

## How to run

Select an image dataset (mnist, cifar) and model type (memae or ae)
```yaml
# default
python main.py model=memae datamodule=mnist
```

Or select a time series dataset (ecg5000, ucranomaly, yahooa1, yahooa2)
```yaml
# e.g.
python main.py model=ae datamodule=yahooa2 datamodule.id=3
```
Note that ucranomaly and yahoo are archives of multiple anomaly datasets. Therefore, a concrete dataset has to be chosen via the `datamodule.id`. To learn more about training configuration see [Configure Training](#configure-training).

### Pretrain a model via MAML

Define a meta data set in the configuration files (e.g. "example1.yaml") and train a specific model (currently only for ae-rolling-window or memae-rolling-window on time series data)
```yaml
# e.g.
python maml.py model=memae-rolling-window datamodule/meta_set=example1

```
For a better understanding of how to setup your MAML see [Configure MAML Pretraining](#configure-maml-pretraining).

## How to configure

### Configure Training
For a more specific setup the model training is highly configurable via the hydra configuration files. They can be found in the `configs` folder. These configuration files define different parameters that can be overwritten or extended via the configuration files or via the command line directly at any time. Hydra then builds a dictionary-like config starting in the `config.yaml` as root and expanding it with subconfigurations as specified.
The folder structure mirrors this dictionary:
- `configs`:
    - `datamodule`:
        - ... : specifies which data to train on and provides a default configuration of what parameters to use for the lightning datamodule
        - `eval_metric`:
            - ... : evaluation metrics for the performance of a model
    - `datamodule_model`:
        - ... : extends the model config by datamodule specific parameters
        - `model_config`:
            - `ae-rolling-window.yaml`: several time series datamodules share common model parameters which are defined here
    - `hydra`:
        - `default.yaml`: defines the hydra run directory
    - `model`:
        - ... : specifies the model (ae or memae)
    - `trainer`: 
        - `default.yaml`: parameters for the lightning trainer
    - `config.yaml`: main entry point of configs

Checkout the configuration files to see all parameters in detail.

### Configure MAML Pretraining
Configurations for MAML pretraining can be found in `maml_configs`. The folder structure is very similar to the `configs` but does not include `datamodule_model` since it is trained on multiple datasets instead of one. Consequently, the model architecture cannot be inferred by the datamodule but must be explicitly defined in `model`. Similarly, the used datasets must be explicitly provided. This can be achieved by providing python classes with their arguments for instantiation in a configuration file under `datamodule/meta_set`.

## How it's build
The main entry point is the `run.py`. It loads the specified model and passes it to a training function which is defined in `setup_and_train.py`. The latter combines all components of this project to setup a model and datamodule, execute the training loop, and save the corresponding logs. The different components of this project are split into separate folders. Overview of the folder structure:

- `analysis`: contains scripts and notebooks to analyse certain configurations and visualize results
- `configs`: hydra configurations for training (see [How to configure](#how-to-configure))
- `configs_maml`: hydra configurations for pretraining with MAML
- `data`: contains raw datasets
- `datamodules`: pytorch lightning datamodules for given datasets to automate data loading
- `logs`: logger outputs of runs and model checkpoints if applicable
- `model_transformers`: model transformers (mainly used for memory transformations on memae)
- `models`: model architectures (ae and memae)
- `modules`: memory module for memae
- `systems`: pytorch lightning modules to automate train and test loop

The entry point for pretraining with MAML is the `maml.py`. It executes the MAML training on the given model but can also automatically create a non-MAML twin of the model. That means it trains a second model of the exact same architecture as the first and on the exact same samples but in a regular/non-MAML fashion.
<br>
