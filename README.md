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
cd tum-adlr-ws22-04

# [OPTIONAL] create a virtual environment using conda or pyenv

# install requirements
pip install -r requirements.txt
```

## How to train

Define an environment function and a training curriculum. An example can be found under [experiments/example/](./experiments/example/). For a more detailed description of how to set up your own environments, see the experiments [README](./experiments/README.md). Once you are ready to train a model, simply start the training by calling the training script:

```yaml
# e.g.
python experiments/example/experiment1/train.py

```

For training in the cloud, we recommend using [tmux](https://github.com/tmux/tmux/wiki). Furthermore, when working with gcloud you can view TensorBoard logs and training videos locally by enabling port forwarding. For instance, if you want to forward the default port of TensorBoard and the default port of our [video server](#video-server) you can achieve this by appending "`-- -L 6006:localhost:6006 -L 8008:localhost:8008`" to the gcloud compute ssh command.

## How to run

There are two main entry points: `play.py` and `run.py`. Use `play.py` to simply observe the defined environment/swarm behavior. This does not require any trained model. You can move around using `W, A, S, D` on your keyboard.
On the other hand, use `run.py` to load a pre-trained model and observe its behavior in the defined environment. Movement is done automatically by the model.

The calling of these scripts is intended to enable the loading of predefined environments which were created during the experiments. In addition, their arguments are intended to be readily adjustable for various scenarios. In order to run the example environment with trajectories turned off, you would for example execute:

```yaml

# play around
python play.py experiments/example/env.py create_env "(trajectories=False)"

# run trained model
python run.py path/to/model.zip experiments/example/env.py create_env "(trajectories=False)"
```

For more details, check out the help of the `play.py`, i.e. `python play.py -h`, or the `run.py` respectively.

## Additional Tools

### Video Server
We provide a video server in order to remotely access training videos. To start it, simply run

```yaml
python video_server.py
```

### Drag Tool
We provide a drag tool found in the `drag.py` which allows analyzing the agent's behavior using a trained model. You can move around every object with `left mouse button` and change its velocity with `right mouse button` which allows you to interactively create any scenario you want.

```yaml
#e.g.
python drag.py path/to/model.zip experiments/example/env.py create_env "()"
```

### Video Recording
In case you want to record videos of the execution of a model, use the `record.py`. For details, look at the help `python record.py -h`.
```yaml
#e.g.
python record.py path/to/model.zip experiments/example/env.py create_env "()" videos/example.mp4 1
```


## How it's build
Next to the scripts for running trained models, the repository is split into

- `analysis`: Contains helper functions for creating policy, reward, or trajectory plots.
- `bas`: BAS for **B**lueprint, **A**gent, **S**warm contains the entire swarm implementation as well as building blocks for creating environments including the blueprint, agent, and various wrappers. See bas [README](./bas/README.md) for more details.
- `experiments`: Contains the scripts for the concrete environments and trainings used for the experiment conduction as well as scripts and notebooks for analyzing certain configurations and visualizing results. See experiments [README](./experiments/README.md) for more details.
- `training`: Contains helper functions for training and logging.
