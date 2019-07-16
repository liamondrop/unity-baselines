# Unity Baselines

> Make the Unity Gym Env easily trainable using OpenAI Baselines [WIP]

The [Unity Gym](https://github.com/Unity-Technologies/ml-agents/tree/master/gym-unity) environment attempts to bridge the gap between the Unity ML-Agents python interface for training reinforcement learning and the standard OpenAI-style Gym interface. This repo goes a step further and adds a few missing pieces so that ML-Agents environments can be trained using OpenAI Baselines algorithms.

This repository contains:

1. Primarily, [a script](run.py) for running various OpenAI algorithms with its various supported network architectures.
2. An example configuration file to take the place of the unwieldy litany of command-line arguments needed to run OpenAI baselines/Unity Gym/etc.

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [Config](#config)
- [Maintainers](#maintainers)
- [Contributing](#contributing)

## Install

First, follow the OpenAI baselines [Prerequisites](https://github.com/openai/baselines#prerequisites) and [Installation](https://github.com/openai/baselines#installation) instructions.

Then, install additional requirements with pip.

```
pip install -r requirements.txt
```

## Usage

To run, call `run.py` with the path to a configuration file as the first argument.

```
python run.py config.yaml
```

See `example-config.yaml` for an example configuration. At a minimum, the configuration should include the `env` argument, which should be a path to an ML-Agents-enabled Unity executable **without** the extension (i.e. `.x86_64`)

You can also add additional arguments after the configuration file.

```
python run.py config.yaml --gamma 0.999 --play
```

## Maintainers

[@liamondrop](https://github.com/liamondrop).

## Contributing

Feel free to dive in! [Open an issue](https://github.com/liamondrop/unity-baselines/issues/new) or submit PRs.

Unity Baselines follows the [Contributor Covenant](http://contributor-covenant.org/version/1/3/0/) Code of Conduct.
