This is my solution for the project **Continuous Control**[1]

I choosed to solve `Option 1: Solve the First Version` with 1 agent.

**Goal**
> In the project **Continuous Controll** the unity **Reacher** [2] environment is given, in which a double-jointed arm can move to a target location. Every timestep the agent's hand is in the target, a reward of +0.1 is given. The goal is to reach mean reward of 30 over 100 consecutive episoded.

> The observation space describes the position, rotation, velocity and angular velocities, a total of 33 variables, of the arm.
> The action space describes the torque applicable to two joints as a vector with four numbers. Every value should be between -1 and 1.

> -- *this is my summary of the goal and enviroment details of the project, see [7] for the original introduction from udacity*

## Setup
- download the reacher environment from [3] section `Version 1: One(1) Agent` for your OS and unzip it in the folder *PATH_WHERE_YOU_CLONED_THIS_REPO_TO/p2_continous-control*
    - set filename of this file in `continuous_control.py` at bottom in line `env_filename=...`, e.g. `env_filename="Reacher.app"`
- download the `python/` folder from [4]
    - `cd PATH_WHERE_YOU_CLONED_THIS_REPO_TO/p2_continuous-control/`
    - `make install`
        - This will download only the `python` folder instead of cloning the whole repository. I copied this git command from [5], credits to the author (Ciro Santilli) for this solution.
            - Additionally, this will install all the dependencies from the `python` folder via `pip` as described in step 3 of [6]
            - Dependencies assume python version <= 3.6
        - Installs dependencies from `requirements.txt`

## Train
- `cd PATH_WHERE_YOU_CLONED_THIS_REPO_TO/p2_continuous-control/`
- `make train` starts training the agent
- `make tensorboard` shows results in tensorboard

## Hyperparameter tuning with tune
I used tune[8] for doing grid search for the hyperparameters on ranges I have given.
- `cd PATH_WHERE_YOU_CLONED_THIS_REPO_TO/p2_continuous-control/`
- adjust hyperparameter-ranges for grid search in `tune.py`
- adjust path to reacher environment (see Setup) in `tune.py` in line `self.env_filename = ...` relative to tune logdir, e.g. `self.env_filename = self.logdir + '../../../Reacher_Linux_NoVis/Reacher.x86_64'`
- `make tune` starts grid search

Some notes on my setup:

- I played around with various hyperparameter ranges in `tune.py`, the version in this commit is only the last version in a row
- I did not configure my setup properly and get a `'OSError: handle is closed' exception` during the whole process, which I did not investigate further. Important to mention that this is a problem of how I configured my whole machine, this is not a problem coming from tune.

## Udacity Honor Code
1. Some of the commits in this project are made by user `Ubuntu`, but it was me. I did some training of the project in AWS and pushed changes directly from E2C-Instance to github, but did not change the VM-Image's git username `Ubuntu` (and email) to my personal data.

2. As my rewards where low during training, I went to this Udacity Knowledge Base Article[9] which gave among others the following advice which I followed:

- use big replay buffer
- use soft update every 10-20 timesteps
- training is possible < 500 episodes
- reduce noise which is added to action over time

[1] https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control

[2] https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher

[3] https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control#getting-started

[4] https://github.com/udacity/deep-reinforcement-learning/tree/master/python

[5] https://stackoverflow.com/a/52269934/2988

[6] https://github.com/udacity/deep-reinforcement-learning

[7] https://github.com/udacity/deep-reinforcement-learning/blob/master/p2_continuous-control/README.md

[8] https://github.com/ray-project/ray/tree/master/python/ray/tune

[9] https://knowledge.udacity.com/questions/277763