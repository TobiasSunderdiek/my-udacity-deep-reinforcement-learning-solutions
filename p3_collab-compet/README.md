This is my solution for the project **Collaboration and Competition**[1]

**Goal**
> In the project **Collaboration and Competition** the unity **Tennis** [2] environment is given, in which two agents control rackets to play tennis. Every time an agent hits the ball over the net, a reward of +0.1 is given, every time the ball hits the ground or is out of bounds, a reward of -0.01 is given. After each episode, the score of the agent which reached the highest score is added to the total rewards. The goal is to reach mean reward of +0.5 over 100 consecutive episodes.

> The observation space describes the position and velocity of ball and the agent's racket. Every agent gets his own observation space with his own racket, a total of 8 variables.
> There are two actions in the continuous action space available, one describes the movement towards or away from the net and the other is jumping.

> -- *this is my summary of the goal and enviroment details of the project, see [1] for the original introduction from udacity*

## Setup
- download the tennis environment from [3] for your OS and unzip it in the folder *PATH_WHERE_YOU_CLONED_THIS_REPO_TO/p3_collab-compet*
    - set filename of this file in `collab-competition.py` at bottom in line `env_filename=...`, e.g. `env_filename='Tennis.app'`
- download the `python/` folder from [4]
    - `cd PATH_WHERE_YOU_CLONED_THIS_REPO_TO/p3_collab-compet/`
    - `make install`
        - This will download only the `python` folder instead of cloning the whole repository. I copied this git command from [5], credits to the author (Ciro Santilli) for this solution.
            - Additionally, this will install all the dependencies from the `python` folder via `pip` as described in step 3 of [6]
            - Dependencies assume python version <= 3.6
        - Installs dependencies from `requirements.txt`

## Train
- `cd PATH_WHERE_YOU_CLONED_THIS_REPO_TO/p3_collab-compet/`
- `make train` starts training the agent
- `make tensorboard` shows results in tensorboard

## Hyperparameter tuning with tune todo
I used tune[7] for doing grid search for the hyperparameters on ranges I have given.
- `cd PATH_WHERE_YOU_CLONED_THIS_REPO_TO/p3_collab-compet/`
- adjust hyperparameter-ranges for grid search in `tune.py`
- adjust path to tennis environment (see Setup) in `tune.py` in line `self.env_filename = ...` relative to tune logdir, e.g. `self.env_filename = self.logdir + '../../../Tennis.app'`
- `make tune` starts grid search

## Saved model weights
Saved model weights of successfully trained agent are provided in `model.pth`

[1] https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet

[2] https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis

[3] https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet#getting-started

[4] https://github.com/udacity/deep-reinforcement-learning/tree/master/python

[5] https://stackoverflow.com/a/52269934/2988

[6] https://github.com/udacity/deep-reinforcement-learning

[7] https://github.com/ray-project/ray/tree/master/python/ray/tune