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

## Udacity Honor Code
As my implementation receives very low mean scores and I could not spot the problem, I reached out to the following resources to get help:

- https://knowledge.udacity.com/questions/172077
    As I tested different values of sigma for the noise, I got confident that 0.1 is an applicable value, that a batch size of 256 is appropriate, that hidden layers of 512x256 are appropriate and batch size can be 1e6.

- https://knowledge.udacity.com/questions/142820
    As I tried a MADDPG and DDPG-only implementation, I got confident that both ways are possible. As the author of this post also has tried a lot of hyperparameter settings, I got confident that I also have a different problem than hyperparameters within my implementation. I also cloned the mentioned repository https://github.com/HBD-BH/Unity_Tennis and looked into the provided solution and compared it to my code. I also tried out the hyperparameters of this solution in my implementation.

- https://knowledge.udacity.com/questions/119483
    As I also use separate models for the agents, I got confident that this is an appropriate way of solving this project. As I also tried to fill the replay buffer with values from random play (which in the beginning got more rewards than my implementation) for some episodes, I got confident that this is an appropriate way. I also got confident to collect random values for 1.000 episodes is an appropriate time. I also looked into the provided repository https://github.com/odellus/tennis and compared it to my code. I also tried out the hyperparameters of this solution in my implementation.

- https://knowledge.udacity.com/questions/101614
    In my implementation, I only manually initialised the last layer. In this post, the initialization of all layers is provided. I copied this to my code.

[1] https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet

[2] https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis

[3] https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet#getting-started

[4] https://github.com/udacity/deep-reinforcement-learning/tree/master/python

[5] https://stackoverflow.com/a/52269934/2988

[6] https://github.com/udacity/deep-reinforcement-learning

[7] https://github.com/ray-project/ray/tree/master/python/ray/tune