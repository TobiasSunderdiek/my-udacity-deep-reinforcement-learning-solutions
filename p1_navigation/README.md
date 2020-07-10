This is my solution for the project **Navigation**[1]

## Goal
> In the project **Navigation** a unity environment is given, in which yellow and blue bananas are placed. These bananas can be collected, for every yellow banana a positive reward of +1 is given, for every blue banana a negative reward of -1. The goal is to create an agent which can get a reward of +13 over 100 consecutive episodes.

> The agent can navigate in the environment in the following discrete action space:
> - `0` moves forward
> - `1` moves backward
> - `2` turns left
> - `3` turns right

> The state space has 37 dimensions containing the agent's velocity and ray-based perception of objects around the forward direction of the agent.

> -- *this is my summary of the goal and enviroment details of the project, see [7] for the original introduction from udacity*

## Setup
- download the Banana.zip-File from the sources provided in [2] for your OS and unzip it in the folder *PATH_WHERE_YOU_CLONED_THIS_REPO_TO/p1_navigation/*
    - set filename of this file in `navigation.py` in line `env = UnityEnvironment(file_name="INSERT-FILENAME")`, e.g. `env = UnityEnvironment(file_name="Banana.app")`
- download the `python/` folder from [3]
    - `cd PATH_WHERE_YOU_CLONED_THIS_REPO_TO/p1_navigation/`
    - `make install`
        - This will download only the `python` folder instead of cloning the whole repository. I copied this git command from [4], credits to the author (Ciro Santilli) for this solution.
            - Additionally, this will install all the dependencies from the `python` folder via `pip` as described in step 3 of [5]
            - Dependencies assume python version <= 3.6
        - Installs dependencies from `requirements.txt`

## Train
`make train` in *PATH_WHERE_YOU_CLONED_THIS_REPO_TO/p1_navigation/* starts training the agent

[1] https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation

[2] https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation#getting-started

[3] https://github.com/udacity/deep-reinforcement-learning/tree/master/python

[4] https://stackoverflow.com/a/52269934/2988

[5] https://github.com/udacity/deep-reinforcement-learning#dependencies

[7] https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md