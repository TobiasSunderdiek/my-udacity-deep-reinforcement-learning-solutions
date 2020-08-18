This is my solution for the project **Continuous Control**[1]

I choosed to solve `Option 1: Solve the First Version` with 1 agent.

## Setup
- download the reacher environment from [2] section `Version 1: One(1) Agent` for your OS and unzip it in the folder *PATH_WHERE_YOU_CLONED_THIS_REPO_TO/p2_continous-control*
    - set filename of this file in `continuous_control.py` in line `env = UnityEnvironment(file_name="INSERT-FILENAME")`, e.g. `env = UnityEnvironment(file_name="Reacher.app")`
- download the `python/` folder from [3]
    - `cd PATH_WHERE_YOU_CLONED_THIS_REPO_TO/p2_continuous-control/`
    - `make install`
        - This will download only the `python` folder instead of cloning the whole repository. I copied this git command from [4], credits to the author (Ciro Santilli) for this solution.
            - Additionally, this will install all the dependencies from the `python` folder via `pip` as described in step 3 of [5]
            - Dependencies assume python version <= 3.6
        - Installs dependencies from `requirements.txt`

[1] https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control

[2] https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control#getting-started

[3] https://github.com/udacity/deep-reinforcement-learning/tree/master/python

[4] https://stackoverflow.com/a/52269934/2988