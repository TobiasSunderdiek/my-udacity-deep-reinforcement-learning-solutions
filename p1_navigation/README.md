This is my solution for the project **Navigation**[1]

## Setup
- download the Banana.zip-File from the sources provided in [2] for your OS and unzip it in the folder *PATH_WHERE_YOU_CLONED_THIS_REPO_TO/p1_navigation/*
    - set filename of this file in `navigation.py` in line `env = UnityEnvironment(file_name="INSERT-FILENAME")`, e.g. `env = UnityEnvironment(file_name="Banana.app")`
- download the `python/` folder from [3]
    - `cd PATH_WHERE_YOU_CLONED_THIS_REPO_TO/p1_navigation/`
    - `make install`
        - This will download only the `python` folder instead of cloning the whole repository. I copied this git command from [4], credits to the author (Ciro Santilli) for this solution.
        - Additionally, this will install all the dependencies from the `python` folder via `pip` as described in step 3 of [5]
        - Dependencies assume python version <= 3.6

[1] https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation

[2] https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation#getting-started

[3] https://github.com/udacity/deep-reinforcement-learning/tree/master/python

[4] https://stackoverflow.com/a/52269934/2988

[5] https://github.com/udacity/deep-reinforcement-learning#dependencies