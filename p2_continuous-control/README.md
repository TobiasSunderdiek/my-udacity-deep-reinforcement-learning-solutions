This is my solution for the project **Continuous Control**[1]

I choosed to solve `Option 1: Solve the First Version` with 1 agent.

**Goal**
> In the project **Continuous Controll** the unity **Reacher** [2] environment is given, in which a double-jointed arm can move to a target location. Every timestep the agent's hand is in the target, a reward of +0.1 is given. The goal is to reach mean reward of 30 over 100 consecutive episoded.

> The observation space describes the position, rotation, velocity and angular velocities, a total of 33 variables, of the arm.
> The action space describes the torque applicable to two joints as a vector with four numbers. Every value should be between -1 and 1.

> -- *this is my summary of the goal and enviroment details of the project, see [7] for the original introduction from udacity*

network structure like in ddpg paper https://arxiv.org/pdf/1509.02971.pdf section 7 Experiment Details
also other details:
batch_size = 64
replay_buffer_size = 10e6
gamma= 0.99
tau= 0.001
actor_learning_rate=10e-4
critic_learning_rate=10e-3
fill last layer of weights with uniform distribution /other layers are filled with pytorch defaults which is the same like in the paper
OrnsteinUhlenbeckActionNoise with same parameters like in the paper is added to the action to enable exploration

first I used replay buffer of size 10 Mio. as described in the paper, this takes a while to fill with only one agent
therefore I decreased the buffer size to 1 Mio. like used in the udacity example in [5]

todo
# todo gradient clipping like in the project description
# todo update network every x timestep, like in the project description
# todo do not do pip install . in cloned repo, instead move required libs to requirements.txt
check todos

## Setup
- download the reacher environment from [3] section `Version 1: One(1) Agent` for your OS and unzip it in the folder *PATH_WHERE_YOU_CLONED_THIS_REPO_TO/p2_continous-control*
    - set filename of this file in `continuous_control.py` in line `env = UnityEnvironment(file_name="INSERT-FILENAME")`, e.g. `env = UnityEnvironment(file_name="Reacher.app")`
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

[1] https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control

[2] https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher

[3] https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control#getting-started

[4] https://github.com/udacity/deep-reinforcement-learning/tree/master/python

[5] https://stackoverflow.com/a/52269934/2988

[6] https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal

[7] https://github.com/udacity/deep-reinforcement-learning/blob/master/p2_continuous-control/README.md