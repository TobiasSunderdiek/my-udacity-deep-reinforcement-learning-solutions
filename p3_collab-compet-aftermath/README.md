This is a second solution for the project **Collaboration and Competition**[1]. Why a second solution? I created this project `p3_collab-compet-aftermath` after graduation of the nanodegree because I want to find out why a former implementation from me did not work. See [9] for more details which parts of my implementation I therefore changed in `p3_collab-compet`. See [this aftermath section](#aftermath-review) in this implementation where I tracked the error.

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

## Hyperparameter tuning with tune
start tune[7] for doing grid search for hyperparameters by
- `cd PATH_WHERE_YOU_CLONED_THIS_REPO_TO/p3_collab-compet/`
- adjust hyperparameter-ranges for grid search in `tune.py`
- adjust path to tennis environment (see Setup) in `tune.py` in line `self.env_filename = ...` relative to tune logdir, e.g. `self.env_filename = self.logdir + '../../../Tennis.app'`
- `make tune` starts grid search

Some notes on my setup:

- I played around with various hyperparameter ranges in tune.py, the version in this commit is only the last version in a row
- I did not configure my setup properly and get a 'OSError: handle is closed' exception during the whole process, which I did not investigate further. Important to mention that this is a problem of how I configured my whole machine, this is not a problem coming from tune. Maybe it is related to the problem described in [8].

## Saved model weights
Saved model weights of successfully trained agent are provided in `model.pth`

## Udacity Honor Code
I used the MADDPG-Lab implementation of the Physical Deception Problem, which is not public available (Udacity course material) as a blueprint for this project and got parts of `multi_agent.py` from there. To not just copy the parts, I had a look at the provided lab-code, tried to implement parts of it from my mind and compared it with the provided lab-code. To correct my errors during this process, I did some loops of comparing with the lab-implementation and changing my code.

As my implementation receives very low mean scores with different hyperparameters, and I could not spot the problem, I reached out to the following resources to get help:

- https://knowledge.udacity.com/questions/172077
    As I tested different values of sigma for the noise, I got confident that 0.1 is an applicable value, that a batch size of 256 is appropriate, that hidden layers of 512x256 are appropriate and batch size can be 1e6.

- https://knowledge.udacity.com/questions/142820
    As I tried a MADDPG and DDPG-only implementation, I got confident that both ways are possible. As the author of this post also has tried a lot of hyperparameter settings, I got confident that I also have a different problem than hyperparameters within my implementation. I also cloned the mentioned repository https://github.com/HBD-BH/Unity_Tennis and looked into the provided solution and compared it to my code. I also tried out the hyperparameters of this solution in my implementation. Credits to the author (Bernhard H.) for this solution.

- https://knowledge.udacity.com/questions/119483
    As I also use separate models for the agents, I got confident that this is an appropriate way of solving this project. As I also tried to fill the replay buffer with values from random play (which in the beginning got more rewards than my implementation) for some episodes, I got confident that this is an appropriate way. I also got confident to collect random values for 1.000 episodes is an appropriate time. I also looked into the provided repository https://github.com/odellus/tennis and compared it to my code. I also tried out the hyperparameters of this solution in my implementation. Credits to the author (Thomas Wood) for this solution.

- https://knowledge.udacity.com/questions/101614
    In this post is mentioned, that random sampling at the beginning alone could take about 1.000 episodes gave me confidence about the total time the training could last in the end.

- https://knowledge.udacity.com/questions/303326
    I got confident that the task can be solved within 3.000 episodes.

- https://knowledge.udacity.com/questions/261898
    I got confident that my procedure is basically correct. I got confident that adding the right amount of noise is an very important part in this project.

## Aftermath review
```diff
+ for aftermath review see diff comments below
```

- https://knowledge.udacity.com/questions/315134
    I got confident, that a correct amount of noise is an essential point in this project.
    I tested the given hyperparameter for learning rate for actor and critic, noise, update intervall target networks, gamma, episodes, buffer size, batch size, tau, weight decay. I also got confidence that my implementation in generell is on the right way, as I looked in the provided pseudo code of the overall process. I cloned the provided repository https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition and I debugged my code by comparing every step of the process. This way, I found different bugs in my implementation:
    - I played around with different values for sigma and theta for the noise, with this solution I got confident that solving the task with sigma = 0.2 and theta = 0.15 is possible
    ```diff
    - Saving the experiences into and reading experiences out of the buffer was not correct within my implementation
    + I found out that my implementation was right
    ```
    ```diff
    - I decreased noise over time within my implementation, but to not decrease noise is necessary in this project so I removed it in my implementation
    + I found out that decreasing noise also works
    ```
    ```diff
    - I did not update the target networks at every steps, but this is necessary
    + I found out that my update intervall of every 100th timestep also works. Reaching mean score of 0.5 takes more episodes.
    ```
    ```diff
    - I also did update the target networks at different locations, but it is necessary to update them both at end of every agent's update loop
    + I found out that updating the target networks at different locations also works, but reaching mean score of 0.5 takes a lot more episodes.
    ```
    ```diff
    - I did not use a new sample for every agent, I used the same one, but a new one is necessary
    + I found out that my implementation of using one sample for all agents within one episode also works.
    ```
    ```diff
    - I used a seed of 2, but a seed of 0 is necessary
    + I found out that my seed of 2 also works
    ```
    ```diff
    - I used weight_decay=0.0001 for the critic, but not manually set weight_decay is necessary
    + I found out that setting weight_decay=0.0001 was one of the main issues in my implementation. With this setting the mean score stayed very low (~ 0.00x for over 1.5k episodes till abort). The default is weight_decay=0.
    ```
    ```diff
    - I did a hard update from local to target networks weights on initialization, but this was not necessary
    + I added this after training did not work at first time, later I removed it. I found out, that both variants work.
    ```
    ```diff
    - I did call `actor_local` and `actor_target` for getting the actions in a wrong way: Within the for-loop for each agent, I again looped over all agents and called the method and merged the results, instead of calling it only for the actual agent of the loop
    + I found out that my implementation of looping over all agents also works
    ```
    - I am not sure if I mentioned here and in my implementation all the bugs I could fix with the help of this repository, maybe there where some more.

    Credits to the author (Andrei Bukalov) for this solution.

[1] https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet

[2] https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis

[3] https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet#getting-started

[4] https://github.com/udacity/deep-reinforcement-learning/tree/master/python

[5] https://stackoverflow.com/a/52269934/2988

[6] https://github.com/udacity/deep-reinforcement-learning

[7] https://github.com/ray-project/ray/tree/master/python/ray/tune

[8] https://github.com/Unity-Technologies/ml-agents/issues/1167

[9] https://github.com/TobiasSunderdiek/my-udacity-deep-reinforcement-learning-solutions/blob/master/p3_collab-compet/README.md#udacity-honor-code


#todo
---- eigene Buffer-Umwandlung
- README anpassen
- alle fremden Punkte in README durchgehen
- sind hyperparameter die selben?
- multi_agent neue learn methode vergleichen
-- immer mit dem selben buffer gearbeitet
-- critic_target hatte ich kein .detach()
-- soft_update nach critic direkt und nicht erst am Ende
-- all_actions_local variable habe ich zu sich selbst zugewiesen, sollte aber möglich sein
-> sonst ist Umformung kein Problem
--> ist das Problem das jeder Lauf unterschiedlich ist, mal geht es mal nicht?
--> oder einfach Geduldsproblem, da am Anfang oft 0.00x WErte und dann wieder 0.0 kommmen?
- alte version nochmal laufen lassen, aber Achtung, OUNoise hat andere Params + ander HParams checken vorher
--> viele schnelle Änderungen und trains schienen Einfluß zu haben

collab_competiton.py
- habe self.agents.reset_noise früher ausgeführt, anstatt self.agents.reset_noise()
- und muß es dann nicht auch self.agents[i].reset_noise() sein in multi_agent.py?