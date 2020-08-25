## Report

#### Learning Algorithm

I have choosen a DDPG to solve this environment. #todo

#### Model

#todo

#### Hyperparameter
#todo

#### Rewards

The agent reaches a mean reward of #todo over the last 100 episodes after episode #todo.

![mean reward plot](tensorboard_reward.png)

#### Ideas for Future work

#todo trpo

- The use of a prioritized experience replay buffer, in which sampling focus lies on values with high error, could make the agent reach the goal faster. Due to the values have high error, there is a lot to learn from this values. Also sparse experiences have the chance to be sampled more often.