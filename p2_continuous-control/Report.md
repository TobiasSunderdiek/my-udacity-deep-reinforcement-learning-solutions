## Report

#### Learning Algorithm

I have choosen a DDPG to solve this environment. #todo

#### Model

#todo

network structure like in ddpg paper [2] section 7 Experiment Details

fill last layer of weights with uniform distribution /other layers are filled with pytorch defaults which is the same like in the paper
OrnsteinUhlenbeckActionNoise with same parameters like in the paper is added to the action to enable exploration

#### Hyperparameter

**buffer_size**
Configures the maximum size of the replay buffer, older values will be discarded. I started with a size of `10e6`, like described in the DDPG paper. As this takes a while to fill with one agent, I decreased the buffer size to `1e6`, like used in the udacity example in [1].

**sample_batch_size**
Configures how much samples at each learning step should be pulled from the replay buffer, actual value `64` identical to the size in the DDPG paper [2].

**gamma**
The factor how much future rewards should be noted in the valuation of the current action, actual value `0.99` like in the DDPG paper [2].

**tau**
Configures the ratio of how much the target weights in the target network should be updated with actual weights during update process, actual value `1e-3` like in the DDPG paper [2].

**actor_learning_rate**
The learning rate of the actors' optimizer, actual value `10e-4` like in the DDPG paper [2].

**critic_learning_rate**
The learning rate of the critic's optimizer, actual value `10e-3` like in the DDPG paper [2].

#### Rewards

The agent reaches a mean reward of #todo over the last 100 episodes after episode #todo.

![mean reward plot](tensorboard_reward.png)

#### Ideas for Future work

- First of all, maybe switch to multiple agents like described in version 2 of this project could speed-up training due to getting experience in parallel.

- In the notes to the `Benchmark Implementation` from the udacity project, several other algorithms like TRPO, TNPG or D4PG are mentioned to be more stable and to achieve better performance in this project.

- The use of a prioritized experience replay buffer, in which sampling focus lies on values with high error, could make the agent reach the goal faster. Due to the values have high error, there is a lot to learn from this values. Also sparse experiences have the chance to be sampled more often.

[1] https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal

[2] https://arxiv.org/abs/1509.02971