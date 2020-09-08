from ray import tune
from unityagents import UnityEnvironment

from collab_competition import CollaborationAndCompetition

class Trainable(tune.Trainable):
    def setup(self, hyperparameter):
        self.env_filename = self.logdir + '../../../Tennis.app'
        self.collab_competition = CollaborationAndCompetition(hyperparameter)

    def step(self):
        env = UnityEnvironment(file_name=self.env_filename)
        episode_reward_mean = self.collab_competition.train(env)
        env.close()
        return {'episode_reward_mean': episode_reward_mean}#todo acutal not episode but all-episodes reward mean

hyperparameter = {'gamma': 0.99,
                'sample_batch_size': tune.grid_search([128]),
                # cast buffer size to int, I got the casting from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/ddpg_agent.py#L12
                # otherwise index error due to float
                'replay_buffer_size': tune.grid_search([int(1e6)]),#todo asha
                'tau': tune.grid_search([0.001, 0.01, 0.1]),
                'actor_learning_rate': tune.grid_search([10e-4, 10e-3]),
                'critic_learning_rate': tune.grid_search([10e-3, 10e-2]),
                'update_every': tune.grid_search([5, 10])
            }

tune.run(
    Trainable,
    config=hyperparameter,
    num_samples=1,
    local_dir='./runs',
    verbose=1,
    resources_per_trial={"cpu": 3, "gpu": 0}#todo
)