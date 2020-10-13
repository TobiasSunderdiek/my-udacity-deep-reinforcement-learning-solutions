from ray import tune
from unityagents import UnityEnvironment

from collab_competition import CollaborationAndCompetition

class Trainable(tune.Trainable):
    def setup(self, hyperparameter):
        self.env_filename = self.logdir + '../../../Tennis.app'
        self.collab_competition = CollaborationAndCompetition(hyperparameter)

    def step(self):
        self.env = UnityEnvironment(file_name=self.env_filename)
        reward_mean = self.collab_competition.train(self.env)
        self.env.close()
        return {'reward_mean': reward_mean}

hyperparameter = {'gamma': 0.99,
                'sample_batch_size': tune.grid_search([512]),
                # cast buffer size to int, I got the casting from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/ddpg_agent.py#L12
                # otherwise index error due to float
                'replay_buffer_size': tune.grid_search([int(1e6)]),
                'tau': tune.grid_search([0.1]),
                'actor_learning_rate': tune.grid_search([0.001]),
                'critic_learning_rate': tune.grid_search([0.001]),
                'update_every': tune.grid_search([5]),
                'init_weights_variance': tune.grid_search([0.03, 0.05, 0.08]),
                'hidden_layer_1': tune.grid_search([400]),
                'hidden_layer_2': tune.grid_search([300]),
                'sigma': tune.grid_search([0.2, 1.0, 2.0]),
                'theta': tune.grid_search([0.15, 1.0])
            }

tune.run(
    Trainable,
    config=hyperparameter,
    num_samples=1,
    local_dir='./runs',
    verbose=1,
    resources_per_trial={"cpu": 3, "gpu": 0}
)