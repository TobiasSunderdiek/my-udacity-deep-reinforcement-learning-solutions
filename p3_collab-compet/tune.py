from ray import tune
from unityagents import UnityEnvironment

from collab-competition import CollaborationAndCompetition

class Trainable(tune.Trainable):
    def setup(self, hyperparameter):
        self.env_filename = self.logdir + '../../../Tennis_Linux_NoVis/Tennis.x86_64'
        self.continuous_control = CollaborationAndCompetition(hyperparameter)

    def step(self):
        env = UnityEnvironment(file_name=self.env_filename)
        score = self.continuous_control.train(env)
        env.close()
        return {'score': score}

hyperparameter = {'gamma': 0.99,
                'sample_batch_size': tune.grid_search([128]),
                # cast buffer size to int, I got the casting from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/ddpg_agent.py#L12
                # otherwise index error due to float
                'replay_buffer_size': tune.grid_search([int(1e6)]),#todo asha
                'tau': tune.grid_search([0.01]),
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
    resources_per_trial={"cpu": 4, "gpu": 1}#todo
)