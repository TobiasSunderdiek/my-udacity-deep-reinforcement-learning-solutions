from ray import tune
from ray.tune.schedulers import FIFOScheduler

from continuous_control import ContinuousControl

class Trainable(tune.Trainable):
    def setup(self, hyperparameter):
        self.continuous_control = ContinuousControl(hyperparameter)

    def step(self):
        score = self.continuous_control.train()
        return {'score': score}

hyperparameter = {'gamma': 0.99,
                'sample_batch_size': tune.grid_search([64, 128]),
                # cast buffer size to int, I got the casting from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/ddpg_agent.py#L12
                # otherwise index error due to float
                'replay_buffer_size': tune.grid_search([int(1e3), int(1e4), int(1e5)]),
                'tau': tune.grid_search([0.0001, 0.001, 0.01]),
                'actor_learning_rate': tune.grid_search([10e-4, 10e-3]),
                'critic_learning_rate': tune.grid_search([10e-3, 10e-2]),
                'update_every': tune.grid_search([5, 10])
            }

tune.run(
    Trainable,
    config=hyperparameter,
    num_samples=1,
    local_dir='./runs',
    checkpoint_at_end = True,
    verbose=1,
    resources_per_trial={"cpu": 0.25, "gpu": 0.25},
    scheduler=FIFOScheduler()
)