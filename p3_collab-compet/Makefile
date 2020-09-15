install:
	git clone --depth 1 --no-checkout https://github.com/udacity/deep-reinforcement-learning.git udacity_deep_reinforcement_learning &&\
	cd udacity_deep_reinforcement_learning &&\
	git checkout master -- python/ &&\
	cd python &&\
	pip install . &&\
	cd ../.. &&\
	pip install -r requirements.txt

train:
	python collab_competition.py

tune:
	python tune.py

tensorboard:
	tensorboard --logdir=runs/