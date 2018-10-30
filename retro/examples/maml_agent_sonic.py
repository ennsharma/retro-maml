from retro_contest.local import make

import numpy as numpy
import tensorflow as tf

class MetaLearner(object):

	def __init__(self,
				 env, 
				 batchsize=3,
				 horizon=15,
				 nn_layers=1):
		self._cost_fn = env.cost_fn
		self._state_dim = env.observation_space.shape[0]
		self._action_dim = env.action_space.shape[0]
		self._horizon = horizon
		self._batchsize = batchsize
		self._nn_layers = nn_layers
		self._learning_rate = 1e-3
		self._meta_learning_rate = 1e-3

	def _setup_placeholders(self):
		# TODO	

	def _metalearn_step(self):
		tasks = self._sample_tasks()
		for task in tasks:
			# sample K trajectories using f_theta on task

			# compute grad loss over these trajectories

			# update current task function with gradient descent

			# sample new trajectories with updated function, and store it

		# update theta using losses over updated functions per task (meta learning tasks)

	def _sample_tasks(self):
		# TODO: Sample self._batchsize tasks
			



def main():
    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()