import tensorflow as tf
import numpy as np

import time

def build_mlp(x, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None, regularizer=None):
    i = 0
    for i in range(n_layers):
        x = tf.layers.dense(inputs=x,units=size, activation=activation, name='fc{}'.format(i), kernel_regularizer=regularizer, bias_regularizer=regularizer)

    x = tf.layers.dense(inputs=x, units=output_size, activation=output_activation, name='fc{}'.format(i + 1), kernel_regularizer=regularizer, bias_regularizer=regularizer)
    return x

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_AC)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

class MAML(object):
	def __init__(self, computation_graph_args):
		self.n_layers = computation_graph_args['n_layers']
		self.alpha = computation_graph_args['alpha']
		self.beta = computation_graph_args['beta']
		self.ob_dim = computation_graph_args['ob_dim']
		self.ac_dim = computation_graph_args['ac_dim']
		self.batch_size = computation_graph_args['batch_size']
		self.horizon = computation_graph_args['horizon']
		self.n_trajectories = computation_graph_args['n_trajectories']

	def init_tf_sess(self):
		tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__()
        tf.global_variables_initializer().run()

    def define_placeholders(self):
    	sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        sy_sampled_ac = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)

    def build_computation_graph(self):
        self.loss = 0.0 # TODO: build full computation graph, I think this should depend on using a MLP to sample actions not sure
    	self.meta_optimizer = tf.AdamOptimizer(self.beta).minimize(self.loss)

    def meta_step(self, learned_policies):
        # TODO: figure out how to pass in learned policies to feed_dict in the best way
        # the loss function uses the sum of the learned policies' individual losses to take a step
    	self.sess.run(self.meta_theta, feed_dict={})

    def sample_trajectories(self, task, for_meta):
        stats = []

        # TODO: Change env depending on input task
        env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
        for i in range(self.n_trajectories):
            trajectory = self.sample_trajectory(env, for_meta=for_meta)
            stats += s
        return stats

    def sample_trajectory(self, env, for_meta):
    	obs = env.reset()
    	for i in range(self.horizon):
    		# Select action using policy
    		ac = self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: obs})

    		# Step the environment
    		obs, rew, done, _ = env.step(ac)

    		# Add to replay buffer
    		if is_evaluation:
    			pass # TODO
    		else:
    			pass # TODO

    		if done:
    			obs = env.reset()


def train_MAML(
		env_name,
		exp_name,
		render,
		n_iter
		alpha,
		beta,
		batch_size,
		horizon,
		n_trajectories
		logdir
		):
	# Initialize clock
	start = time.time()

	# Setup logger
	setup_logger(logdir, locals())

	# Initialize environment
	env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    env.reset()
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

	computation_graph_args = {
        'n_layers': n_layers,
        'alpha': alpha,
        'beta': beta,
        'ob_dim': ob_dim
        'ac_dim': ac_dim
        'batch_size': batch_size
        'horizon': horizon
        'n_trajectories': n_trajectories  
    }

    # Initialize metalearner
    maml_agent = MAML(computation_graph_args)
    maml_agent.build_computation_graph()
    maml_agent.init_tf_sess()

    for i in range(n_iter):
    	print("********** Iteration %i ************"%itr)
    	# TODO: replace num_tasks with number of training levels
    	tasks = np.random.choice(num_tasks, batch_size)
    	for task in tasks:
    		maml_agent.sample_trajectories(task, for_meta=False)

	    	# TODO: Extract trajectories from replay buffer
	    	# Evaluate gradients and update adapted parameters

            maml_agent.sample_trajectories(task, for_meta=True)

        # TODO: pass learned policies to the meta step somehow
	    agent.meta_step()


def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vac')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--horizon', '-h', type=int, default=100)
    parser.add_argument('--n_trajectories', '-t', type=int, default=10)
	args = parser.parse_args()

	data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = 'maml_' + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

	train_MAML(
		env_name=args.env_name,
		exp_name=args.exp_name,
		render=args.render,
		n_iter=args.n_iter,
		alpha=args.alpha,
		beta=args.beta,
		batch_size=args.batch_size,
		logdir=logdir,
		horizon=horizon,
		n_trajectories=n_trajectories
	)
