import numpy as numpy
import tensorflow as tf

class MetaLearner(object):

	def __init__(self):
		self._sess, self._training_data, self._training_labels,\
			self._test_data, self._test_labels = self._setup_graph()

	def _setup_placeholders(self, inputs=None):
		if inputs is None:
			training_data = tf.placeholder(name="training_data", dtype=tf.float32)
			training_labels = tf.placeholder(name="training_labels", dtype=tf.float32)
			test_data = tf.placeholder(name="test_data", dtype=tf.float32)
			test_labels = tf.placeholder(name="test_labels", dtype=tf.float32)
		else:
			training_data = inputs['training_data']
			training_labels = inputs['training_labels']
			test_data = inputs['test_data']
			test_labels = inputs['test_labels']

		return training_data, training_labels, test_data, test_labels

	def _setup_graph(self):

		sess = tf.Session()

		training_data, training_labels, test_data, test_labels = self._setup_placeholders()

		return sess, training_data, training_labels, test_data, test_labels

	def _metalearn(self, reuse=True):
		training_preds = self._forward(self._training_data, self._weights, reuse=reuse)
        training_loss = self._compute_loss(training_preds, self._training_data)

