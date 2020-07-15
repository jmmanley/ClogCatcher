"""
network architectures for clog loss classification challenge
Jason Manley, jmanley@rockefeller.edu
"""

import os, sys
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CNNFeatureLSTM:
	"""
	A Tensorflow model for many-to-one binary classification of videos.
	First, a pre-trained and pre-specified CNN is applied to the video data
	to get a set of CNN image features for each frame.
	A LSTM-based network is then utilized for binary classification given
	the sequence of CNN features.

	INPUTS:
	cnn       = pre-trained model for generating CNN features
	cnn_dim   = output size of cnn.predict(x). Default is 1000 (i.e., output of ImageNet models)
	dropout   = dropout rate. Default is 0.1.
	padframes = number of frames used as input, pad as necessary.
	*args, **kwargs for model.compile() command

	METHODS:
	.summary(), .fit(), .predict() designed to feel like Keras.Model methods.
	"""

	def __init__(self, cnn, cnn_dim=1000, dropout=0.1, padframes=100,
		         lstm_units=64, dense_units=64, *args, **kwargs):

		self.cnn = cnn
		self.cnn_dim = cnn_dim
		self.padframes = padframes

		# DEFINE MANY-TO-ONE LSTM MODEL FOR CLASSIFICATION
		self.model = tf.keras.Sequential()
		self.model.add(Input(shape=(padframes,cnn_dim)))
		self.model.add(Bidirectional(LSTM(lstm_units)))
		self.model.add(Dropout(dropout))
		self.model.add(Dense(dense_units, activation='relu'))
		self.model.add(Dense(1, activation='sigmoid'))

		self.model.compile(loss='binary_crossentropy',
						   optimizer='adam',
						   metrics=['accuracy', matthewcorr],
						   *args, **kwargs)


	def summary(self):
		print('CNN Model:', self.cnn._name)
		self.model.summary()


	def fit(self, clogData, epochs=10, validate=0.2, **kwargs):
		"""Fits CNN+LSTM model to training data in a ClogData object."""

		# LOAD DATA AND PUT THROUGH CNN
		train_file = os.path.join(clogData.path, self.cnn._name + '_features.npy')

		if os.path.exists(train_file):
			print('LOADING CNN FEATURES...')
			train_data = np.load(train_file)

		else:
			n_train = len(clogData.train)
			train_data = np.zeros((n_train, self.padframes, self.cnn_dim), dtype=np.float32)

			print('FINDING CNN FEATURES...')

			for i in tqdm(range(n_train), file=sys.stdout):
				vid = clogData.load(clogData.train.index[i], train=True)
				cnn_features = self.cnn.predict(vid)

				train_data[i] = pad_sequences(cnn_features.T, maxlen=self.padframes, dtype='float').T

			np.save(train_file, train_data)

		self.scalers = []
		for i in range(train_data.shape[2]):
			ss = StandardScaler()
			train_data[:,:,i] = ss.fit_transform(train_data[:,:,i])
			self.scalers.append(ss)

		train_labels = np.asarray(clogData.train.stalled.values)

		# TRAIN LSTM
		print()
		print('TRAINING CLASSIFICATION MODEL...')
		history = self.model.fit(train_data, train_labels, epochs=epochs, validation_split=validate, **kwargs)
		return history


	def predict_array(self, x):
		"""Predicts array utilizing trained LSTM model."""

		cnn_features = pad_sequences(self.cnn.predict(x).T, maxlen=self.padframes, dtype='float').T

		if len(cnn_features.shape) < 3:
			cnn_features = cnn_features[np.newaxis,:,:]

		for i in range(cnn_features.shape[2]):
			cnn_features[:,:,i] = self.scalers[i].transform(cnn_features[:,:,i])

		return self.model.predict(cnn_features)


	def predict(self, clogData, train=False):
		"""Predicts train or test data in a ClogData obejct."""

		if train: df = clogData.train
		else: df = clogData.test

		predictions = []

		for i in tqdm(df.index.values):
			x = clogData.load(i, train=train)

			predictions.append(self.predict_array(x))

		return np.asarray(predictions)


"""
HELPER FUNCTIONS
"""

def matthewcorr(labels, y_pred, threshold=0.5):
	"""Tensorflow loss for Matthew's correlation coefficient.

	Could use tensorflow_addons implementation, but not compatible with tf 2.1.
	"""

	labels      = tf.cast(labels, tf.float32)
	labels_pred = tf.cast(tf.greater(y_pred, threshold), tf.float32)

	TP = tf.math.count_nonzero(labels_pred     * labels)
	TN = tf.math.count_nonzero((labels_pred-1) * (labels-1))
	FP = tf.math.count_nonzero(labels_pred     * (labels-1))
	FN = tf.math.count_nonzero((labels_pred-1) * labels)

	norm = tf.cast((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN), tf.float32)

	mcc = tf.cast((TP * TN) - (FP * FN), tf.float32) / tf.sqrt(norm)

	return tf.cond(tf.math.is_nan(mcc), 
		           lambda: tf.constant(-1, dtype=tf.float32), 
		           lambda: mcc)
