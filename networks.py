"""
network architectures for clog loss classification challenge
Jason Manley, jmanley@rockefeller.edu
"""

import os, sys
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
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
	cnn_dim   = output size of cnn.predict(x). Default is 10000 (i.e., output of ImageNet models)
	dropout   = dropout rate. Default is 0.1.
	padframes = 
	*args, **kwargs for model.compile() command

	METHODS:
	.summary(), .fit(), .predict() designed to feel like Keras.Model methods.
	"""

	def __init__(self, cnn, cnn_dim=1000, dropout=0.1, padframes=100, 
		         lstm_units=64, *args, **kwargs):

		self.cnn = cnn
		self.cnn_dim = cnn_dim
		self.padframes = padframes

		# DEFINE MANY-TO-ONE LSTM MODEL FOR CLASSIFICATION
		self.model = tf.keras.Sequential()
		self.model.add(Input(shape=(padframes,cnn_dim)))
		self.model.add(Bidirectional(LSTM(lstm_units)))
		self.model.add(Dropout(dropout))
		self.model.add(Dense(64,      activation='relu'))
		self.model.add(Dense(1,       activation='sigmoid'))

		self.model.compile(loss='binary_crossentropy',
						   optimizer='adam',
						   metrics=['accuracy'],
						   callbacks=[EarlyStopping(monitor='val_loss', patience=5)],
						   *args, **kwargs) 


	def summary(self):
		print('CNN Model:', self.cnn._name)
		self.model.summary()


	def fit(self, clogData, epochs=10, validate=0.2, **kwargs):

		# LOAD DATA AND PUT THROUGH CNN
		train_file = os.path.join(clogData.path, self.cnn._name + '_features.npy')

		if os.path.exists(train_file):
			print('LOADING CNN FEATURES...')
			train_data = np.load(train_file)

		else:
			n_train = len(clogData.train)
			train_data = np.zeros((n_train, self.padframes, self.cnn_dim))

			print('FINDING CNN FEATURES...')

			for i in tqdm(range(n_train), file=sys.stdout):
				vid = clogData.load(clogData.train.index[i], train=True)
				cnn_features = self.cnn.predict(vid)

				train_data[i] = pad_sequences(cnn_features.T, maxlen=self.padframes, dtype='float').T

			np.save(train_file, train_data)

		for i in range(train_data.shape[2]):
			train_data[:,:,i] = StandardScaler().fit_transform(train_data[:,:,i])

		train_labels = np.asarray(clogData.train.stalled.values)

		# TRAIN LSTM
		print()
		print('TRAINING CLASSIFICATION MODEL...')
		self.model.fit(train_data, train_labels, epochs=epochs, 
			           validation_split=validate, **kwargs)


	def predict(self, x):
		"""Predicts utilizing trained LSTM model."""

		return self.model.predict(pad_sequences(self.cnn.predict(x).T, self.padframes).T)