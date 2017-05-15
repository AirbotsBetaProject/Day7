# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

K.set_image_dim_ordering('th')

class LeNet:
	@staticmethod
	def build(width, height, depth, classes, weightsPath=None):
		# initialize the model
		model = Sequential()

		# first set of CONV => TANH => POOL
		model.add(Conv2D(8, (5, 5), padding="same",
			input_shape=(depth, height, width)))
		model.add(Activation("tanh"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# second set of CONV => TANH => POOL
		model.add(Conv2D(8, (5, 5), padding="same"))
		model.add(Activation("tanh"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# third set of CONV => TANH => POOL
		model.add(Conv2D(8, (5, 5), padding="same"))
		model.add(Activation("tanh"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# fourth set of CONV => TANH => POOL
		model.add(Conv2D(8, (5, 5), padding="same"))
		model.add(Activation("tanh"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# set of FC => TANH layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("tanh"))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# if a weights path is supplied (inicating that the model was
		# pre-trained), then load the weights
		if weightsPath is not None:
			model.load_weights(weightsPath)

		# return the constructed network architecture
		return model
