# USAGE
# python lenet_mnist.py --save-model 1 --weights output/lenet_weights.hdf5
# python lenet_mnist.py --load-model 1 --weights output/lenet_weights.hdf5

# import the necessary packages
from pyimagesearch.cnn.networks import LeNet
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import keras.callbacks
import numpy as np
import argparse
import cv2
import os
import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())

# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)
print("[INFO] generating dataset...")
data = []
dataset = []
z=[]
dirname = "dataset"
li = os.listdir(os.path.join(dirname))
for l in li:
	z.append(l)
z.sort()
print z
for i in range(0,len(z)):
	li1 = os.listdir(os.path.join(dirname,z[i]))
	z1=[]
	for j in li1:
		z1.append(j)
	print z[i]
	for k in z1:
		print dirname
		print z[i]
		print k
		print os.path.join(dirname,z[i],k)
		img = cv2.imread(os.path.join(dirname,z[i],k),0)
		#print img.height, img.width
		#cv2.ShowImage("output",img)
		h,w = img.shape[:2]
		print h,w
		res = cv2.resize(img,(140, 140), interpolation = cv2.INTER_CUBIC)
		h,w = img.shape[:2]
		print h,w
		h,w = res.shape[:2]
		print h,w
		data.append(res)
		print i
		dataset.append(i)

#dataset = datasets.fetch_mldata("MNIST Original")

# reshape the MNIST dataset from a flat list of 784-dim vectors, to
# 28 x 28 pixel images, then scale the data to the range [0, 1.0]
# and construct the training and testing splits
#data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
print np.array(data).shape
data = np.array(data)
dataset = np.array(dataset)
data = data[:, np.newaxis, :, :]
print data.shape
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data / 255.0, dataset.astype("int"), test_size=0.33)
print np.array(trainData).shape
print np.array(testData).shape
print np.array(trainLabels).shape
print np.array(testLabels).shape

print trainLabels
print testLabels

# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels
trainLabels = np_utils.to_categorical(trainLabels, 3)
testLabels = np_utils.to_categorical(testLabels, 3)
print trainLabels
print testLabels
# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.0005)
model = LeNet.build(width=140, height=140, depth=1, classes=3,
	weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# only train and evaluate the model if we *are not* loading a
# pre-existing model
test_acc=[]
train_acc=[]
cut = 0
if args["load_model"] < 0:
	print("[INFO] training...")
	for i in range(1,5):
		a=model.fit(trainData, trainLabels, batch_size=1, epochs=5,verbose=2,callbacks=[keras.callbacks.History()])
#		model.fit(trainData, trainLabels, batch_size=1, epochs=40,verbose=2)
		temp=a.history.values()
		for j in temp[0]:
			g = float(j) 
			train_acc.append(g)
		(loss, accuracy) = model.evaluate(testData, testLabels,	batch_size=1, verbose=1)
		print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
		test_acc.append(accuracy)
		#x = np.arange(1,float(len(train_acc)+1))
		#y = np.arange(1,float(len(train_acc)+1),5)
		#plt.close('all')
		print train_acc
		print test_acc
		#print cut
		#if len(test_acc)>2 and (accuracy - test_acc[len(test_acc)-2]) < 0.0000001 :
			#break

	# show the accuracy on the testing set
	print("[INFO] evaluating...")
	(loss, accuracy) = model.evaluate(testData, testLabels,
		batch_size=1, verbose=1)
	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# check to see if the model should be saved to file
print fuck
if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"], overwrite=True)
axis_x = np.arange(1.0/(len(train_acc)+1),1.0,1.0/(len(train_acc)+1))
axis_y = np.arange(1.0/(len(test_acc)+1),1.0,1.0/(len(test_acc)+1))
plt.plot(axis_x,np.array(train_acc),axis_y,np.array(test_acc))
plt.show()
print fuck
# randomly select a few testing digits
tot = 0.0
corr = 0.0
for i in np.random.choice(np.arange(0, len(testLabels)), size=(26,)):
	# classify the digit
	probs = model.predict(testData[np.newaxis, i])
	prediction = probs.argmax(axis=1)

	# resize the image from a 28 x 28 image to a 96 x 96 image so we
	# can better see it
	#image = (testData[i][0] * 255).astype("uint8")
	#image = cv2.merge([image] * 3)
	#image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	#cv2.putText(image, str(prediction[0]), (5, 20),
		#cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
	#print probs
	#print prediction
	#print str(prediction)
	#print prediction[0]
	#print str(prediction[0])
	# show the image and prediction
	print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
		np.argmax(testLabels[i])))
	tot+=1
	if int(format(prediction[0])) == int(np.argmax(testLabels[i])) :
		corr+=1
	#cv2.imshow("Digit", image)
	#cv2.waitKey(0)
print corr/tot
