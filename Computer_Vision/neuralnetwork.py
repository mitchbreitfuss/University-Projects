'''Self contained neural network'''
import os
import glob
import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import LPRL

def load_data():  # This method is responsible for loading the training data.
					# Currently it loads from the root directory.
	filenames = [img for img in glob.glob("*.png")]
	images = []
	for i in range(0, len(filenames)):
		name = f"{i}.png"
		print(name)
		temp = cv.imread(name,cv.IMREAD_GRAYSCALE)
		images.append(temp)
	return np.asarray(images)
	
mnist = tf.keras.datasets.mnist
images = load_data()
my_ytrain = np.loadtxt('ynew.txt')  # np.asarray([35,10,11,0,0,10]*200)

og = my_ytrain
for i in range(0, 19):
	my_ytrain = np.append(my_ytrain,og,axis=0)
# (x_tra	in, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(images, axis=1)

# Regular NN implimentation
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(36, activation=tf.nn.softmax))

# Convolutional NN implimentation
#my_ytrain = tf.keras.utils.to_categorical(my_ytrain)
# x_train = x_train[..., np.newaxis]
#model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(219,125,1)))
# model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(36, activation='softmax'))

model.compile(optimizer='adam',
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy'])
				

model.fit(x_train,my_ytrain,epochs=5)
model.save('model.model')
# Import image for testing (default = 'testimage.jpg')
test_image = cv.imread('testimage.jpg',cv.IMREAD_GRAYSCALE)
test_image = cv.resize(test_image,(940,627))
instance = LPRL.imMan(test_image)
bin_seg = np.asarray(instance.binary_image)

og = bin_seg
for i in range(0, 6):
	
	
	plt.subplot(1,6,i+1)
	plt.imshow(bin_seg[i],cmap='gray')
plt.show()

bin_seg = bin_seg[...,np.newaxis]

predict = model.predict([bin_seg])
bin_seg = np.squeeze(bin_seg,axis=3)
plate_guess = []

# Prints the predictions and the segments from the test image.
for i in range(0, 6):
	
	print(np.argmax(predict[i]))
	plate_guess.append(np.argmax(predict[i]))
	plt.subplot(1,6,i+1)
	plt.imshow(bin_seg[i,:,:])


# Dictionary for converting the numerical output back to characters.
character_dict = {
		0: "0",
		1: "1",
		2: "2",
		3: "3",
		4: "4",
		5: "5",
		6: "6",
		7: "7",
		8: "8",
		9: "9",
		10: "A",
		11: "B",
		12:"C",
		13: "D",
		14: "E",
		15: "F",
		16: "G",
		17: "H",
		18: "I",
		19: "J",
		20: "K",
		21: "L",
		22: "M",
		23: "N",
		24: "O",
		25: "P",
		26: "Q",
		27: "R",
		28: "S",
		29: "T",
		30: "U",
		31: "V",
		32: "W",
		33: "R",
		34: "X",
		35: "Y",
		36: "Z" 
		
}

output = []
for i in range(0, 6):
	output.append(character_dict[plate_guess[i]])

print(output)
plt.show()
