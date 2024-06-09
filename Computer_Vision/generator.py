# This will be a licence plate generator used for testing.

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image as im


# Imports the image to a numpy array.
image = im.open("testimage.png").covert('RGB')
image = np.array(image)

plt.imshow(image)
plt.show()





