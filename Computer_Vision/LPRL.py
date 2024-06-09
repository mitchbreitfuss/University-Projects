'''Licence Plate Reader Library. This library is responsibe for all image processing and manipulation before it is passed into the neural network.'''
# Authors: Mitchell Breitfuss
# LPRL stands for Licence Plate Reader Library

# NOTE: A mixture of PIL and OpenCV image formats are used throughout this class.
# PIL --> CV Image conversion
# image --> np.asarray(image)

# PIL may be easier to use as it is object based and doesn't use numpy
# PIL will be depreciated, just use OpenCV.

# TODO
# [ ] Improve segment method, devise a smart way to determine window location
# [✓] Streamline the library, require less dependancies and stick with one image processing library (CV or PIL)
# [ ] Test effectiveness with many input images.
## Library Importation.
import os, os.path
import cv2 as cv # Imports the OpenCV libary. Uses short hand cv vs original cv2 for ease of code writing.
import numpy as np # Imports numpy math library.
import matplotlib.pyplot as plt  # Imports the matplotlib plotting library
import tkinter as tk
from tkinter import filedialog
import glob

DATA_ENTRY_MODE = False


# The library has been contained to a single class containing all required methods, allowing an easilly repeatable workflow.
# (Image In --> Binary Segmented Image)
# If unsure of how this structure works or it's benefits, please read into some OOP tutorials for Python3

class imMan(): # Structuring the functions as class methods allows us to create objects that are easier to pass to the neural net.


    def __init__(self,inputImage): # __init__ method runs whenever an instance of the class is created.
			           # Because the class only needs to run each method once, we simply get it to run everything in order.
				   # i.e, when the class is called it processes the input image immediately.
        # Initialise object
        # print("Size: {0}".format(inputImage.size))
        
        self.inputImage = inputImage
        self.cornerDetect()
        self.binary_image = [[],[],[],[],[],[]]
        self.sizex,self.sizey = self.inputImage.shape
        self.segment()
        self.edgeDetect()
        self.fill()
        for i in range(0, len(self.binary_image)):
            self.binary_image[i] = cv.resize(self.binary_image[i],(125,219))

    
    def cornerDetect(self):
        '''Method to automatically crop input image to ensure best segmentation results'''
        # This method is responsible for the image being automatically cropped.
		# Needs renaming
			    
        '''Cropping for segmentation'''
        def closest_node(node, locations):
             nodes = np.asarray(locations)
             dist_2 = np.sum((nodes - node)**2, axis=1)
             return np.argmin(dist_2)
        dst = cv.cornerHarris(self.inputImage, 2, 3, 0.04)
        dst = cv.dilate(dst, None)
        threshold = [dst > 0.001 * dst.max()]
        locations = np.column_stack(np.where(threshold))
        test = locations[:,1:3]
        
        closestTL = closest_node((0, 0), test)
        closestTR = closest_node((627, 0), test)
        closestBL = closest_node((0, 940), test)
        closestBR = closest_node((627,940),test)
        
        TL = test[closestTL]
        BL = test[closestTR]
        TR = test[closestBL]
        BR = test[closestBR]

        y1, y2 = int((TL[0]+TR[0])/2), int((BL[0]+BR[0])/2)
        x1,x2 = int((BL[1]+TL[1])/2),int((TR[1]+BR[1])/2)
        plt.imshow(self.inputImage,cmap='gray')
        plt.show()
        self.inputImage[threshold] = [255]
        # self.inputImage = self.inputImage[y1:y2,x1:x2]
        # plt.imshow(self.inputImage,cmap='gray')
        # plt.show()

    def edgeDetect(self):
        # This method takes in the segmented parts of the image, and converts them to a edge image
        for i in range(0,len(self.segments)):
            gray = self.segments[i]
            gray = cv.blur(gray,(15,15)) # Gaussian blur results in more clearly defined edges
										# This is important when filling the image background.

            # Edge detection            
            _, im_th = cv.threshold(gray,117,220,cv.THRESH_BINARY_INV)
            edges = cv.Canny(im_th,90,255)

            _,contours,_ = cv.findContours(edges,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(edges,contours,-1,(255,0,0),cv.FILLED)
            
            self.binary_image[i] = edges
            
            # plt.imshow(edges,cmap='gray',interpolation='bicubic')
            # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            # plt.show()
        
        
    def segment(self):
         # This function could use some work, it's pretty dumb at the moment and uses trial and error values to segment the image
         # 

        work = self.inputImage
        worksize = work.shape
        
        # plt.imshow(work)
        # plt.show()
        
        # Segment the image into individual characters
        # Segments the image based on percentage of dimensions instead of flat values.
        # Many of these percentages were determined based on trial and error.
        top = 220#int(0.14*worksize[0])#230
        bottom = 430#int(0.82*worksize[0])#420
        windowsize = 127#int(0.134*worksize[1])#125

        x1 = int(0.054*worksize[1])
        x2 = x1 + windowsize
        x3 = x2 + windowsize
        x4 = x3 + windowsize
        
        x5 = x4 + int(0.11*worksize[1])
        x6 = x5 + windowsize
        x7 = x6 + windowsize
        x8 = x7 + windowsize

        
        self.seg1 = work[top:bottom, x1:x2]
        self.seg2 = work[top:bottom, x2:x3]
        self.seg3 = work[top:bottom,x3:x4]

        self.seg4 = work[top:bottom,x5:x6]
        self.seg5 = work[top:bottom,x6:x7]
        self.seg6 = work[top:bottom,x7:x8]
        

        self.segments = (self.seg1, self.seg2, self.seg3, self.seg4, self.seg5, self.seg6)
        # This is for testing
        #for i in range(0,6):
        #     plt.subplot(1,7,i+1)
        #     plt.imshow(self.segments[i],cmap='gray',interpolation='bicubic')
        #     plt.xticks([]), plt.yticks([])
        # plt.show()
        
    def fill(self):
        '''Method to fill the segmented images.'''
		# 
        i=0
        for segment in self.binary_image:
            
            if segment != []:
                segment = np.asarray(segment) # Converts PIL image segment to a OpenCV compatible numpy array.
                h,w = segment.shape[:2] # Gets height and width of segment.
                mask = np.zeros((h + 2, w + 2), np.uint8) # Creates an empty mask 
                cv.floodFill(segment, mask, (14, 14), 255) # fills the image white from point (0,0) (Top left corner)
                
                mask = mask*255
                mask = cv.resize(mask,(w,h))    
                newflood = cv.bitwise_xor(mask, segment) 

                mask = np.zeros((h + 2, w + 2), np.uint8)
                _, threshold = cv.threshold(newflood, 254, 255, cv.THRESH_TOZERO)
                
                cv.floodFill(threshold, mask, (14, 14), 255)
                threshold = self.invert(threshold)
                out = threshold | segment
                out = self.invert(out)
                # Much of this method is magic and I can't remember how it was done. It works fine though.
                
                # Testing Code 
                # if (i == 1):
                #     print('test')
                #     pass
                self.binary_image[i] = out
                # plt.imshow(out,cmap='gray')
                # plt.show()
                #print("stop")
            i += 1
            

    def invert(self,image):
        image = (255-image)
        return image

    def channel(self):
        self.R,self.G,self.B = cv.split(np.array(self.inputImage))



# This class is for the manual importation of each possible character (Alphanumeric characters.)
# This class should not handle any sort of cropping at this stage, the 
class import_training_character():
    ''' This class is for the manual importation of each possible character. (Alphanumeric characters.) \n Accepts an input character image in OpeCV format, and the corresponing character code. (See the README to see a list of character codes.'''
    
    def __init__(self):
        print('\n\n\n\n\n\n\n\n\n\n\n')
        script_dir = os.path.dirname(__file__)
        training_folder_name = "trainingdata"
        self.training_dir = os.path.join(script_dir, training_folder_name)

        # Checks to see if the training directory is empty.
        # If it is, creates the y data file.
        self.check_training_index()
        if (self.training_index == -1):
            np.savetxt(os.path.join(self.training_dir, 'y.txt'), [])
            print("No training data: Creating y.txt.")
        
        root = tk.Tk()
        root.withdraw()
        self.still_using = 1
        self.main()
        # self.character_image = character_image
        # self.character_attribute = character_attribute
    
    def check_training_index(self):
                # Gets the current number of training files, minus one for the text file.
                self.training_index = len([name for name in os.listdir(self.training_dir) if os.path.isfile(os.path.join(self.training_dir, name))]) - 1
                # print("There are currently " + str(self.training_index) + " training files.")
        
    def main(self):
        input_length = 0
        
        while self.still_using == 1:
            
            print("Please open a licence plate file.")
            file_path = filedialog.askopenfilename()
            if (file_path.__len__() == 0):
                print("Error, no plate selected. Aborting.")
                exit()
            plate_image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)  # Reads plate image
            plt.imshow(plate_image)
            plt.show()
            while input_length != 6:  # Sanity check to ensure that the input is 6 characters 
                
                print('Please enter the Licence data for this image. (eg, YAB00A)')
                licence_id = input("➤")
                licence_id = licence_id.upper()
                input_length = len(licence_id)
                if (input_length != 6):
                    print("Error: Incorrect Length")
            input_length = 0
            # Put code here that segments the image and writes training data.
            # Calls imMan class
            
            process = imMan(plate_image)
            segments = process.binary_image

            # Check what training number we are up to.
            for i in range(0, 6):
                self.check_training_index()
                new_filename = f"{self.training_index}.png"
                ind_character = segments[i]
                cv.imwrite(os.path.join(self.training_dir,new_filename),ind_character)

            # Writing Y array
            outfile = open(os.path.join(self.training_dir, 'y.txt'), 'a')
            for i in range(0, 6):
                char = licence_id[i]
                try:
                    int(char)
                    char = ord(char) - 48
                except:
                    char = ord(char) + 36
                
                outfile.write(str(char) + "\n")
            outfile.close()
            print()
            print('Do you want to input another plate? [y/n]')
            response = input('➤')
            if (response == 'n'):
                self.still_using = 0
            
    def structure_data(self):
        
        pass

class fixtraining():

    def load_data(self):  # This method is responsible for loading the training data.
                    # Currently it loads from the root directory.
        filenames = [img for img in glob.glob("*.png")]
        images = []
        for i in range(0, len(filenames)):
            name = filenames[i]
            print(name)
            temp = cv.imread(name,cv.IMREAD_GRAYSCALE)
            images.append(temp)
        return np.asarray(images)

    def fix(self):
        images = self.load_data()
        outimage = []
        j=0
        for i in range(0, images.shape[0] * 20):
            
            name = f"{i}.png"
            tempout = np.asarray(images[j])
            tempout = imMan.invert(self,tempout)
            tempout = self.resize(tempout)
            cv.imwrite(name, tempout)
            j += 1
            if (j == 25):
                j=0
            
        print("Hello")

    def resize(self,image):
         outimage = cv.resize(image, (125, 219))
         return outimage

if(DATA_ENTRY_MODE == True):
    instance = import_training_character()

# instance = fixtraining()
# instance.fix()

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
