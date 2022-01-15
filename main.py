

import sys, numpy
from gui1 import Ui_MainWindow
from PyQt5 import  QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from PIL.ImageQt import ImageQt
from matplotlib import pyplot
from math import exp
from pylab import *
from sklearn.cluster import MeanShift, estimate_bandwidth



###Same line of code is not explained /commented again in the code###

class myClass(Ui_MainWindow):
    
        Image_loaded=0 #If we click on any editing button and no image is loaded then it wont crash. it checks whether 
                       #an image is loaded or not.
                       
        
        def __init__(self,window): #trigerring and connection of various buttons to their respective functions.
            self.setupUi(window)
            self.pushButton.pressed.connect(self.load_image_fn)
            self.pushButton_2.pressed.connect(self.save_image_fn)
            
        def load_image_fn(self):#loads an image
             name1,junk=QFileDialog.getOpenFileName() #opening a file dialog 
             if name1: #if file selected then it loads otherwise prevents crash.
                  myClass.Image_loaded=1                # sets the image loaded value to 1
                  self.im =Image.open(name1) #opens the image in label box after converting it to greyscale
                  self.temp=self.im      #a temporary variable to store original image for undo
                  #Display images
                  #plt.contour(self.im,colors='red')
                  #self.im.show()
                  qim1 = ImageQt(self.temp)   #conversion of image from PIL image format to QImage.
                  pix1 = QtGui.QPixmap.fromImage(qim1) #conversion of image from QImage image format to pixmap, so that we can display in label
                  self.label_2.setPixmap(pix1)
                  pix_val = np.asarray(self.im)
                  #print pix_val
                  print pix_val.shape
                  print pix_val.dtype
                  if pix_val.ndim >2:
                        flat_image=np.reshape(pix_val, [-1, 3])
                  else:
                        flat_image=np.reshape(pix_val, [-1, 1])
                  print flat_image.shape
                  bw=estimate_bandwidth(flat_image,quantile=.5, n_samples=200)
                  ms= MeanShift(bw, bin_seeding=True)
                  labels=ms.fit_predict(flat_image)
                  print labels
                  print labels.shape
                  print labels.dtype
                  labels=labels.astype(np.uint8)
                  print labels.dtype
                  maxi=np.amax(labels)
                  print labels
                  labels= (labels/maxi)*255
                  im1=Image.fromarray((np.reshape(labels, [-1,pix_val.shape[1]])))
                  #contour(im1, levels=[245], colors='red', origin='image')
                  im1.show()#obtains PIL image from array
                  #plt.show()
                  qim = ImageQt(im1)
                  pix = QtGui.QPixmap.fromImage(qim)
                  self.label_3.setPixmap(pix)
                  plt.contour(im1,colors='red')
                  plt.imshow(np.reshape(labels, [-1,pix_val.shape[1]]))
                 
         
      
        def save_image_fn(self): #saves edited image
            
            if myClass.Image_loaded:
                name1,junk=QFileDialog.getSaveFileName() #function for saving a file, dialog box called.
                if name1: #if file selected then it saves otherwise prevents crash.
                    self.im.save(name1)    #it saves image using PIL save function  
        
                        
    
if __name__ == "__main__": # main 
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = myClass(MainWindow)
        MainWindow.show()
        sys.exit(app.exec_())
