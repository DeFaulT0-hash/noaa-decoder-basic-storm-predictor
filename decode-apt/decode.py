import numpy as np
import scipy.signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import cv2
from tkinter import *

class NOAA:
	def __init__(self, filename, norm):
		#extract the data from WAV file
		self.sampling_rate, self.data = wavfile.read(filename)

		#normalization value which will be later used to bring the amplitudes to be in 0 to 255 range
		self.norm = norm
		self.am_envelope = self.hilbert(self.data)

	def hilbert(self, data):
		#hilbert transfrom will give us an analytical signal
		#this will help us extract the envelopes instantaneously
	    #find the analytical signal
	    analytic_signal = scipy.signal.hilbert(data)
	    
	    #extract the amplitude envelope
	    am_envelope = np.abs(analytic_signal)
	    
	    return am_envelope

	def getImageArray(self, am_envelope, norm):
		print("Processing image...")

		#calculate the width and height of the image
		width = int(self.sampling_rate*0.5)
		height = self.am_envelope.shape[0]//width
		print(f"width: {width}, height: {height}")
		
		#create a numpy array with three channels for RGB and fill it up with zeroes
		img_data = np.zeros((height, width, 3), dtype=np.uint8)

		#keep track of pixel values
		x = 0
		y = 0

		#traverse through the am_envelope and replace zeroes in numpy array with intensity values
		for i in range(self.am_envelope.shape[0]):

		    #get the pixel intensity
		    intensity = int(self.am_envelope[i]//norm)

		    #make sure that the pixel intensity is between 0 and 255
		    if intensity < 0:
		        intensity = 0
		    if intensity > 255:
		        intensity = 255

		    #put the pixel on to the image
		    img_data[y][x] = intensity

		    x += 1

		    #if x is greater than width, sweep or jump to next line
		    if x >= width:
		        x = 0
		        y = y+1

		        if y >= height:
		            break

		print("Image processed.")

		return img_data

	def plot(self):
		#get the image data as numpy array
		img_data = self.getImageArray(self.am_envelope, self.norm)
		imname='Sat Image'
		# cv2.namedWindow("output", cv2.WINDOW_NORMAL)
		im_resize=cv2.resize(img_data,(1920,1080))
		#plot the numpy array as an image
		print("Plotting the image")
		cv2.imshow(imname,im_resize)
		cv2.waitKey(0) 
  
		#closing all open windows 
		cv2.destroyAllWindows() 
		# plt.imshow(img_data, aspect="auto")
		# plt.show()


#The GUI part

window=Tk()
window.title('NOAA Signal Decoder')
window.geometry("800x400")

lbl1=Label(window, text='Enter the filename (Example : noaa3.wav): ',font=("Times New Roman", 15))
lbl2=Label(window, text='Enter the normalization index : ',font=("Times New Roman", 15))
t1=Entry(font=("Times New Roman", 15))
t2=Entry(font=("Times New Roman", 15))
lbl1.place(x=100, y=50)
t1.place(x=500, y=50)
lbl2.place(x=100, y=100)
t2.place(x=500, y=100)



def click():
    filename=t1.get()
    norm=t2.get()
    norms=int(norm)
    decoder = NOAA(filename, norms)
    decoder.plot()

btn=Button(window, text="Process", fg='blue',font=('Helvetica 18'),command=click)
btn.place(x=350,y=200)

# decoder = NOAA(txtfld, 40)

#create an instance of NOAA object

window.mainloop()


# filename = input("Enter the WAV file name: ")
# norm = int(input("Enter the normalization (50 to 70 would be good): "))

#create an instance of NOAA object

# decoder.plot()
