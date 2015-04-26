# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:30:36 2015

@author: pbustos
"""

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
height = frame.shape[0]
width = frame.shape[1]

def computeMean(row):
	mean = 0
	count = 0
	for x in range(len(row)):
		if row[x] == 0:
				count = count + 1
				mean = mean + x
	if count > 0:
		return mean/count
	else:
		return -1

def localizarLinea(otsu):
	meanList = []
	meanAnt = width/2
	for y in range(height-1, height/3, -10):
		mean = computeMean(otsu[y,:])
		if mean > 0:
			if len(meanList) == 0:
				meanList.append(mean)
				meanAnt = mean
			else:
				if np.abs(mean-meanAnt)<80:
					meanList.append(mean)
					cv2.circle(frame,(mean,y),10,(0,255,0),2)
					meanAnt = mean
	return meanList


while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	# Concertir a gris
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Gaussian filtering
	grayG = cv2.GaussianBlur(gray,(5,5),0)
	# Otsu's thresholding
	ret2, otsu = cv2.threshold(grayG, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	# Localizar la línea negra
	linea = localizarLinea(otsu)
	# compute advance speed prop to len(linea)
	adv = 1.0*len(linea)
	# compute rot speed prop to mean distance of linea to center
	rot = sum([ width/2-l for l in linea])/len(linea)
	print adv,rot
    # Display the resulting frame
	cv2.imshow('frame',frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

