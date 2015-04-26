# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:30:36 2015

@author: pbustos
"""

import numpy as np
import cv2


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


#def searchCandidate(otsu):
#	h = otsu.shape[0]
#	test = np.random(h/2,h,10)
#	for y in tange(test):
#		cool, mean =computeMean(otsu[y,:])

cap = cv2.VideoCapture(0)

#Road
meanList = []
ret, frame = cap.read()
height = frame.shape[0]
width = frame.shape[1]
mentalRoad = [ np.array([width/2, y]) for y in range(height-1, height/3, -50)]

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Gaussian filtering
	grayG = cv2.GaussianBlur(gray,(5,5),0)

	# Otsu's thresholding
	ret2, otsu = cv2.threshold(grayG, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	# compute atractors. Each black point attracts with a force proportional to its distance
	for i in range(1, len(mentalRoad)-2):
		w0 = mentalRoad[i-1]
		w1 = mentalRoad[i]
		w2 = mentalRoad[i+1]

		n = np.linalg.norm(w0-w1) / np.linalg.norm(w0-w1)
		attF = (w2 - w0)*n - (w1-w0)
		res = np.array([0,0])
		for y in range(height-1, height/2,-10):
			for x in range(0,width,10):
				if otsu[y,x]==0:
					force = np.array([y,x]) - c
					force = force / np.square(np.linalg.norm(force))
					res	= res + force
		print res
		c = c + res

	for c in mentalRoad:
		cv2.circle(frame,tuple(c),15,(0,0,255),2)



		#mean = computeMean(otsu[y,:])
			#if mean > 0:
			#	meanList.append(mean)
			#	cv2.circle(frame,(mean,y),10,(0,255,0),2)



    # Display the resulting frame
	cv2.imshow('frame',frame)
	#cv2.imshow('bin',otsu)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

