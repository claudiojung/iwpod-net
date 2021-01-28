import sys

import numpy as np
import cv2
import time

from os.path import splitext

from src.label import Label
from src.utils import getWH, nms
from src.projection_utils import getRectPts, find_T_matrix


class DLabel (Label):

	def __init__(self,cl,pts,prob):
		self.pts = pts
		tl = np.amin(pts,1)
		br = np.amax(pts,1)
		Label.__init__(self,cl,tl,br,prob)


def load_model(path,custom_objects={},verbose=0):
	from tensorflow.keras.models import model_from_json

	path = splitext(path)[0]
	with open('%s.json' % path,'r') as json_file:
		model_json = json_file.read()
	model = model_from_json(model_json, custom_objects=custom_objects)
	model.load_weights('%s.h5' % path)
	if verbose: print('Loaded from %s' % path)
	return model



def detect_lp_width(model, I,  MAXWIDTH, net_step, out_size, threshold):
	
	#
	#  Resizes input image and run IWPOD-NET
	#

	# Computes resize factor
	factor = min(1, MAXWIDTH/I.shape[1])
	w,h = (np.array(I.shape[1::-1],dtype=float)*factor).astype(int).tolist()
	
	# dimensions must be multiple of the network stride
	w += (w%net_step!=0)*(net_step - w%net_step)
	h += (h%net_step!=0)*(net_step - h%net_step)

	# resizes image
	Iresized = cv2.resize(I,(w,h), interpolation = cv2.INTER_CUBIC)
	T = Iresized.copy()

	# Prepare to feed to IWPOD-NET
	T = T.reshape((1,T.shape[0],T.shape[1],T.shape[2]))

	#
	#  Runs LP detection network
	#
	start 	= time.time()
	Yr 		= model.predict(T)
	Yr 		= np.squeeze(Yr)
	elapsed = time.time() - start

	#
	# "Decodes" network result to find the quadrilateral corners of detected plates 
	#
	L,TLps = reconstruct_new (I, Iresized, Yr, out_size, threshold)

	return L,TLps,elapsed




def reconstruct_new(Iorig, I, Y, out_size, threshold=.9):

	net_stride 	= 2**4 
	side 	= ((208. + 40.)/2.)/net_stride # based on rescaling of training data

	Probs = Y[...,0]
	Affines = Y[...,-6:]  # gets the last six coordinates related to the Affine transform
	rx,ry = Y.shape[:2]

	#
	#  Finds cells with classification probability greater than threshold
	#
	xx,yy = np.where(Probs>threshold)
	WH = getWH(I.shape)
	MN = WH/net_stride

	#
	#  Warps canonical square to detected LP
	#
	vxx = vyy = 0.5 #alpha -- must match training script
	base = lambda vx,vy: np.matrix([[-vx,-vy,1.],[vx,-vy,1.],[vx,vy,1.],[-vx,vy,1.]]).T
	labels = []

	for i in range(len(xx)):
		y,x = xx[i],yy[i]
		affine = Affines[y,x]
		prob = Probs[y,x]

		mn = np.array([float(x) + .5,float(y) + .5])

		#
		#  Builds affine transformatin matrix
		#
		A = np.reshape(affine,(2,3))
		A[0,0] = max(A[0,0],0.)
		A[1,1] = max(A[1,1],0.)

		pts = np.array(A*base(vxx,vyy)) #*alpha
		pts_MN_center_mn = pts*side
		pts_MN = pts_MN_center_mn + mn.reshape((2,1))

		pts_prop = pts_MN/MN.reshape((2,1))


		labels.append(DLabel(0,pts_prop,prob))

	final_labels = nms(labels,.1)
	TLps = []  # list of detected plates

	if len(final_labels):
		final_labels.sort(key=lambda x: x.prob(), reverse=True)
		for i,label in enumerate(final_labels):
			ptsh 	= np.concatenate((label.pts*getWH(Iorig.shape).reshape((2,1)),np.ones((1,4))))
			t_ptsh 	= getRectPts(0,0, out_size[0] ,out_size[1])
			H = find_T_matrix(ptsh, t_ptsh)
			Ilp = cv2.warpPerspective(Iorig, H, out_size, flags = cv2.INTER_CUBIC, borderValue=.0)
			TLps.append(Ilp)
	return final_labels,TLps




