upsamplepath = '../upsampling'
import sys

if not(upsamplepath in sys.path):
	sys.path.append(upsamplepath)


import numpy as np
import cv2
import time

from os.path import splitext
#from src_upsample.keras_utils_upsample import run_gray_upsample

from src.label import Label
from src.utils import getWH, nms
from src.projection_utils import getRectPts, find_T_matrix
#from ProcessOcrPlatesClass import ComputePlateSize

def adjust_lp_image(model, img, factor):
	#
	# dobules the imaga factor times
	#
	out = img.copy()
	for k in range(factor):
		out = run_gray_upsample(model, out)
#	cv2.imshow('Original plate', img)
#	cv2.imshow('Upsampled plate', out)
#	cv2.waitKey()
	return out


class DLabel (Label):

	def __init__(self,cl,pts,prob):
		self.pts = pts
		tl = np.amin(pts,1)
		br = np.amax(pts,1)
		Label.__init__(self,cl,tl,br,prob)

def save_model(model,path,verbose=0):
	path = splitext(path)[0]
	model_json = model.to_json()
	with open('%s.json' % path,'w') as json_file:
		json_file.write(model_json)
	model.save_weights('%s.h5' % path)
	if verbose: print('Saved to %s' % path)

def load_model(path,custom_objects={},verbose=0):
	from keras.models import model_from_json

	path = splitext(path)[0]
	with open('%s.json' % path,'r') as json_file:
		model_json = json_file.read()
	model = model_from_json(model_json, custom_objects=custom_objects)
	model.load_weights('%s.h5' % path)
	if verbose: print('Loaded from %s' % path)
	return model



def detect_lp_width(model, I,  MAXWIDTH, net_step, out_size, threshold, up_model = []):
	
	#
	#  MUDANCA JUNG: width is fixed, based on MAXWIDTH
	#
	#MAXWIDTH = 288
	
	factor = min(1, MAXWIDTH/I.shape[1])
	w,h = (np.array(I.shape[1::-1],dtype=float)*factor).astype(int).tolist()
	
	w += (w%net_step!=0)*(net_step - w%net_step)
	h += (h%net_step!=0)*(net_step - h%net_step)
	#print('Width of resized image fed to IWPOD-NET: %d' % w)
	#print('Aspect ratio: %f:' % (1.0*w/h))
	#print('Dimensions: %d, %d' % (w, h))

	Iresized = cv2.resize(I,(w,h), interpolation = cv2.INTER_CUBIC)
	#cv2.imshow('Input to WPOD', Iresized)
	#cv2.waitKey()
	#print(Iresized.shape)

	T = Iresized.copy()
	T = T.reshape((1,T.shape[0],T.shape[1],T.shape[2]))

	#
	#  Runs LP detection network
	#

	start 	= time.time()
	Yr 		= model.predict(T)
	Yr 		= np.squeeze(Yr)
	#
	#  Shows prob map
	#
	#gt = np.concatenate( (cv2.cvtColor(Iresized, cv2.COLOR_BGR2GRAY), cv2.resize(Yr[:,:,0], (w,h), interpolation = cv2.INTER_CUBIC)), axis = 1)
	#cv2.imshow('Prob map', gt)
	#cv2.waitKey()
	elapsed = time.time() - start

	L,TLps = reconstruct_new (I, Iresized, Yr, out_size, threshold, up_model)

	return L,TLps,elapsed




def reconstruct_new(Iorig, I, Y, out_size, threshold=.9, up_model = []):

	AreaTh = 0.1*240*80;
	#
	# If plate is too small, performs deep upsampling if upsampling network is provided
	#
	net_stride 	= 2**4
	side 	= ((208. + 40.)/2.)/net_stride # 7.75

	Probs = Y[...,0]
	#Affines = Y[...,2:]
	Affines = Y[...,-6:]  # getrs the last six coordinates
	rx,ry = Y.shape[:2]
	#ywh = Y.shape[1::-1]
	#iwh = np.array(I.shape[1::-1],dtype=float).reshape((2,1))

	xx,yy = np.where(Probs>threshold)
	
	#print(xx)

	WH = getWH(I.shape)
	MN = WH/net_stride

	vxx = vyy = 0.5 #alpha -- must match training script

	base = lambda vx,vy: np.matrix([[-vx,-vy,1.],[vx,-vy,1.],[vx,vy,1.],[-vx,vy,1.]]).T
	labels = []

	for i in range(len(xx)):
		y,x = xx[i],yy[i]
		affine = Affines[y,x]
		prob = Probs[y,x]

		mn = np.array([float(x) + .5,float(y) + .5])

		A = np.reshape(affine,(2,3))
		A[0,0] = max(A[0,0],0.)
		A[1,1] = max(A[1,1],0.)

		pts = np.array(A*base(vxx,vyy)) #*alpha
		pts_MN_center_mn = pts*side
		pts_MN = pts_MN_center_mn + mn.reshape((2,1))

		pts_prop = pts_MN/MN.reshape((2,1))


		labels.append(DLabel(0,pts_prop,prob))

	final_labels = nms(labels,.1)
	TLps = []

	if len(final_labels):
		final_labels.sort(key=lambda x: x.prob(), reverse=True)
		for i,label in enumerate(final_labels):
#			adpts = label.pts*getWH(Iorig.shape).reshape((2,1))
			ptsh 	= np.concatenate((label.pts*getWH(Iorig.shape).reshape((2,1)),np.ones((1,4))))
	
			t_ptsh 	= getRectPts(0,0, out_size[0] ,out_size[1])
			H = find_T_matrix(ptsh, t_ptsh)
			Ilp = cv2.warpPerspective(Iorig, H, out_size, flags = cv2.INTER_CUBIC, borderValue=.0)
#				Ilp = cv2.warpPerspective(Iorig, H, out_size, flags = cv2.INTER_CUBIC, borderValue=.0)
			TLps.append(Ilp)

	return final_labels,TLps




