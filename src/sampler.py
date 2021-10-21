import cv2
import numpy as np
import random
import glob

from src.utils 	import im2single, getWH, hsv_transform
from src.label	import Label
from src.projection_utils import perspective_transform, find_T_matrix, getRectPts

#
#  Use UseBG if you want to pad distorted images with bakcground data
#
UseBG = True
bgimages = []
dim0 = 208
BGDataset = 'bgimages\\'  # directory with background images using to pad images in data augmentation


#
# Creates list with BG images
#
if UseBG:
	imglist = glob.glob(BGDataset + '*.jpg')
	for im in imglist:
		img = cv2.imread(im)
		factor = max(1, dim0/min(img.shape[0:2]))
		img = cv2.resize(img, (0,0), fx = factor, fy = factor).astype('float32')/255
		bgimages.append(img)

	
def random_crop(img, width, height):
	#
	#  generates random crop of img with desired size
	#
	or_height = img.shape[0]
	or_width = img.shape[1]
	top = int(np.random.rand(1)*(or_height - height))
	bottom = int(np.random.rand(1)*(or_width - width))
	crop = img[top:(top+height), bottom:(bottom+width),:]
	return crop


def GetCentroid(pts):
	#
	#  Gets centroids of a quadrilateral
	#
	return np.mean(pts, 1);


def ShrinkQuadrilateral(pts, alpha=0.75):
	#
	#  Shribks quadtrilateral by a factor alpha
	#
	centroid = GetCentroid(pts)
	temp = centroid + alpha * (pts.T - centroid)
	return temp.T


def LinePolygonEdges(pts):
	#
	# Finds the line equations of the polygon edges (given the verices in clockwise order)
	#
	lines = []
	for i in range(4):
		x1 = np.hstack((pts[:, i], 1))
		x2 = np.hstack((pts[:, (i + 1) % 4], 1))
		lines.append(np.cross(x1, x2))
	return lines


def insidePolygon(pt, lines):
	#
	#  Checks is a point pt is inisde quadrilateral given by pts. Must be all negative??
	#
	pth = np.hstack((pt, 1))
	allsigns = []
	output = True
	#
	# Scans all edges
	#
	for i in range(len(lines)):
		sig = np.dot(pth, lines[i])
		allsigns.append(sig)
		if sig < 0:
			output = False
			break
	return output


def labels2output_map(labelist, lpptslist, dim, stride, alfa=0.75):
	#
	#  Generates outpmut map with binary (classification) labels and quadrilateral corners (regression)
	#  label is the bounding box of the quadrilateral, and its locations are given in a list of plates
	#	lpptslist
	#
	
	dim0 = 208	# used to define the range of LP scales in the training procedure
	#
	#  Aveage LP side in output layer (with spatial dimension outsize)
	#
	side = ((float(dim0) + 40.) / 2.) / stride  # 7.75 when dim = 208 and stride = 16
	outsize = int(dim / stride)
	
	#
	# Prepares GT map with 9 channels
	#	
	Y = np.zeros((outsize, outsize, 2 * 4 + 1), dtype='float32')
	MN = np.array([outsize, outsize])
	WH = np.array([dim, dim], dtype=float)
	
	#
	#  Scans all annotated LPs in the image
	#
	for i in range(0, len(labelist)):
		
		
		#
		#  Gets corners and labels (labels igored at the moment)
		#
		lppts = lpptslist[i]
		label = labelist[i]
		
		#
		#  Gets location of bounding box of LP resized to output resolution
		#		
		tlx, tly = np.floor(np.maximum(label.tl(), 0.) * MN).astype(int).tolist()
		brx, bry = np.ceil(np.minimum(label.br(), 1.) * MN).astype(int).tolist()
		
		#
		#  Finds location of quadrilateral in the output resolution  
		#
		p_WH = lppts * WH.reshape((2, 1))
		p_MN = p_WH / stride


		#
		#  Finds line equations of shrunk quadrilaterals 
		#
		pts2 = (ShrinkQuadrilateral(lppts, alfa).T * MN).T;
		lines = LinePolygonEdges(pts2);
		
		#
		#  Scans LP bounding box and triggers a classification label if point is inside
		#  shrunk LP
		#
		for x in range(tlx, brx):
			for y in range(tly, bry):

				mn = np.array([float(x) + .5, float(y) + .5])		
				#
				#  Tests if current point is inside shrunk LP
				#				
				if insidePolygon(mn, lines):				
					#
					#  Translates LP points to the cell center
					#
					p_MN_center_mn = p_MN - mn.reshape((2, 1))					
					#
					#  Re-scales according to avergate LP side 
					#
					p_side = p_MN_center_mn / side
					#
					#  Defines classification label and re-scaled LP locations to be regressed
					#
					Y[y, x, 0] = 1.
					Y[y, x, 1:] = p_side.T.flatten()
		#
		#  Always set a true label at centroid if not fake LP (first test)
		#
		if (max(lppts[0,]) - min(lppts[0,]) > .01 and max(lppts[1,]) - min(lppts[1,]) > .01):
			cc = np.array(np.round(GetCentroid(lppts) * MN - 0.5), np.int8)
			#
			#  Centroid of LP in output resolution (round to smallest integer )
			#
			cc = np.array(np.round(GetCentroid(p_MN) - 0.5), np.int8)
			#
			#  Ensures that it is in a valid location of the output map
			#
			x = max(0, min(cc[0], outsize - 1))
			y = max(0, min(cc[1], outsize - 1))
			mn = np.array([float(x) + .5, float(y) + .5])
			p_MN_center_mn = p_MN - mn.reshape((2, 1))
			#
			#  Defines classification label and re-scaled LP locations to be regressed
			#
			p_side = p_MN_center_mn / side
			Y[y, x, 0] = 1.
			Y[y, x, 1:] = p_side.T.flatten()
	return Y



def pts2ptsh(pts):
	#
	#  Gets homogeneous coordinates
	#
	return np.matrix(np.concatenate((pts, np.ones((1, pts.shape[1]))), 0))


def project(I, T, pts, dim):
	#
	#  Projects image I and points pts according to matrix T
	#
	ptsh = np.matrix(np.concatenate((pts, np.ones((1, 4))), 0))
	ptsh = np.matmul(T, ptsh)
	ptsh = ptsh / ptsh[2]
	ptsret = ptsh[:2]
	ptsret = ptsret / dim
	Iroi = cv2.warpPerspective(I, T, (dim, dim), borderValue=.0, flags = cv2.INTER_CUBIC)
	return Iroi, ptsret


def project_all(I, T, ptslist, dim, bgimages = bgimages):
	#
	#  Warps image I to desired dimensions using matrix T. if bgimage is not empty,
	#  completes with background
	#  Also projects LP coordinates given in ptslist to keep coherence
	#
	#
	outptslist = []
	#
	#  Scans annotated LPs and warps them
	#
	for pts in ptslist:
		ptsh = np.matrix(np.concatenate((pts, np.ones((1, 4))), 0))
		ptsh = np.matmul(T, ptsh)
		ptsh = ptsh / ptsh[2]
		ptsret = ptsh[:2]
		ptsret = ptsret / dim
		outptslist.append(np.array(ptsret))
	#
	#  Warps input image (possibly padding with BG images)
	#
	Iroi = cv2.warpPerspective(I, T, (dim, dim), borderValue=(.5,.5,.5), flags=cv2.INTER_CUBIC)
	if len(bgimages) > 0:
		bgimage = bgimages[int(np.random.rand()*len(bgimages))]
		bgimage = random_crop(bgimage, dim, dim)
		bw = np.ones(I.shape)
		bw = cv2.warpPerspective(bw, T, (dim, dim), borderValue= (0, 0, 0), flags=cv2.INTER_LINEAR)
		Iroi[bw == 0] = bgimage[bw == 0]
	return Iroi, outptslist

def randomblur(img):
	#
	#  Applies random blur to image
	#
	maxblur = np.min(img.shape)/10
	sig = abs(np.random.normal(0, .1))*maxblur
	ksize = 2*int(0.5 + 2*sig) + 1
	out =  cv2.GaussianBlur(img, (ksize, ksize), sig)
	return out

def flip_image_and_ptslist(I, ptslist):
	#
	#  Applies random flip to image and labels
	#
	I = cv2.flip(I, 1)
	for i in range(len(ptslist)):
		pts = ptslist[i]
		pts[0] = 1. - pts[0]
		idx = [1, 0, 3, 2]
		pts = pts[..., idx]
		ptslist[i] = pts
	return I, ptslist


def flip_image_and_pts(I, pts):
	I = cv2.flip(I, 1)
	pts[0] = 1. - pts[0]
	idx = [1, 0, 3, 2]
	pts = pts[..., idx]
	return I, pts


def augment_sample(I, shapelist, dim, maxangle = 2 * np.array([65.,65.,55.]), maxsum = 140):
	#
	#  Main augmentation function. Generates an augmented version
	#  of input image I and the corresponding LP corners given in shapelist
	#
	
	#
	#  Input is image I, list of shape elements (shape), and input dim
	#

	#
	#  Gets first LP corners and label
	#
	pts = shapelist[0].pts
	vtype = shapelist[0].text
	ptslist = []
	#
	#  List of ROI corners
	#
	for entry in shapelist:
		ptslist.append(entry.pts)

	#
	#  maximum 3D rotation angles
	#
	angles = (np.random.rand(3) - 0.5) * maxangle

	if sum(np.abs(angles)) > maxsum:
		angles = (angles/angles.sum())*(maxangle/maxangle.sum())

	#
	# Normalizes intensities to [0,1]
 	#	
	I = im2single(I)
	#
	#  Possible negative of the image
	#
	if np.random.uniform(0,1) < 0.05:
		I = 1 - I;
	
	#
	# Possible blur
	#	
	if np.random.rand() < 0.15:
		I = randomblur(I)
	
	
	#
	#  Gets image dimensions
	#
	iwh = getWH(I.shape)
	
	#
	#  Checks is annotation is a real or fake plate
	#

	if (pts[0][1] - pts[0][0] > .002): ## if not fake plate

		for i in range(len(ptslist)):
			#
			#  LP region from relative to absolute coordinates
			#
			ptslist[i] = ptslist[i] * iwh.reshape((2, 1))
		
		#
		#  Target aspect ratio of the LP (bike or car)
		#
		if vtype == 'bike':
			whratio = random.uniform(1.25, 2.5)
		else:
			whratio = random.uniform(2.5, 4.5)
		
		#
		#  Width of LP in training image 
		#
		dim0 = 208 # augments data w.r.t. a fixed resolution
		
		#
		#  Defines range of LP widths w.r.t to baseline resolution dim0 = 208
		#
		wsiz = random.uniform(dim0*0.2, dim0*1.0)
		
		#
		#  Defines height based on width and aspect ratio
		#
		hsiz = wsiz/whratio
		
		#
		#  Defines horizontal and vertical offsets
		#
		dx = random.uniform(0.,dim - wsiz)
		dy = random.uniform(0.,dim - hsiz)
		
		#
		#  Warps annotated plate to a rectified rectangle - frontal view
		#
		pph = getRectPts(dx, dy, dx+wsiz, dy+hsiz)
		pts = pts*iwh.reshape((2,1))
		T = find_T_matrix(pts2ptsh(pts), pph)

		#
		# Finds 3D rotation matrix based on angles
		#
		H = perspective_transform((dim,dim), angles=angles)

		#
		#  Applies 3D rotation to rectification transform
		#
		H = np.matmul(H,T)

		#
		# projects images and labels according to 3D rotation
		#
		Iroi, ptslist = project_all(I, H, ptslist, dim)
		pts = ptslist[0]
	else:  # if fake plate
		#
		#  Random BG crop if no plate is present, resizes in x and y
		#	
		rfactorx = max(dim/I.shape[0],  0.5 + 0.5*np.random.rand())
		rfactory = max(rfactorx, dim/I.shape[1])
		Iroi = cv2.resize(I, (0,0), fx = rfactorx, fy = rfactory)
		Iroi = random_crop(I, dim, dim)
		
	#
	#	Just a sanity check, test should never hold
	#	
	if (Iroi.shape[0] < dim):
		cv2.imshow('Imagem', I)
		print(I.shape)
		print(Iroi.shape)
		cv2.waitKey()


	#
	#  Set of non-geometric transforms
	#
	
	#
	# Color transformations in HSV space
	#
	hsv_mod = np.random.rand(3).astype('float32')
	hsv_mod = (hsv_mod - .5)*.4
	hsv_mod[0] *= 360
	Iroi = hsv_transform(Iroi, hsv_mod)
	
	#
	#  Finds bounding boxes of all annotated plates
	#
	labelist = []
	for pts in ptslist:
		tl, br = pts.min(1), pts.max(1)
		labelist.append(Label(0, tl, br))

	#
	#  Returns image, and two lists with LP lalbels and quadrilateral points
	#
	return Iroi, labelist, ptslist

