import numpy as np
import cv2
import sys
from .drawing_utils import draw_losangle

from glob import glob

def trimmed_mean(lista):
	#
	#
	#
	lista2 = np.sort(lista)
	#
	#  removes smalles and largest, and computes mean
	#
	if (len(lista2) > 2): # if more than two elements:
		media = np.mean(lista2[1:-1])
	else:
		media = np.mean(lista2)
	return media

def FindAspectRatio(pts):
	#
	#  Estimates the aspect raio of a quadrilateral
	#
	dsts = []
	if len(pts) > 0:
		for i in range(4):
			dsts.append(np.linalg.norm(pts[:,i] - pts[:,(i+1)%4]))
		return ( dsts[0] + dsts[2]) / ( dsts[1] + dsts[3])
	else:
		return []
	

def letterbox_image_cv2_float(image, expected_size):
  ih, iw, _ = image.shape
  eh, ew = expected_size
  scale = min(eh / ih, ew / iw)
  nh = int(ih * scale)
  nw = int(iw * scale)

  image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
  new_img = np.full((eh, ew , 3), 0.5, dtype='float32')
  # fill new image with the resized image and centered it
  new_img[(eh - nh) // 2:(eh - nh) // 2 + nh,
          (ew - nw) // 2:(ew - nw) // 2 + nw,
          :] = image.copy()
  return new_img


def FindBestLP(Llp, LlpImgs):
	#
	#  Find and returns only the LP with the highest prob
	#
	probs = [];
	for lp in Llp:
		probs.append(lp.prob())
		#print('LP probability: %1.2f' % (lp.prob()))
	if len(probs) > 0:
		ind  = np.argmax(probs)
		return [Llp[ind]], [LlpImgs[ind]]
	else:
		return [], []



def im2single(I):
	assert(I.dtype == 'uint8')
	return I.astype('float32')/255.


def getWH(shape):
	return np.array(shape[1::-1]).astype(float)


def IOU(tl1,br1,tl2,br2):
	wh1,wh2 = br1-tl1,br2-tl2
	assert((wh1>=.0).all() and (wh2>=.0).all())

	intersection_wh = np.maximum(np.minimum(br1,br2) - np.maximum(tl1,tl2),0.)
	intersection_area = np.prod(intersection_wh)
	area1,area2 = (np.prod(wh1),np.prod(wh2))
	union_area = area1 + area2 - intersection_area;
	return intersection_area/union_area


def IOU_labels(l1,l2):
	return IOU(l1.tl(),l1.br(),l2.tl(),l2.br())

def IOU_labels_darkflow(l1,l2):
	tl1 = np.array([l1['topleft']['x'], l1['topleft']['y']])
	br1 = np.array([l1['bottomright']['x'], l1['bottomright']['y']])

	tl2 = np.array([l2['topleft']['x'], l2['topleft']['y']])
	br2 = np.array([l2['bottomright']['x'], l2['bottomright']['y']])

	return IOU(tl1,br1,tl2,br2)


def IOU_centre_and_dims(cc1,wh1,cc2,wh2):
	return IOU(cc1-wh1/2.,cc1+wh1/2.,cc2-wh2/2.,cc2+wh2/2.)


def nms_darkflow(Labels, iou_threshold=.5):
	SelectedLabels = []
	Labels.sort(key=lambda l: l['confidence'], reverse=True)

	for label in Labels:
		non_overlap = True
		for sel_label in SelectedLabels:
			if IOU_labels_darkflow(label, sel_label) > iou_threshold:
				non_overlap = False
				break

		if non_overlap:
			SelectedLabels.append(label)

	return SelectedLabels


#
#  Process bike LP data
#
def nms_bike_darkflow_target(ocr, iou_threshold=.3, target_characters = np.inf):
	#
	#  can try first nms, then find ocr-rows
	#
	top = [] # top row
	bottom = [] # bottom row
	for p in ocr:
		if(p['topleft']['y'] + p['bottomright']['y'] < 80):
			top.append(p)
		else:
			bottom.append(p)
	top = nms_darkflow_target (top, .3, 3)  # three characters top
	bottom = nms_darkflow_target (bottom, .3, 4)  #four characters bottom
	return top + bottom

def get_bike_string(ocr):
	#
	#  Given a listr of ocr detections, find the mean vetical point as splits into two rows
	#  of characters
	#
	
	#
	#  JUNG -- seguir aqui!!
	#
	
	top = [] # top row
	bottom = [] # bottom row
	
	centers =  [ (o['topleft']['y']+o['bottomright']['y'])/2 for o in ocr]
	#
	# Find alfa-trimmed mean
	#
	media = trimmed_mean(centers)
	for o in ocr:
		if (o['topleft']['y']+o['bottomright']['y'])/2 <= media:
			top.append(o)
		else:
			bottom.append(o)	
	top.sort(key=lambda l: l['topleft']['x'], reverse = False)
	bottom.sort(key=lambda l: l['topleft']['x'], reverse = False)

	return top + bottom


#
#  Process brazlian bike LP data
#
def get_bike_string_brazilian(ocr):
	#
	#  Given a listr of ocr detection with 7 characters, extracts the final string
	#
	top = [] # top row
	bottom = [] # bottom row
	ocr.sort(key=lambda l: l['topleft']['y'] + l['bottomright']['y'], reverse = False)

	top = ocr[0:3]
	top.sort(key=lambda l: l['topleft']['x'], reverse = False)

	bottom = ocr[3:7]
	bottom.sort(key=lambda l: l['topleft']['x'], reverse = False)

	return top + bottom

def get_bike_string_brazil(ocr):
	#
	#  Given a listr of ocr detection with 7 characters, extracts the final string
	#
	top = [] # top row
	bottom = [] # bottom row
	ocr.sort(key=lambda l: l['topleft']['y'] + l['bottomright']['y'], reverse = False)
	

	top = ocr[0:3]
	top.sort(key=lambda l: l['topleft']['x'], reverse = False)

	bottom = ocr[3:7]
	bottom.sort(key=lambda l: l['topleft']['x'], reverse = False)

	return top + bottom



#
#  NMS with target range of characters. Runs OCR with a small threshold, and evaluates the number of detections
#  with a default threshold (0.4). If number of detections outside the range, increases or decrases effective threshold	
#
def nms_darkflow_range(Labels, iou_threshold =.25, min_threshold = 0.4, min_characters = 0, max_characters = np.inf):
	SelectedLabels = []
	#
	#  Sorts detection in descending order of confidence
	#
	Labels.sort(key=lambda l: l['confidence'], reverse=True)
	#
	#  Finds labels with small IoU
	#
	for label in Labels:
		non_overlap = True
		for sel_label in SelectedLabels:
			if IOU_labels_darkflow(label, sel_label) > iou_threshold:
				non_overlap = False
				break
		if non_overlap:
			SelectedLabels.append(label)
#			print(label)

		#
		#  Stops if minumum number is reached and ocr confidence is low
		#
		if len(SelectedLabels) > min_characters and SelectedLabels[-1]['confidence'] < min_threshold:
			del(SelectedLabels[-1])
			break
		#
		#  Stops if maximum number is reached
		#
		if len(SelectedLabels) == max_characters:
			break
	return SelectedLabels



#
#  NMS with target number of characters
#
def nms_darkflow_target(Labels, iou_threshold =.25, target_characters = np.inf):
	SelectedLabels = []
	#
	#  Sorts detection in descending order of confidence
	#
	Labels.sort(key=lambda l: l['confidence'], reverse=True)
	
	#
	#  Finds labels with small IoU
	#
	for label in Labels:
		non_overlap = True
		for sel_label in SelectedLabels:
			if IOU_labels_darkflow(label, sel_label) > iou_threshold:
				non_overlap = False
				break
		if non_overlap:
			SelectedLabels.append(label)
		#
		#  Stops when number of characters is reached
		#
		if len(SelectedLabels) ==  target_characters:
			break
	return SelectedLabels

def generate_bb_yolo(ocr_entry, width = 240, height = 80):
	#
	# For an image with dimensions height x width, generates a tuple with the BB in YOLO's format
	#
	bbwidth = ocr_entry['bottomright']['x'] - ocr_entry['topleft']['x']
	bbheight = ocr_entry['bottomright']['y'] - ocr_entry['topleft']['y']
	x = (ocr_entry['topleft']['x'] + bbwidth/2)/width
	y = (ocr_entry['topleft']['y'] + bbheight/2)/height
	w = bbwidth/width
	h = bbheight/height
	return (x,y,w,h)

def nms(Labels,iou_threshold=.5):

	SelectedLabels = []
	Labels.sort(key=lambda l: l.prob(),reverse=True)

	for label in Labels:
		non_overlap = True
		for sel_label in SelectedLabels:
			if IOU_labels(label,sel_label) > iou_threshold:
				non_overlap = False
				break

		if non_overlap:
			SelectedLabels.append(label)

	return SelectedLabels


def image_files_from_folder(folder,upper=True):
	extensions = ['jpg','jpeg','png']
	img_files  = []
	for ext in extensions:
		img_files += glob('%s/*.%s' % (folder,ext))
		if upper:
			img_files += glob('%s/*.%s' % (folder,ext.upper()))
	return img_files


def is_inside(ltest,lref):
	return (ltest.tl() >= lref.tl()).all() and (ltest.br() <= lref.br()).all()


def crop_region(I,label,bg=0.5):

	wh = np.array(I.shape[1::-1])

	ch = I.shape[2] if len(I.shape) == 3 else 1
	tl = np.floor(label.tl()*wh).astype(int)
	br = np.ceil (label.br()*wh).astype(int)
	outwh = br-tl

	if np.prod(outwh) == 0.:
		return None

	outsize = (outwh[1],outwh[0],ch) if ch > 1 else (outwh[1],outwh[0])
	if (np.array(outsize) < 0).any():
		pause()
	Iout  = np.zeros(outsize,dtype=I.dtype) + bg

	offset 	= np.minimum(tl,0)*(-1)
	tl 		= np.maximum(tl,0)
	br 		= np.minimum(br,wh)
	wh 		= br - tl

	Iout[offset[1]:(offset[1] + wh[1]),offset[0]:(offset[0] + wh[0])] = I[tl[1]:br[1],tl[0]:br[0]]

	return Iout

def hsv_transform(I,hsv_modifier):
	I = cv2.cvtColor(I,cv2.COLOR_BGR2HSV)
	I = I + hsv_modifier
	return cv2.cvtColor(I,cv2.COLOR_HSV2BGR)

def IOU(tl1,br1,tl2,br2):
	wh1,wh2 = br1-tl1,br2-tl2
	assert((wh1>=.0).all() and (wh2>=.0).all())

	intersection_wh = np.maximum(np.minimum(br1,br2) - np.maximum(tl1,tl2),0.)
	intersection_area = np.prod(intersection_wh)
	area1,area2 = (np.prod(wh1),np.prod(wh2))
	union_area = area1 + area2 - intersection_area;
	return intersection_area/union_area

def IOU_centre_and_dims(cc1,wh1,cc2,wh2):
	return IOU(cc1-wh1/2.,cc1+wh1/2.,cc2-wh2/2.,cc2+wh2/2.)


def show(I,wname='Display'):
	cv2.imshow(wname, I)
	cv2.moveWindow(wname,0,0)
	key = cv2.waitKey(0) & 0xEFFFFF
	cv2.destroyWindow(wname)
	if key == 27:
		sys.exit()
	else:
		return key

#
#  Relates WPOD-NET with vehicle
#
def adjust_pts(pts, result):
	tl = np.array([result['topleft']['x'], result['topleft']['y']],dtype = float).reshape(2,1)
	wh = np.array([result['bottomright']['x'] - result['topleft']['x'], result['bottomright']['y'] - result['topleft']['y']],dtype = float).reshape(2,1)
	return pts*wh + tl  # points of the quadrilateral

#
#  Adds BBs and class labels to detected characters
#
def print_digits(Ilp, ocr_list, font = 1):
	#heights = []
	#widths = []
	if np.max(Ilp) > 2:
		rec_color = (0, 255, 0)
		dig_color = (0, 0, 255)
	else:
		rec_color = (0, 1, 0)
		dig_color = (0, 0, 1)
		
	for ocr in ocr_list:
		tlx = ocr['topleft']['x'];
		tly = ocr['topleft']['y'];
		brx = ocr['bottomright']['x'];
		bry = ocr['bottomright']['y'];
	#	heights.append(bry - tly)
	#	widths.append(brx - tlx)
		cv2.rectangle(Ilp, (tlx,tly), (brx,bry), rec_color, thickness=1)
		cv2.putText(Ilp, ocr['label'], (tlx,tly), cv2.FONT_HERSHEY_SIMPLEX, font, dig_color, 2 )
	#print(np.mean(heights))
	#print(np.mean(widths))


def IOU_Quadrilateral(pts1, pts2):
	allpts = np.concatenate([pts1, pts2], axis = 1)
	xmin, ymin = np.min(allpts, 1)
	xmax, ymax = np.max(allpts, 1)
	dx = xmax - xmin
	dy = ymax - ymin
	
	#
	# First binary image
	#
	img1 = np.zeros((dy, dx)).astype(np.uint8)
	draw_losangle(img1, pts1 -  np.array([xmin, ymin]).reshape(2,1), 1, 1)
	translated_centroid = np.mean(pts1 ,1) -  np.array([xmin, ymin])
	cv2.floodFill(img1, None, tuple(np.uint16(translated_centroid)), 1)
	
	
	#
	# Second binary image
	#
	img2 = np.zeros((dy, dx)).astype(np.uint8)
	draw_losangle(img2, pts2 -  np.array([xmin, ymin]).reshape(2,1), 1, 1)
	translated_centroid = np.mean(pts2 ,1) -  np.array([xmin, ymin])
	cv2.floodFill(img2, None, tuple(np.uint16(translated_centroid)), 1)

	#
	#  Union
	#
	img_union = img1 + img2
	img_union[img_union > 1] = 1
	
	#
	# Intersection
	#
	img_inter = img1*img2


	return np.sum(img_inter)/np.sum(img_union)
