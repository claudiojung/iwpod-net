import numpy as np
import cv2
from src.keras_utils 			import detect_lp
from src.utils 					import im2single, nms_darkflow, nms_darkflow_target, adjust_pts, print_digits
#from src.label 					import Shape, writeShapes
from src.drawing_utils			import draw_losangle


#
#  Runs vehicle detector
#
def detect_vechicle(tfnet_yolo, imgcv):
	result = tfnet_yolo.return_predict(imgcv)
	vehicles = []
	for det in result:
		if (det['label'] in ['car','bus']):
			vehicles.append(det)
	return vehicles

#
#  Scans all vehicles for LPs
#
def scan_vehicles(vehicles,  imgcv, wpod_net, lp_threshold):
	#
	#  Adds a fake detection with the full image if no vehicle is detected
	#
	plate = []
	plateimgs = []
	if len(vehicles) == 0:
		vehicles = [{'label': 'car',  'confidence': 1,  'topleft': {'x': 1, 'y': 1}, 'bottomright': {'x': imgcv.shape[1], 'y': imgcv.shape[0]}}]

	#
	#  Scans all vehicles
	#
	for car in vehicles:
		#
		#  Crops vehicle from image
		#
		tlx = car['topleft']['x'];
		tly = car['topleft']['y'];
		brx = car['bottomright']['x'];
		bry = car['bottomright']['y'];
		Ivehicle = imgcv[tly:bry, tlx:brx]
		#cv2.imshow('Vehicle', Ivehicle); cv2.waitKey(); cv2.destroyAllWindows()
		#
		#  Adjusts input resolution for WPOD-NET
		#
		WPODResolution = 416 #FOr Yolo-voc
		#WPODResolution = 608 #For Yolov2
		ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
		side  = int(ratio*288.)
		bound_dim = min(side + (side%(2**4)), WPODResolution)
		#print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))
		#
		#  Runs WPOD-NET
		#
		Llp,LlpImgs,_ = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, 2**4, (240,80), lp_threshold)
		#print(len(Llp))
		#
		#  returns a list of plates per car
		#
		plate.append(Llp)
		plateimgs.append(LlpImgs)
	return (plate, plateimgs, vehicles)



# =============================================================================
# #
# #  Finds warped LPs
# #
# def ocr_plates(tfnet_ocr, result,  imgcv, platelist, plateimgslist):
# 	listocr = [];
# 	listimgs = [];
# 	numplates = 0;
# 	for LlpImgs in plateimgslist:
# 		if len(LlpImgs):
# 			Llp = platelist[numplates]
# 			Ilp = LlpImgs[0]
# 			Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
# 			Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
# 			print(Ilp.shape)
# 			ptspx = adjust_pts(Llp[0].pts, result[numplates])
# 			draw_losangle(imgcv, ptspx, (0, 0, 255), 3)
# 			#
# 			#   Run OCR
# 			#
# 			ocr = tfnet_ocr.return_predict(Ilp * 255.)
# 			#
# 			# Applies nms
# 			#
# 			ocr = nms_darkflow(ocr)
# 			#
# 			#  Image with OCR-ed LP
# 			#
# 			print_digits(Ilp, ocr)
# 			# sorts OCR results and generrates string
# 			ocr.sort(key=lambda x: x['topleft']['x'])
# 			lp_str = ''.join([r['label'] for r in ocr])
# 			listocr.append(lp_str)
# 			listimgs.append(Ilp)
# 		numplates = numplates + 1;
# 	return listocr, listimgs
#
# =============================================================================

#
#  Finds warped LPs
#
def ocr_plates(tfnet_ocr, result,  imgcv, platelist, plateimgslist):
	listocr = [];
	listimgs = [];
	lp_str = []
	#
	#  Scans all detected  vehicles
	#
	for numcars in range(0, len(result)):
		LlpImgs = plateimgslist[numcars]
		Llp = platelist[numcars]
		#
		#  Scans detected Lps per vehicle
		#
		for k in range(0, len(LlpImgs)):
			Ilp = LlpImgs[k]
			Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
			Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
			#print(Ilp.shape)
			#
			#  Adjust warped plate to full image
			#
			ptspx = adjust_pts(Llp[k].pts, result[numcars])
			draw_losangle(imgcv, ptspx, (0, 0, 255), 3)
			#
			#   Run OCR
			ocr = tfnet_ocr.return_predict(Ilp * 255.)
			#
			# Applies nms
			#
			#ocr = nms_darkflow(ocr, 0.3)
			if config.isBrazilianLP:
				ocr = nms_darkflow_target(ocr, 0.3, 7)
			else:
				ocr = nms_darkflow_target(ocr, 0.3)
			#
			#  Image with OCR-ed LP
			#
			print_digits(Ilp, ocr)
			
			# sorts OCR results and generates string
			ocr.sort(key=lambda x: x['topleft']['x'])
			lp_str = ''.join([r['label'] for r in ocr])
			# If Brazlian LP, post-process OCR result
			if config.isBrazilianLP:
				lp_str = SwapCharactersLPBrazilian(lp_str)
			listocr.append(lp_str)
			listimgs.append(Ilp)
# =============================================================================
# 	if len(lp_str) < 7:
# 		cv2.imshow('Orig', imgcv); 
# 		if len(lp_str) > 0:
# 			cv2.imshow('OCR', Ilp); cv2.waitKey(); cv2.destroyAllWindows()
# 		else:
# 			cv2.waitKey(); cv2.destroyAllWindows()
# =============================================================================
	return listocr, listimgs

#
#  Saves images and prints txt files
#
def save_print_files(listocr, listimgs, outputdir, rootname):
	for i in range(0, len(listocr)):
		ocr = listocr[i]
		img = listimgs[i]
		if config.SaveTxt:
			with open(outputdir + '%s_str_%d.txt' % (rootname, i + 1),'w') as f:
				f.write(ocr + '\n')
		if config.SaveImages:
			cv2.imwrite(outputdir + rootname +  '_plate_%d' % (i + 1) + '_ocr.png', img*255.)


def run_all(tfnet_yolo, imgcv, wpod_net, lp_threshold, tfnet_ocr, outputdir, rootname):
	#
	#  Find cars
	#
	result = detect_vechicle(tfnet_yolo, imgcv)
	#result = [{'label': 'car',  'confidence': 1,  'topleft': {'x': 1, 'y': 1}, 'bottomright': {'x': imgcv.shape[1], 'y': imgcv.shape[0]}}]
	#
	#  Find LPs
	#
	platelist, plateimgslist, result = scan_vehicles(result,  imgcv, wpod_net, lp_threshold)
	#
	# Performs OCR at detected LPs
	#
	listocr, listimgs = ocr_plates(tfnet_ocr, result,  imgcv, platelist, plateimgslist)
	#
	#  Saves images and print OCR
	#
	save_print_files(listocr, listimgs, outputdir, rootname)
	return listocr

#
#   Ebforce the first 3 characters to be lettters, and the next four digits
#
def SwapCharactersLPMercosul(instring):
	#
	#  Format; AAA0A00
	#
	outstring = list(instring);
	if len(instring) == 7:
		for i in range(0,3):
			outstring[i] = imposeLetter(instring[i])
		outstring[3] = imposeDigit(instring[3])
		outstring[4] = imposeLetter(instring[4])
		for i in range(5,7):
			outstring[i] = imposeDigit(instring[i])
	return "".join(outstring)




def SwapCharactersLPBrazilian(instring):
	#
	#  Format AAA0000
	#
	outstring = list(instring);
	if len(instring) == 7:
		for i in range(0,3):
			outstring[i] = imposeLetter(instring[i])
		for i in range(3,7):
			outstring[i] = imposeDigit(instring[i])
	return "".join(outstring)


def SwapCharactersLPChinese(instring):
	#
	#  Format FLAAAAA (A is any), F is a fake chinese character
	#
	
	#
	#  If seven characters are detected, discards the first one
	#
	outstring = list(instring);
	if len(instring) == 7:
		outstring = outstring[1:]
	if len(outstring) == 6:
			outstring[0] = imposeLetter(outstring[0])
	return "".join(outstring)


def imposeLetterString(instring):
	#
	#  Transform all characters into letters
	#
	outstring = list(instring);
	for i in range(0, len(instring)):
		outstring[i] = imposeLetter(instring[i])
	return "".join(outstring)

def imposeDigitString(instring):
	#
	#  Transform all characters into digits
	#
	outstring = list(instring);
	for i in range(0, len(instring)):
		outstring[i] = imposeDigit(instring[i])
	return "".join(outstring)


def imposeLetter(inchar):
	diglist = '0123456789'
	#charlist = 'OIZUASGJBR'
	charlist = 'OIZBASETBS'
	outchar = inchar
	if inchar.isdigit():
		ind = diglist.index(inchar)
		outchar = charlist[ind]
	return outchar

def imposeDigit(inchar):
	#charlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
	#diglist = '48605568113133030451877992'
	charlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
	diglist =  '48006661113191080651011017'
	outchar = inchar
	if inchar.isalpha():
		ind = charlist.index(inchar)
		outchar = diglist[ind]
	return outchar

def ClassifyPlate(img, ocr):
	#
	#  Classifies blate as brazilian or mercosul. Heuristic: compare horizonntal stripes above and vbelow
	# OCR-ed characters
	# 
	
	Debug = False
	
	#
	# Find upper and lower stripes, as well as average character height
	#
	
	# y - vertical, x - horizontal
	offset = 4;
	vminy = []
	vmaxy = []
	vminx = []
	vmaxx = []
	vheight = []
	for car in ocr:
		vminy.append(car['topleft']['y']);
		vmaxy.append(car['bottomright']['y']);
		vminx.append(car['topleft']['x']);
		vmaxx.append(car['bottomright']['x']);
		vheight.append(vmaxy[-1] - vminy[-1]);
	
#	if Debug:
#		print(vminy)
#		print(vmaxy)
#		print(vminx)
#		print(vmaxx)
#		print(vheight)
	
	#
	#  Finds min and max boundaries, respecting an offset
	#
		
	miny = max(offset, min(vminy))
	maxy = min(79 - offset, max(vmaxy))
	minx = max(offset, min(vminx))
	maxx = min(239 - offset, max(vmaxx))
	height = sum(vheight) / len(vheight);
	
	
	if Debug:
		print([miny, maxy, minx, maxx, height])
		imp = img.copy()
	#print('Height: %1.2f'%height)
	
	#
	#  Sripe height: half character height, minimum height one
	#
	u_height =  int(max(1, min(miny, height/2)))
	l_height = int(max(1, min(u_height/4, 79 - maxy)))
	
	
	
	#
	#  Upper and lower stripe
	#
	#cv2.imshow('Placa', img); cv2.waitKey(); cv2.destroyAllWindows()
	channel = 2;
	img0 = img[:,:,channel].copy()


	#up_intensity = np.median(img0[minx:maxx, miny:maxy])
	#low_intensity = np.median(img0[minx:maxx, maxy:(maxy + l_height)])
	#median_intensity = np.median(img0[:, miny:maxy])

	up_intensity = np.median(img0[(miny - u_height):miny, minx:maxx])
	#low_intensity = np.median(img0[(maxy+1):(maxy + l_height), minx:maxx])
	middle_intensity = np.median(img0[miny:maxy, minx:maxx])
	median_intensity = np.median(img0[:, minx:maxx])

	if Debug:
		print('Upper:%1.2f   --  Lower:%1.2f -- Middle:%1.2f -- Median %1.2f' %(up_intensity, low_intensity, middle_intensity, median_intensity ))
		print('Upper:%1.2f   --  Lower:%1.2f -- Median %1.2f' %(up_intensity, low_intensity, median_intensity ))
		print('Upper/Middle ratio: %1.2f' % (up_intensity/middle_intensity))
		#print('Lower/upper ratio: %1.2f' % (low_intensity/up_intensity))
		#print('Median/upper ratio: %1.2f' % (median_intensity/up_intensity))
		# upper stripe
		cv2.rectangle(imp, (minx, miny - u_height), (maxx, miny), (125, 255, 51), thickness=2)
		# lower stripe
		cv2.rectangle(imp, (minx, miny), (maxx, maxy), (0, 0, 255), thickness=2)
		# middle stripe
		cv2.rectangle(imp, (minx, maxy+1), (maxx, maxy + l_height), (0, 255, 0), thickness=2)
		cv2.imshow('Placa', imp); 
	
	
	
	
	#lixo = img[minx:maxx, maxy:(maxy + l_height), channel];
	#print(lixo)
	
	
	
	#cv2.rectangle(img0, (minx, miny-u_height), (maxx, miny), (125, 255, 51), thickness=2)
	#cv2.rectangle(img0, (minx, maxy), (maxx, maxy+l_height), (125, 255, 51), thickness=2)
	#cv2.rectangle(img0, (0, miny), (240, maxy), (125, 255, 51), thickness=2)
	#cv2.imshow('Placa', img0); 
	#
	#  Lower must be brighter than upper
	#
	
	if up_intensity < 0.6*middle_intensity:
	#if median_intensity > 1.4*up_intensity:
		return 'Mercosul'
	else:
		return 'Brazilian'
	
	
	

