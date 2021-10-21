"""
Main file for training IWPOD-NET license plate detector. 
In the paper training was performed in a per-batch manner, but in this
file training is performed per-epoch.

In general, you should run at least 100-150K iterations for the paper dataset.
Hence, choose the number of epochs based on the number of iterations per epoch

@author: Claudio Rosito Jung
"""


import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import pandas as pd
import argparse
from tensorflow import keras

from os.path import isfile, isdir, splitext
from os import makedirs

from src.label import readShapes, Shape
from src.loss import iwpodnet_loss
from src.utils import image_files_from_folder
from src.data_generator_tf2 import ALPRDataGenerator
from create_model_iwpodnet import create_model_iwpodnet
from tensorflow.keras.callbacks import LearningRateScheduler


#
#  LR scheduler - can use it to reduce learning rate
#
def lr_scheduler(ChangeEpoch = 20000, initial_lr = 1e-3):
	def scheduler(epoch):
		if epoch < ChangeEpoch:
			return initial_lr
		elif epoch < 2*ChangeEpoch:
			return initial_lr/5
		else:
			return initial_lr/25
	return scheduler


def load_network(modelpath, input_dim):
	#
	#  Creates model topology
	#
	model = create_model_iwpodnet()
	
	#
	#  Loads weights -- if they exist
	#
	if isfile(modelpath + '.h5'):
		model.load_weights(modelpath + '.h5')
		print('Loaded weights')
	else:
		print('Training from scratch')
	input_shape = (input_dim,input_dim,3)

	# Fixed input size for training
	inputs  = keras.layers.Input(shape=(input_dim,input_dim, 3))
	
	#
	#  Gets size of output layer
	# 
	outputs = model(inputs)
	if tf.__version__.startswith('1'):
		output_shape = tuple([s.value for s in outputs.shape[1:]])
	else:
		output_shape = tuple([s for s in outputs.shape[1:]])

	output_dim   = output_shape[1]
	model_stride = input_dim / output_dim

	assert input_dim % output_dim == 0, \
		'The output resolution must be divisible by the input resolution'

	assert model_stride == 2**4, \
		'Make sure your model generates a feature map with resolution ' \
		'16x smaller than the input'

	return model, model_stride, input_shape, output_shape

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-md' 		,'--model-dir'			,type=str   , default = 'weights'			,help='Directory containing models and weights')
	parser.add_argument('-cm' 		,'--cur_model'			,type=str   , default = 'fake_name'			,help='Pre-trained model')
	parser.add_argument('-n' 		,'--name'				,type=str   , default = 'iwpodnet_retrained'	,help='Output model name')
	parser.add_argument('-tr'		,'--train-dir'			,type=str   , default = 'train_dir'			,help='Input data directory for training')
	parser.add_argument('-e'		,'--epochs'				,type=int   , default = 60000					,help='Number of epochs (default = 1.500)')
	parser.add_argument('-bs'		,'--batch-size'			,type=int   , default = 52						,help='Mini-batch size (default = 64)')
	parser.add_argument('-lr'		,'--learning-rate'		,type=float , default = 0.001					,help='Learning rate (default = 0.001)')
	parser.add_argument('-se'		,'--save-epochs'		,type = int , default = 2000					,help='Freqnecy for saving checkpoints (in epochs) ')
	args = parser.parse_args()
	

	#
	#  Training parameters
	#
	
	MaxEpochs = args.epochs
	batch_size 	= args.batch_size
	learning_rate = args.learning_rate
	save_epochs = args.save_epochs			


	netname = args.name
	train_dir = args.train_dir
	modeldir = args.model_dir
	train_dir = args.train_dir 
	
	modelname = '%s/%s'  % (modeldir, args.cur_model)
	
	#
	#  Additional parameters
	#
	dim = 208 # spatial dimension of images in training stage
	opt = Adam(learning_rate = learning_rate) # Optimizer -- can change

	
	if not isdir(modeldir):
		makedirs(modeldir)
	 
	#
	#   Loads model with pre-trained weights - if present
	#
	model, model_stride, xshape, yshape = load_network(modelname, dim)
	model_path_final  = '%s/%s'  % (modeldir, netname)

	#
	#  Loads training data
	#
	print ('Loading training data...')
	Files = image_files_from_folder(train_dir)
	
	#
	#  Defines size of "fake" tiny LP annotation, used when no LP is present
	#				
	fakepts = np.array([[0.5, 0.5001, 0.5001, 0.5], [0.5, 0.5, 0.5001, 0.5001]])
	fakeshape = Shape(fakepts)
	Data = []
	ann_files = 0
	for file in Files:
		labfile = splitext(file)[0] + '.txt'
		if isfile(labfile):
			ann_files += 1
			L = readShapes(labfile)
			I = cv2.imread(file)
			if len(L) > 0:
				Data.append([I, L])
		else:
			#
			#  Appends a "fake"  plate to images without any annotation
			#
			I = cv2.imread(file)
			Data.append(  [I, [fakeshape] ]  )

	print ('%d images with labels found' % len(Data) )
	print ('%d annotation files found' % ann_files )

	#
	#  Training generator with lots of data augmentation	
	#
	train_generator = ALPRDataGenerator(Data, batch_size = batch_size, dim =  dim, stride = int(model_stride), shuffle=True, OutputScale = 1.0)

	#
	#  Compiles Model
	#
	model.compile(
	    loss = iwpodnet_loss,
	    optimizer = opt,
	    )
  
	#
	#  Callbacks
	#  
	
	# -> Model Chekcpoints --  save evey "save_epochs" epochs
	ckpt = ModelCheckpoint(
		filepath = model_path_final + '_epoch{epoch:03d}.h5',
		save_freq= int( np.floor(len(Data)/batch_size)*save_epochs)  # defines frequency of checkpoints
	)

	# -> Learning rate control -- can also reduce learning rate dynamically		
	learn_control = LearningRateScheduler(lr_scheduler(ChangeEpoch = MaxEpochs//3, initial_lr =  learning_rate))
		  
	      
	# -> early stopping criteria -- not currently used		
	es = EarlyStopping(monitor='loss',
	                     patience = MaxEpochs//30,
	                     restore_best_weights = False)
	
	
	#
	#  Trains model 
	#
	print('Starting to train the model')
	history = model.fit(x = train_generator,
	                      steps_per_epoch = np.floor(len(Data)/batch_size),
	                      epochs = MaxEpochs, 
	                      verbose = 1,
	                      callbacks=[learn_control, ckpt])  


	print('Finished to train the model')
	
	#
	#  Saves training details to excel file
	#
	df = pd.DataFrame(history.history)
	df.to_excel(model_path_final + '.xlsx')
	
	#
	#  Saves trained weights and model (TF2) format
	#
	print ('Saving model (%s)' % model_path_final)
	model.save_weights(model_path_final + '.h5', save_format  ='h5')
	model.save(model_path_final)
