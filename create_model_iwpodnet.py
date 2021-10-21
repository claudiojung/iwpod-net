from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Add, Activation, Concatenate, Input
from tensorflow.keras.models import Model


def res_block(x,sz,filter_sz=3,in_conv_size=1):
	xi  = x
	for i in range(in_conv_size):
		xi  = Conv2D(sz, filter_sz, activation='linear', padding='same')(xi)
		xi  = BatchNormalization()(xi)
		xi 	= Activation('relu')(xi)
	xi  = Conv2D(sz, filter_sz, activation='linear', padding='same')(xi)
	xi  = BatchNormalization()(xi)
	xi 	= Add()([xi,x])
	xi 	= Activation('relu')(xi)
	return xi

def conv_batch(_input,fsz,csz,activation='relu',padding='same',strides=(1,1)):
	output = Conv2D(fsz, csz, activation='linear', padding=padding, strides=strides)(_input)
	output = BatchNormalization()(output)
	output = Activation(activation)(output)
	return output


def end_block_iwpodnet(x):
	xprobs    = conv_batch(x, 64, 3, activation='relu')
	xprobs    = conv_batch(xprobs, 32, 3, activation='linear')
	xprobs    = Conv2D(1, 3, activation='sigmoid', padding='same',  kernel_initializer = 'he_uniform')(xprobs)
	xbbox    = conv_batch(x, 64, 3, activation='relu')
	xbbox    = conv_batch(xbbox, 32, 3, activation='linear')
	xbbox     = Conv2D(6, 3, activation='linear' , padding='same',  kernel_initializer = 'he_uniform')(xbbox)
	return Concatenate(3)([xprobs,xbbox])

def create_model_iwpodnet():
	#
	#  Creates additonal layers to discriminate the tasks of detection and
	#  localization. Can freeze common layers and train specialized layers
	#  separately
	#
	
	input_layer = Input(shape=(None,None,3),name='input')

	x = conv_batch(input_layer, 16, 3)
	x = conv_batch(x, 16, 3)
	x = MaxPooling2D(pool_size=(2,2))(x)
	x = conv_batch(x, 32, 3)
	x = res_block(x, 32)
	x = MaxPooling2D(pool_size=(2,2))(x)
	x = conv_batch(x, 64, 3)
	x = res_block(x,64)
	x = res_block(x,64)
	x = MaxPooling2D(pool_size=(2,2))(x)
	x = conv_batch(x, 64, 3)
	x = res_block(x,64)
	x = res_block(x,64)
	x = MaxPooling2D(pool_size=(2,2))(x)
	x = conv_batch(x, 128, 3)
	x = res_block(x,128)
	x = res_block(x,128)
	x = res_block(x,128)
	x = res_block(x,128)
	x = end_block_iwpodnet(x)

	return Model(inputs=input_layer,outputs=x)


if __name__ == '__main__':

	model = create_model_iwpodnet()
	print ('Finished')

