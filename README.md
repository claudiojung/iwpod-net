# ALPR Using an Improved Warped Planar Object Detection Network (IWPOD-NET)

This repository contains the author's implementation of the paper "A Flexible Approach for Automatic License Plate Recognition in Unconstrained Scenarios", published in the journal IEEE Transactions on Intelligent Transportation Systems (to appear).

We provide the IWPOD-NET newtwork model and the pre-trained weights described in the paper, as well as a few examples of input images and simple code to run the detection module and license plate rectification. The input to IWPOD-NET is typically an image containing a vehicle crop (output of a vehicle detector) and the vehicle type (car, bus, truck, bike). If the input image already contains the vehicles roughly framed, you can feed the input image directly, choosing as vehicle type "fullimage".

## Running a simple test

The basic usage is

python example_plate_detection.py --image [image name] --vtype [vehicle type] --lp_threshold [detection threshold]

You can run a simple test based on the provided images.

```
python example_plate_detection.py --image images\example_aolp_fullimage.jpg --vtype fullimage
python example_plate_detection.py --image images\example_bike.jpg --vtype bike
```
