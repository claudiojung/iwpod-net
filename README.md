# IWPOD-NET
Repository for the paper "A Flexible Approach for Automatic License Plate Recognition in Unconstrained Scenarios"

This repository contains the IWPOD-NET newtwork model and the pre-trained weights described in the paper "A Flexible Approach for Automatic License Plate Recognition in Unconstrained Scenario - IEEE Transactions on Intelligent Transportation Systems", as well as a simple code to run the detection module and license plate rectification. The input to IWPOD-NET is typically an image containing a vehicle crop (output of a vehicle detector) and the vehicle type (car, bus, truck, bike). If the input image already contains the vehicles roughly framed, you can feed the input image directly, choosing as vehicle type "fullimage".

Some examples based on (the few) provided images:

python example_plate_detection.py --image images\example_aolp_fullimage.jpg --vtype fullimage

python example_plate_detection.py --image images\example_bike.jpg --vtype bike




