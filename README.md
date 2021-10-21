# ALPR Using an Improved Warped Planar Object Detection Network (IWPOD-NET)

This repository contains the author's implementation of the paper "A Flexible Approach for Automatic License Plate Recognition in Unconstrained Scenarios", published in the journal IEEE Transactions on Intelligent Transportation Systems. If you use this repository, you must cite the paper:

```
@article{silva2021flexible,
  title={A Flexible Approach for Automatic License Plate Recognition in Unconstrained Scenarios},
  author={Silva, Sergio M and Jung, Cl{\'a}udio Rosito},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2021},
  publisher={IEEE}
}
```

We provide the IWPOD-NET newtwork model and the pre-trained weights described in the paper, as well as a few examples of input images and simple code to run the detection module and license plate rectification. The input to IWPOD-NET is typically an image containing a vehicle crop (output of a vehicle detector) and the vehicle type (car, bus, truck, bike). If the input image already contains the vehicles roughly framed, you can feed the input image directly, choosing as vehicle type "fullimage".

## Running a simple test

The basic usage is

```
python example_plate_detection.py --image [image name] --vtype [vehicle type] --lp_threshold [detection threshold]
```

You can run a simple test based on the provided images.

```
python example_plate_detection.py --image images\example_aolp_fullimage.jpg --vtype fullimage
python example_plate_detection.py --image images\example_bike.jpg --vtype bike
```

## Training a model 

You can also train your model from scratch or fine-tune a pre-trained model. In the paper we used a per-batch training strategy (in TF1), but this repository provides a per-epoch training strategy (in TF2). The main function for training a model is

```
python train_iwpodnet_tf2.py [-h] [-md MODEL_DIR] [-cm CUR_MODEL] [-n NAME] [-tr TRAIN_DIR] [-e EPOCHS] [-bs BATCH_SIZE] [-lr LEARNING_RATE] [-se SAVE_EPOCHS]
```

You can train a model through:

```
python train_iwpodnet_tf2.py -md weights -n my_trained_iwpodnet -tr train_dir -e 20000 -bs 64 -lr 0.001 -se 5000
```
to train a model from scracth for 20.000 epochs, batch-size 64 and initial learning rate of 1e-3, saving intermediate checkpoints every 5.000 epochs.

We provide a few annotated samples in the directory *train_dir*, all of them extracted from the CCPD dataset (https://github.com/detectRecog/CCPD). The annotation files contain the (relative) locations of the four LP corners -- you can find an annotation tool in the repo of our previous ECCV paper (https://github.com/sergiomsilva/alpr-unconstrained). In the folder *bgimages* you can add images without LPs, which are used in the data augmentation procedure to reduce the number of false positives. The folder *weights* is the default directory for loading/storing models and weights.

**Note:** training a model using *only* the provided images might produce an OK model. The results shown in the paper were obtained using annotated images containing LPs from several datasets/regions, which yields a single model able to detect generic LPs. If you want to train a detector for a *specific region*, maybe using only plates for that region can be adequate.
