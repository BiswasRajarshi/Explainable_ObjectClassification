import os

import torch
import torch.hub
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, transforms

from utility import utils
from methods import baseoperations
from methods import backpropagation


def SimpleBackpropagation(image_paths, target_layer, arch, topk, output_dir, cuda):
    """
    Function for Visualizing simple backpropagation 
    for a particular neural model (neural network)
    """
    device = utils.device_signature(cuda)
    # Accessing class labels
    classes = utils.access_classlabels()

    # Retriving chosen model from torchvision model collection 
    model = models.__dict__[arch](pretrained=True)
    model.to(device)
    model.eval()

    # Setting up the list of images
    images = []
    raw_images = []
    
    #This option is for multiple input images images which needs to be figured out 
    print("Input Images:")
    image_files = os.listdir(image_paths)
    image_filepaths = [os.path.join(image_paths, image_file) for image_file in image_files]
    for i, image_path in enumerate(image_filepaths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = utils.image_preprocessing(image_path)
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    
    print("Visualizing Simple Backpropagation:")

    bp = backpropagation.BackPropagation(model=model)
    probs, ids = bp.forward(images)

    for i in range(topk):
        bp.backward(ids=ids[:, [i]])
        gradients = bp.generate()

        for j in range(len(images)):
            print("\tImage@position: {}; Prediction: {}, (Prediction Probability: {:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            utils.write_gradient(
                filename=os.path.join(
                    output_dir,
                    "{}-{}-simplebackprop-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

    bp.remove_hook()

