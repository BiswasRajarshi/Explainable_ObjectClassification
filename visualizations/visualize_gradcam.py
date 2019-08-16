import os

import torch
import torch.hub
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, transforms

from utility import utils
from methods import baseoperations
from methods import backpropagation
from methods import guidedbackpropagation
from methods import gradcam


def Grad_CAM(image_paths, target_layer, arch, topk, output_dir, cuda):
    """
    Function for Grad-CAM based Visualizations 
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
    print("Input Images:")
    image_files = os.listdir(image_paths)
    image_filepaths = [os.path.join(image_paths, image_file) for image_file in image_files]
    for i, image_path in enumerate(image_filepaths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = utils.image_preprocessing(image_path)
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)
    
    print("Visualizing Grad-CAM:")
    
    bp = backpropagation.BackPropagation(model=model)
    probs, ids = bp.forward(images)

    gcam = gradcam.GradCAM(model=model)
    _ = gcam.forward(images)

    guided_bp = guidedbackpropagation.GuidedBackPropagation(model=model)
    _ = guided_bp.forward(images)

    for i in range(topk):
        # Guided Backpropagation
        guided_bp.backward(ids=ids[:, [i]])
        gradients = guided_bp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print("\tImage@position: {}; Prediction: {}; (Prediction Probability: {:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # Gradient_Class_Activation_Mapping(GradCAM)
            utils.write_gradcam(
                filename=os.path.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, arch, target_layer, classes[ids[j, i]]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )



