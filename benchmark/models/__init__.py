import torch
import numpy as np
from PIL import Image



def skip(*args, **kwargs):
    pass
torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip


def get_image(image):
    if type(image) is str:
        try:
            return Image.open(image).convert("RGB")
        except Exception as e:
            print(f"Fail to read image: {image}")
            exit(-1)
    elif type(image) is Image.Image:
        return image
    else:
        raise NotImplementedError(f"Invalid type of Image: {type(image)}")


def get_BGR_image(image):
    image = get_image(image)
    image = np.array(image)[:, :, ::-1]
    image = Image.fromarray(np.uint8(image))
    return image


def get_model(args,device):
    if args.model_name == 'ClipPythia':
        from benchmark.models.test_ClipPythia import TestClipPythia
        return TestClipPythia(args,device)
    elif args.model_name == 'Pythia':
        from benchmark.models.test_Pythia import TestPythia
        return TestPythia(args,device)
    else:
        raise ValueError(f"Invalid model_name: {args.model_name}")
