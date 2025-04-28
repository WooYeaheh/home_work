import torch
from anti import *

def model_loader(model_name):
    if model_name == 'MMDG': net = MMDG()
    elif model_name == 'MMDG2': net = MMDG2()
    elif model_name == 'MMDG3': net = MMDG3()
    elif model_name == 'MAViT': net = MAViT()
    elif model_name == 'FMViT': net = FMViT()
    elif model_name == 'ViT': net = ViT()
    elif model_name == 'ViT2': net = ViT2()
    elif model_name == 'ViT3': net = ViT3()
    elif model_name == 'ViT_prompt': net = ViT_prompt()
    elif model_name == 'ViT_prompt_CDC': net = ViT_prompt_CDC()
    elif model_name == 'Diff_attention': net = Diff_attention()
    elif model_name == 'ViT_text': net = ViT_text()
    elif model_name == 'ViT_text2': net = ViT_text2()
    elif model_name == 'ViT_text3': net = ViT_text3()
    elif model_name == 'CLIP_star': net = CLIP_star()
    else:
        raise NameError('UNKNOWN MODEL NAME')
    return net

