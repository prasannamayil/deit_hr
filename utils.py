## utils.py Imports
import torch
import numpy as np
import pickle
import re
import math
import torch
import collections

from torch import nn
from functools import partial
from torch.utils import model_zoo
from torch.nn import functional as F

from resnet import resnet50

##<==============Utlilty funcs for ViT copy pasted From Lukelamas/tchzhangzi=======================>##

################################################################################
### Help functions for model architecture
################################################################################

# Params: namedtuple
# get_width_and_height_from_size and calculate_output_image_size

# Parameters for the entire model (stem, all blocks, and head)
Params = collections.namedtuple('Params', [
    'image_size', 'patch_size', 'emb_dim', 'mlp_dim', 'num_heads', 'num_layers',
    'num_classes', 'attn_dropout_rate', 'dropout_rate', 'resnet'
])

# Set Params and BlockArgs's defaults
Params.__new__.__defaults__ = (None, ) * len(Params._fields)


def get_width_and_height_from_size(x):
    """Obtain height and width from x.
    Args:
        x (int, tuple or list): Data size.
    Returns:
        size: A tuple or list (H,W).
    """
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()


################################################################################
### Helper functions for loading model params
################################################################################

# get_model_params and efficientnet:
#     Functions to get BlockArgs and GlobalParams for efficientnet
# url_map and url_map_advprop: Dicts of url_map for pretrained weights
# load_pretrained_weights: A function to load pretrained weights


def vision_transformer(model_name):
    """Create Params for vision transformer model.
    Args:
        model_name (str): Model name to be queried.
    Returns:
        Params(params_dict[model_name])
    """

    params_dict = {
        'ViT-B_16': (384, 16, 768, 3072, 12, 12, 1000, 0.0, 0.1, None),
        'ViT-B_32': (384, 32, 768, 3072, 12, 12, 1000, 0.0, 0.1, None),
        'ViT-L_16': (384, 16, 1024, 4096, 16, 24, 1000, 0.0, 0.1, None),
        'ViT-L_32': (384, 32, 1024, 4096, 16, 24, 1000, 0.0, 0.1, None),
        'R50+ViT-B_16': (384, 1, 768, 3072, 12, 12, 1000, 0.0, 0.1, resnet50),
    }
    image_size, patch_size, emb_dim, mlp_dim, num_heads, num_layers, num_classes, attn_dropout_rate, dropout_rate, resnet = params_dict[
        model_name]
    params = Params(image_size=image_size,
                    patch_size=patch_size,
                    emb_dim=emb_dim,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    num_classes=num_classes,
                    attn_dropout_rate=attn_dropout_rate,
                    dropout_rate=dropout_rate,
                    resnet=resnet)

    return params


def get_model_params(model_name, override_params):
    """Get the block args and global params for a given model name.
    Args:
        model_name (str): Model's name.
        override_params (dict): A dict to modify params.
    Returns:
        params
    """
    params = vision_transformer(model_name)

#     if override_params:
#         # ValueError will be raised here if override_params has fields not included in params.
#         params = params._replace(**override_params)
    return params


# train with Standard methods
# check more details in paper(An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale)
url_map = {
    'ViT-B_16':
    'https://github.com/tczhangzhi/VisionTransformer-PyTorch/releases/download/1.0.1/ViT-B_16_imagenet21k_imagenet2012.pth',
    'ViT-B_32':
    'https://github.com/tczhangzhi/VisionTransformer-PyTorch/releases/download/1.0.1/ViT-B_32_imagenet21k_imagenet2012.pth',
    'ViT-L_16':
    'https://github.com/tczhangzhi/VisionTransformer-PyTorch/releases/download/1.0.1/ViT-L_16_imagenet21k_imagenet2012.pth',
    'ViT-L_32':
    'https://github.com/tczhangzhi/VisionTransformer-PyTorch/releases/download/1.0.1/ViT-L_32_imagenet21k_imagenet2012.pth',
    'R50+ViT-B_16':
    'https://github.com/tczhangzhi/VisionTransformer-PyTorch/releases/download/1.0.1/R50+ViT-B_16_imagenet21k_imagenet2012.pth',
}


def load_pretrained_weights(model,
                            model_name,
                            weights_path=None,
                            load_fc=True,
                            advprop=False):
    """Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): The whole model of vision transformer.
        model_name (str): Model name of vision transformer.
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
    """
    if isinstance(weights_path, str):
        state_dict = torch.load(weights_path)
    else:
        state_dict = model_zoo.load_url(url_map[model_name])

    if load_fc:
        ret = model.load_state_dict(state_dict, strict=False)
#         assert not ret.missing_keys, 'Missing keys when loading pretrained weights: {}'.format(
#             ret.missing_keys)
    else:
        state_dict.pop('classifier.weight')
        state_dict.pop('classifier.bias')
        ret = model.load_state_dict(state_dict, strict=False)
#         assert set(ret.missing_keys) == set([
#             'classifier.weight', 'classifier.bias'
#         ]), 'Missing keys when loading pretrained weights: {}'.format(
#             ret.missing_keys)
#     assert not ret.unexpected_keys, 'Missing keys when loading pretrained weights: {}'.format(
#         ret.unexpected_keys)
    print('Missing keys when loading pretrained weights: {}'.format(ret.missing_keys))
    print('Loaded pretrained weights for {}'.format(model_name))


##===========Edit pos embedding by (Mine) ===============>##
def edit_pos_embedding(model, img_size=384, patch_size=16, embedding_size=768, timm=False, interpolate=False):
    num_scales = model.num_scales

    if num_scales < 2:
        print("Error: pos embedding no need edits")
    else:
        num_patches_row = int(img_size / patch_size)

        ## get the position embedding, and break cls token and rest

        if timm:
            pos_embedding = model.pos_embed.data.detach().clone()
            pos_embedding_copy = model.pos_embed.data.detach().clone()
        else:
            pos_embedding = model.transformer.pos_embedding.pos_embedding.data.detach().clone()
            pos_embedding_copy = model.transformer.pos_embedding.pos_embedding.data.detach().clone()

        cls_token = pos_embedding[:, 0, :].reshape(1, 1, embedding_size)
        pos_embedding = pos_embedding[:, 1:, :]
        pos_embedding_copy = pos_embedding_copy[:, 1:, :]

        ## Average pooling kernel

        for scale in range(num_scales - 1):

            kernel_size = 2 ** (scale + 1)
            transform = torch.nn.AvgPool2d(kernel_size)
            if interpolate:
                posemb_grid = pos_embedding_copy.reshape(1, num_patches_row, num_patches_row, embedding_size).permute(0, 3, 1, 2)
                num_patches_row_new = int(num_patches_row / (kernel_size))
                posemb_grid = F.interpolate(posemb_grid, size=(num_patches_row_new, num_patches_row_new),mode='bilinear')
                new_scale_pos_embedding = posemb_grid.permute(0, 2, 3, 1)
            else:
                new_scale_pos_embedding = transform(pos_embedding_copy.reshape(1, num_patches_row, num_patches_row, embedding_size).permute(0, 3, 1, 2)).permute(0,2,3,1)

            b, h, w, c = new_scale_pos_embedding.shape
            new_scale_pos_embedding = new_scale_pos_embedding.reshape(b, h * w, c)

            ## Concat it to position embedding
            pos_embedding = torch.cat([pos_embedding, new_scale_pos_embedding], dim=1)

        ## Concatenate the position embeddings and put it back to the model
        new_pos_embedding = torch.cat([cls_token, pos_embedding], dim=1)

        ## if timm or tchzhangzhi
        if timm:
            model.pos_embed.data = new_pos_embedding.detach().clone()
        else:
            model.transformer.pos_embedding.pos_embedding.data = new_pos_embedding.detach().clone()

##<==============Other utility function (Mine)=======================>##

def setup_device(required_gpus):
    actual_gpus = torch.cuda.device_count()
    if required_gpus > 0 and actual_gpus == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        required_gpus = 0
    if required_gpus > actual_gpus:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(required_gpus, actual_gpus))
        required_gpus = actual_gpus
    device = torch.device('cuda:0' if required_gpus > 0 else 'cpu')
    list_ids = list(range(required_gpus))
    return device, list_ids

## Saves entire state of training with state_dict, optimizer, scheduler etc.

def save_model(save_dir, epoch, model, optimizer, lr_scheduler, device_ids, best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    filename = str(save_dir + 'current.pth')
    torch.save(state, filename)

    if best:
        filename = str(save_dir + 'best.pth')
        torch.save(state, filename)

## Function that computes the expected gradient norm of the input for a batch. 

def l2_norm_grads(images, labels, net, criterion, optimizer, return_input_grad=False):
    images.requires_grad = True # Setting true so that gradients can be backproped to input
    temp_loss = criterion(net(images), labels)
    temp_loss.backward()

    input_grad = images.grad

    optimizer.zero_grad() # Make grads zero
    images.requires_grad = False
    
    expected_norm_grads = input_grad.pow(2).sum(dim=(1,2,3)).pow(0.5).mean().item()
    if return_input_grad:
        return input_grad.cpu().detach().clone().numpy(), expected_norm_grads
    else:
        return expected_norm_grads

## The dictionary of all losses, accuracies, gradient norms for natural images and corresponding adversaries.

def init_stat_dictionaries(args):
    if not args['resume_training']:
        losses_adversaries = {
                  'tr_epoch': [],
                  'va_epoch': [],
                  'te_epoch': []
        }


        vulnerabilities = {
                          'tr_epoch': [],
                          'va_epoch': [],
                          'te_epoch': []
        }

        accuracies_adversaries = {
                      'tr_epoch': [],
                      'va_epoch': [],
                      'te_epoch': []
        }

        grad_norms_adversaries = {
                      'tr_epoch': [],
                      'va_epoch': [],
                      'te_epoch': []
        }

        grad_norms_input = {
                      'tr_epoch': [],
                      'va_epoch': [],
                      'te_epoch': []
        }
    else:
          with open(args['directory']+'losses_adversaries.pickle', 'rb') as handle:
                losses_adversaries = pickle.load(handle)
          
          with open(args['directory']+'vulnerabilities.pickle', 'rb') as handle:
                vulnerabilities = pickle.load(handle)
          
          with open(args['directory']+'accuracies_adversaries.pickle', 'rb') as handle:
                accuracies_adversaries = pickle.load(handle)
          
          with open(args['directory']+'grad_norms_adversaries.pickle', 'rb') as handle:
                grad_norms_adversaries = pickle.load(handle)

          with open(args['directory']+'grad_norms_input.pickle', 'rb') as handle:
                grad_norms_input = pickle.load(handle)

    return losses_adversaries, vulnerabilities, accuracies_adversaries, \
            grad_norms_adversaries, grad_norms_input


## Function to save all trains statistics

def save_dictionaries(directory, vulnerabilities, losses_adversaries, accuracies_adversaries, grad_norms_input, grad_norms_adversaries):
  with open(directory+'vulnerabilities.pickle', 'wb') as handle:
      pickle.dump(vulnerabilities, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'losses_adversaries.pickle', 'wb') as handle:
      pickle.dump(losses_adversaries, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'accuracies_adversaries.pickle', 'wb') as handle:
      pickle.dump(accuracies_adversaries, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'grad_norms_input.pickle', 'wb') as handle:
      pickle.dump(grad_norms_input, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'grad_norms_adversaries.pickle', 'wb') as handle:
      pickle.dump(grad_norms_adversaries, handle, protocol=pickle.HIGHEST_PROTOCOL)


### Eval utils functions ###########

def init_eval_stat_dictionaries(eval_attack_keys, eval_data_keys):
    ## Getting the stat dicts ready
    adversaries_images = dict({i:{j: None for j in eval_data_keys} for i in eval_attack_keys})
    normal_images_dict = dict({i:{j: None for j in eval_data_keys} for i in eval_attack_keys})

    accuracies_adversaries = dict({i:{j: {} for j in eval_data_keys} for i in eval_attack_keys})
    vulnerabilities = dict({i:{j: {} for j in eval_data_keys} for i in eval_attack_keys})

    adv_norm_grads = dict({i:{j: {} for j in eval_data_keys} for i in eval_attack_keys})
    logit_images = dict({i:{j: {} for j in eval_data_keys} for i in eval_attack_keys})
    correct_labels = dict({i:{j: {} for j in eval_data_keys} for i in eval_attack_keys})
    input_grad_norms_stack = dict({i:{j: {} for j in eval_data_keys} for i in eval_attack_keys})
    input_grad_stack = dict({i:{j: {} for j in eval_data_keys} for i in eval_attack_keys})
    logit_advs = dict({i:{j: {} for j in eval_data_keys} for i in eval_attack_keys})
    advs_success_failures = dict({i:{j: {} for j in eval_data_keys} for i in eval_attack_keys})
    
    ## Weights change dict
    wts_change_dict = dict()
    wts_q_norm = dict()
    wts_k_norm = dict()

    return adversaries_images, normal_images_dict, accuracies_adversaries, vulnerabilities, \
      adv_norm_grads, logit_images, correct_labels, input_grad_norms_stack, input_grad_stack, \
      logit_advs, advs_success_failures, wts_change_dict, wts_q_norm, wts_k_norm 


## Function to save all eval statistics

def save_eval_dictionaries(directory, vulnerabilities, accuracies_adversaries, adversaries_images, normal_images_dict, adv_norm_grads, \
                           logit_images, correct_labels, input_grad_norms_stack, input_grad_stack, logit_advs, advs_success_failures, \
                           wts_change_dict, wts_q_norm, wts_k_norm):
  with open(directory+'vulnerabilities.pickle', 'wb') as handle:
      pickle.dump(vulnerabilities, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'accuracies_adversaries.pickle', 'wb') as handle:
      pickle.dump(accuracies_adversaries, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'adversaries_images.pickle', 'wb') as handle:
      pickle.dump(adversaries_images, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'normal_images.pickle', 'wb') as handle:
      pickle.dump(normal_images_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'adv_norm_grads.pickle', 'wb') as handle:
      pickle.dump(adv_norm_grads, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'logit_images.pickle', 'wb') as handle:
      pickle.dump(logit_images, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'correct_labels.pickle', 'wb') as handle:
      pickle.dump(correct_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'input_grad_norms_stack.pickle', 'wb') as handle:
      pickle.dump(input_grad_norms_stack, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'input_grad_stack.pickle', 'wb') as handle:
      pickle.dump(input_grad_stack, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'logit_advs.pickle', 'wb') as handle:
      pickle.dump(logit_advs, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'advs_success_failures.pickle', 'wb') as handle:
    pickle.dump(advs_success_failures, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'wts_change_dict.pickle', 'wb') as handle:
    pickle.dump(wts_change_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'wts_q_norm.pickle', 'wb') as handle:
    pickle.dump(wts_q_norm, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'wts_k_norm.pickle', 'wb') as handle:
    pickle.dump(wts_k_norm, handle, protocol=pickle.HIGHEST_PROTOCOL)

