## eval.py imports

import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from time import time
import copy
from PIL import Image
import json
import random

from data import *
from config import *
from model import *
from attack import *
from utils import *

## Main eval.py function

## Can pass all epsilons in one go, make this change later, instead of passing all in one go

def eval(data_loader, net, device, criterion, attack, fmodel, optimizer, sample_size, eps_stack, save_length):
  optimizer.zero_grad() # Making grads zero just in case (paranoid)

  ## Initialize variables that accumulate statistics for each batch

  avg_vulnerabilities = dict({eps:0 for eps in eps_stack})
  avg_adv_acc = dict({eps:0 for eps in eps_stack})
  avg_adv_norm_grads = dict({eps:0 for eps in eps_stack})

  adv_images = {}
  advs_success_failures = {}
  logits_advs = {}
  input_grad_norms_stack = []

  total_size = 0

  start_time = time()

  marker_save = [i for i in range(save_length)] ## Getting a sample of 128 images

  for i, data in enumerate(data_loader):
      if i%10 == 0:
          print("number of processed batches = {}, images = {}, time_taken for 10 batches = {}".format(i, total_size, time() - start_time))
          start_time = time()

      images, labels = data
      images = images.to(device)
      labels = labels.to(device)

      total_size+=images.shape[0] ## Increasing total size

      ## Input grad stuff
      net.eval()
      input_grad, input_grad_norms = l2_norm_grads(images, labels, net, criterion, optimizer, return_input_grad=True)
      input_grad_norms_stack.append(input_grad_norms)

      if i in marker_save:
          if i == 0:
              normal_images = images.cpu().numpy()
              logits_images = torch.nn.functional.softmax(net(images), dim = 1).cpu().detach().numpy()
              correct_labels = labels.cpu().numpy()

              ## image gradients
              input_grad_stack = input_grad


          else:
              normal_images = np.vstack((normal_images, images.cpu().numpy()))
              logits_images = np.vstack((logits_images, torch.nn.functional.softmax(net(images), dim = 1).cpu().detach().numpy()))
              correct_labels = np.hstack((correct_labels, labels.cpu().numpy()))

              ## image gradients
              input_grad_stack = np.vstack((input_grad_stack, input_grad))


      # Adversaries stats (can pass all epsilons in one go, make this change later)
      for k in eps_stack:
          if k!=0.0:
              _, [advs], advs_success = attack(fmodel, images, labels, epsilons=[k])
          else:
              advs = images.detach()
              advs_success = 0


          optimizer.zero_grad()

          if i in marker_save:
              if i == 0:
                  ## Images: normal, segmented etc.
                  adv_images[k] = advs.cpu().numpy() # Taking the first batch of adversaries for that particular epsilon essentially

                  ## Labels and logits
                  advs_success_failures[k] = advs_success.cpu().numpy()[0, :]
                  logits_advs[k] = torch.nn.functional.softmax(net(advs), dim = 1).cpu().detach().numpy()
              else:
                  ## Images: normal, segmented etc.
                  adv_images[k] = np.vstack((adv_images[k], advs.cpu().numpy()))

                  ## Labels and logits
                  advs_success_failures[k] = np.hstack((advs_success_failures[k], advs_success.cpu().numpy()[0, :]))
                  logits_advs[k] = np.vstack((logits_advs[k], torch.nn.functional.softmax(net(advs), dim = 1).cpu().detach().numpy()))

          avg_vulnerabilities[k]+=advs_success.sum().item()
          optimizer.zero_grad()

          # Adversarial images stats and advesarial input gradients

          advs.requires_grad = True
          out_advs = net(advs)
          loss_advs = criterion(out_advs, labels)
          reps_advs = out_advs.argmax(1)
          acc_advs = ((labels == reps_advs).sum()).float()
          loss_advs.backward()

          avg_adv_acc[k]+=acc_advs.item()

          advs_grad = advs.grad.detach().clone()
          optimizer.zero_grad()
          advs.required_grad = False
          avg_adv_norm_grads[k]+=advs_grad.pow(2).sum(dim=(1,2,3)).pow(0.5).sum().item()

      # Using this to break, so as to compute statistics only for a small subset of the datapoints
      if sample_size < total_size:
          break

  # Average out adversary stats
  for k in eps_stack:
      avg_vulnerabilities[k] = avg_vulnerabilities[k]/total_size
      avg_adv_acc[k] = avg_adv_acc[k]/total_size
      avg_adv_norm_grads[k] = avg_adv_norm_grads[k]/total_size


  return avg_vulnerabilities, avg_adv_acc, avg_adv_norm_grads, adv_images, normal_images, logits_images, correct_labels, input_grad_norms_stack, input_grad_stack, logits_advs, advs_success_failures

## Main function that puts everything together

def main(args):

    ## Get devices
    device, device_ids = setup_device(args['number_of_gpus'])

    ## Get stat accumulators ready
    adversaries_images, normal_images_dict, accuracies_adversaries, vulnerabilities, \
    adv_norm_grads, logit_images, correct_labels, input_grad_norms_stack, input_grad_stack, \
    logit_advs, advs_success_failures, wts_change_dict, wts_q_norm, wts_k_norm \
      = init_eval_stat_dictionaries(args['eval_attack_keys'], args['eval_data_keys'])
    
    ## Load the dataloader
    if args['dataset'] == 'CIFAR10':
        tr_loader, va_loader, te_loader = dataloader_CIFAR10(args)
    elif args['dataset'] == 'IMAGENET12':
        """ May need to be filled
        """
        tr_loader, va_loader, te_loader = dataloader_IMAGENET12(args)
    
    data_loader_dict = dict({'train': tr_loader, 'test': te_loader, 'val': va_loader})

    ## Load the attacks, and the epsilon stack of the attacks    
    attack_dict = dict({attack_key: return_attack(attack_key, args['pgd_steps']) for attack_key in args['eval_attack_keys']})
    eps_stack_dict = dict({attack_key: args['eps_stack_'+attack_key] for attack_key in args['eval_attack_keys']})

    ## Download base models and pretrained models.

    model = get_model(args)
    pretrained_model = get_model(args).to(device) ## 1000 classes for imagenet

    net = wrap_model(model, args).to(device)
    criterion = nn.CrossEntropyLoss() ## the loss function

    ## Load best or current
    if os.path.isfile(args['directory_net']+"best.pth"):
        checkpoint = torch.load(args['directory_net']+"best.pth")
    else:
        checkpoint = torch.load(args['directory_net']+"current.pth")

    ## actual net loading
    net.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.SGD(net.parameters(), lr=args['learning_rate'], momentum=args['momentum'], weight_decay=args['weight_decay']) ## Optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])

    ## Compute weights change
    state_dict_keys = list(pretrained_model.state_dict().keys())[:-2] ## Getting keys of state dict apart from the classifier
    only_wts = ['weight' in i for i in state_dict_keys]
    state_dict_keys = list(np.array(state_dict_keys)[only_wts])
    for key in state_dict_keys:
        wts_change_dict[key] = torch.norm(pretrained_model.state_dict()[key] - net.model.state_dict()[key]).item()/torch.norm(pretrained_model.state_dict()[key]).item()

    ## Compute norm of attention weights q and k
    if args['model_name'] == 'vit':
        for i in range(len(net.model.transformer.encoder_layers)):
            wts_q_norm[i] = torch.norm(net.model.transformer.encoder_layers[i].attn.query.weight).item()
            wts_k_norm[i] = torch.norm(net.model.transformer.encoder_layers[i].attn.key.weight).item()

    ## Parallelization
    if len(device_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=device_ids)
    net.to(device) ## After data parallel, one can generally put in device (Don't quite get the concept properly yet)
    net.eval()

    ## Evaluate different attack on model on different datasets 

    sample_size = args['eval_sample_size']
    fmodel = fb.PyTorchModel(net, bounds=args['bounds'])
    save_length = args['save_length'] 
    for a_key in args['eval_attack_keys']:
        attack = attack_dict[a_key]
        eps_stack = eps_stack_dict[a_key]
        print(f"Starting evaluation of upsampled {args['model_name']} on adversaries generated from {a_key} attack")
        for key in args['eval_data_keys']:
            d_loader = data_loader_dict[key]

            vulnerabilities[a_key][key], accuracies_adversaries[a_key][key], adv_norm_grads[a_key][key], \
            adversaries_images[a_key][key], normal_images_dict[a_key][key], logit_images[a_key][key], \
            correct_labels[a_key][key], input_grad_norms_stack[a_key][key], input_grad_stack[a_key][key], \
            logit_advs[a_key][key], advs_success_failures[a_key][key] = eval(d_loader, net, device, criterion, attack, \
                                                                                          fmodel, optimizer, sample_size, eps_stack, save_length)

            save_eval_dictionaries(args['directory'], vulnerabilities, accuracies_adversaries, adversaries_images, normal_images_dict, adv_norm_grads, \
                           logit_images, correct_labels, input_grad_norms_stack, input_grad_stack, logit_advs, advs_success_failures, \
                           wts_change_dict, wts_q_norm, wts_k_norm)

    save_eval_dictionaries(args['directory'], vulnerabilities, accuracies_adversaries, adversaries_images, normal_images_dict, adv_norm_grads, \
                           logit_images, correct_labels, input_grad_norms_stack, input_grad_stack, logit_advs, advs_success_failures, \
                           wts_change_dict, wts_q_norm, wts_k_norm)

if __name__ == '__main__':
    args = get_args_eval()
    print(args)
    main(args)
