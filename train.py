## train.py imports

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
import pickle

from data import *
from config import *
from model import *
from attack import *
from utils import *

## Train one epoch

def one_epoch(data_loader, net, device, criterion, optimizer, scheduler, fmodel, attack, is_train, eps, sample_size):

    ## Initialize variables that accumulate stats
    avg_vulnerabilities = 0
    avg_adv_accuracy = 0
    avg_adv_loss = 0
    avg_adv_grad_norm = 0
    avg_input_grad_norm = 0
    total_images = 0

    start_time_100 = time()

    for i, data in enumerate(data_loader):

        ## Break out of the function if we exceed the given sample_size and it is not train
        if is_train is not True and total_images >= sample_size:
            break

        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        dtrain = time()

        ## Finding adversaries for that particular epsilon 
        net.eval()
        _, [advs], temp_success = attack(fmodel, images, labels, epsilons=[eps])
        optimizer.zero_grad()

        ## Change it to train mode if we want to train
        if is_train:
            net.train()

        out_advs = net(advs)
        loss_advs = criterion(out_advs, labels)
        reps_advs = out_advs.argmax(1)
        acc_advs = ((labels == reps_advs).sum()).float()

        ## Training model on that particular batch of advs if it is train
        if is_train:
            loss_advs.backward()
            optimizer.step()
            optimizer.zero_grad()


        ## Losses, Accuracies, and Grad norms of adversarial images at each batch
        avg_vulnerabilities+=temp_success.sum().item() ## total vulnerable images
        avg_adv_loss+=loss_advs.item()*len(images)
        avg_adv_accuracy+=acc_advs.item()

        net.eval() ## For calculating input gradients norms
        grad_norm_batch = l2_norm_grads(images, labels, net, criterion, optimizer)
        avg_input_grad_norm+=grad_norm_batch*len(images) 

        net.eval() ## For calculating advs gradients norms
        adv_grad_norm_batch = l2_norm_grads(advs, labels, net, criterion, optimizer)
        avg_adv_grad_norm+=adv_grad_norm_batch*len(images) 
        
        total_images+=len(images)
        if (i+1)%100 == 0:
            print("i = {} Accuracy_Advs = {} Loss_Advs = {} Grad_norm_natural {}".format(i+1, acc_advs.item(), loss_advs.item(), adv_grad_norm_batch))
            print(f"time taken for 100 batches: {time()-start_time_100}")
            start_time_100 = time()
    

    return avg_vulnerabilities/total_images, avg_adv_loss/total_images, avg_adv_accuracy/total_images, avg_input_grad_norm/total_images, avg_adv_grad_norm/total_images


def main(args):
    ## Get the data loaders
    if args['dataset'] == 'CIFAR10':
        tr_loader, va_loader, te_loader = dataloader_CIFAR10(args)
    elif args['dataset'] == 'IMAGENET12':
        """ May need to be filled
        """
        tr_loader, va_loader, te_loader = dataloader_IMAGENET12(args)
        


    ## Get stat accumulators ready
    losses_adversaries, vulnerabilities, accuracies_adversaries, \
    grad_norms_adversaries, grad_norms_input \
     = init_stat_dictionaries(args)
    
    ## Get devices ready
    device, device_ids = setup_device(args['number_of_gpus'])


    ## Get pretrained model and wrap it
    model = get_model(args)
    net = wrap_model(model, args)

    ## Get criterion, optimizer and scheduler
    criterion = nn.CrossEntropyLoss() ## the loss function
    optimizer = optim.SGD(net.parameters(), lr=args['learning_rate'], momentum=args['momentum'], weight_decay=args['weight_decay']) ## Optimizer

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer=optimizer,
                    max_lr=args['learning_rate'],
                    pct_start=args['warmup_steps']/args['train_steps'],
                    total_steps=args['train_steps'])

    ## To resume training
    if args['resume_training'] == True:
        ## Load best or current
        if os.path.isfile(args['directory_net']+"best.pth"):
            checkpoint = torch.load(args['directory_net']+"best.pth")
        else:
            checkpoint = torch.load(args['directory_net']+"current.pth")
        
        ## Load the state
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args['resume_epoch'] = checkpoint['epoch']

   

    ## Parallelization of the network
    if len(device_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=device_ids)

    net = net.to(device)


    
    ## Get attack and convert network to foolbox model
    attack = return_attack(args['attack_key'], args['pgd_steps'])

    net.eval()
    fmodel = fb.PyTorchModel(net, bounds=args['bounds'])

    print(f"Starting training of upsampled {args['model_name']} on adversaries generated from {args['attack_key']} attack with epsilon = {args['epsilon']}")

    start_time_epoch = time()
    for e in range(args["num_epochs"]-args['resume_epoch']+1):

        # Train model on train set for an epoch
        avg_vulnerabilities, avg_adv_loss, avg_adv_accuracy, avg_input_grad_norm, avg_adv_grad_norm \
            = one_epoch(tr_loader, net, device, criterion, optimizer, scheduler, fmodel, attack, True, args['epsilon'], args['sample_size'])

        vulnerabilities['tr_epoch'].append(avg_vulnerabilities)
        losses_adversaries['tr_epoch'].append(avg_adv_loss)
        accuracies_adversaries['tr_epoch'].append(avg_adv_accuracy)
        grad_norms_input['tr_epoch'].append(avg_input_grad_norm)
        grad_norms_adversaries['tr_epoch'].append(avg_adv_grad_norm)

        # Step learning rate
        if e!=0:
            scheduler.step()

        # Validation stats for a small sample size
        avg_vulnerabilities, avg_adv_loss, avg_adv_accuracy, avg_input_grad_norm, avg_adv_grad_norm \
            = one_epoch(va_loader, net, device, criterion, optimizer, scheduler, fmodel, attack, False, args['epsilon'], args['sample_size'])

        vulnerabilities['va_epoch'].append(avg_vulnerabilities)
        losses_adversaries['va_epoch'].append(avg_adv_loss)
        accuracies_adversaries['va_epoch'].append(avg_adv_accuracy)
        grad_norms_input['va_epoch'].append(avg_input_grad_norm)
        grad_norms_adversaries['va_epoch'].append(avg_adv_grad_norm)


        # Test stats for a small sample size
        avg_vulnerabilities, avg_adv_loss, avg_adv_accuracy, avg_input_grad_norm, avg_adv_grad_norm \
            = one_epoch(te_loader, net, device, criterion, optimizer, scheduler, fmodel, attack, False, args['epsilon'], args['sample_size'])

        vulnerabilities['te_epoch'].append(avg_vulnerabilities)
        losses_adversaries['te_epoch'].append(avg_adv_loss)
        accuracies_adversaries['te_epoch'].append(avg_adv_accuracy)
        grad_norms_input['te_epoch'].append(avg_input_grad_norm)
        grad_norms_adversaries['te_epoch'].append(avg_adv_grad_norm)


        ## Saving the best and current states and stats dictionaries

        ## Saving the current state
        save_model(args['directory_net'], e, net, optimizer, scheduler, device_ids)

        ### Saving some statistics at least val_loss
        if e==0:
            va_loss_best = copy.deepcopy(losses_adversaries['va_epoch'][-1])
        else:
            if losses_adversaries['va_epoch'][-1] <= va_loss_best:
                va_loss_best = copy.deepcopy(losses_adversaries['va_epoch'][-1])

                ##Saving the best state
                save_model(args['directory_net'], e, net, optimizer, scheduler, device_ids, True)

        save_dictionaries(args['directory'], vulnerabilities, losses_adversaries, accuracies_adversaries, grad_norms_input, grad_norms_adversaries)

        ### Printing stats after an epoch
        print(f"epoch = {e+1}, train adv acc = {accuracies_adversaries['tr_epoch'][-1]}, test adv acc = {accuracies_adversaries['te_epoch'][-1]}, train loss = {losses_adversaries['tr_epoch'][-1]}, test loss = {losses_adversaries['te_epoch'][-1]}")
        print(f"time taken for epoch {e+1} is: {time()-start_time_epoch}")
        start_time_epoch = time()

if __name__ == '__main__':
    args = get_args_train()
    print(args)
    main(args)
