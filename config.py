## config.py imports
import sys
import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import pickle

## Command line arguments are always strings. Tuples/list arguments need a fix.

def get_args_train():
    parser = argparse.ArgumentParser()
    
    ## Training procedure args
    parser.add_argument("--model_name", type=str, default='hit', help="what model to PGD train.  Possible args are vit, resnet, efficientnet, deit_tiny, hit")
    parser.add_argument("--train_all_params", type=bool, default=True, help="Train the entire network or just the classifier")
    parser.add_argument("--train_steps", type=int, default=20000, help="number_of_steps_train")
    parser.add_argument("--warmup_steps", type=int, default=500, help='learning rate warm up steps')
    parser.add_argument("--learning_rate", type=float, default=0.03, help='learning rate to train the model')
    parser.add_argument("--num_workers", type=int, default=8, help='number of workers for the data loader')
    parser.add_argument("--pgd_steps", type=int, default=20, help='number of PGD steps')
    parser.add_argument("--attack_key", type=str, default='L2PGD', help='The PGD attack type')
    parser.add_argument("--epsilon", type=float, default=0.0, help='epsilon to attack the model')
    parser.add_argument("--batch_size", type=int, default=128, help='batch size')
    parser.add_argument("--number_of_gpus", type=int, default=8, help='number of GPUs to train the model with')
    parser.add_argument("--sample_size", type=int, default=5000, help='sample size to collect stats from test and val datasets')
    parser.add_argument("--resume_training", type=bool, default=False, help='To resume training from checkpoint')
    parser.add_argument("--base_directory", type=str, default= "/cluster/scratch/pmayilvahana/deit_hr_week13/", help="directory to save everything")
    parser.add_argument("--bounds", type=tuple, default=(0, 1), help='bounds for foolbox fmodel')

    parser.add_argument("--optimizer", type=str, default='AdamW', help='SGD/Adam/AdamW')
    parser.add_argument("--weight_decay", type=float, default=0.0,help='weight decay for optim')
    parser.add_argument("--momentum", type=float, default=0.9,help='momentum for optim')

    parser.add_argument("--colab", type=bool, default=False,help='marker for training on colab')
    ## Dataset args
    parser.add_argument("--dataset", type=str, default='CIFAR10', help="the dataset you want to train or test on. Possible args are CIFAR10, IMAGENET12")
    parser.add_argument("--data_root", type=str, default='/cluster/project/infk/krause/pmayilvahana/datasets/CIFAR10/data', help="root location of dataset")
    parser.add_argument("--val_split_size", type=float, default=0.1, help="split size of validation")

    ## ViT/DeiT args
    parser.add_argument("--num_scales", type=int, default='5', help="number of scales for multi-scale ViT")
    parser.add_argument("--ablation", type=bool, default=False, help="fix qk values for ablation study")
    parser.add_argument("--sa_stats", type=bool, default=False, help="retreive self attention stats like q, k, v, attn etc.")
    parser.add_argument("--interpolate_pos_embedding", type=bool, default=True, help="interpolate pos embedding or just take average")
    parser.add_argument("--rw_attn", type=str, default='hierarchical', help="reweight attention, options: standard, hierarchical")
    parser.add_argument("--rw_coeff", type=float, default=1, help="reweight attention, options: 2, 4. Note: if hierarchical, always give at least 1. If standard, give 2 or 4. ")

    """ Can make the entire chunk/s below cleaner
    """

    ap = parser.parse_args()

    ## Create args and add some extra things (make it cuter later)
    args = vars(ap)

    ## Getting the model key and model input size

    if args['model_name'] == 'vit': ## ViT
        args['model_key'] = str(1)
        args['model_input_size'] = 384

    elif args['model_name'] == 'resnet': ## Resnet
        args['model_key'] = str(2)
        args['model_input_size'] = 384

    elif args['model_name'] == 'efficientnet': ## Efficientnet
        args['model_key'] = str(3)
        args['model_input_size'] = 380

    elif args['model_name'] == 'deit_tiny':  ##  ## deit tiny
        args['model_key'] = str(4)
        args['model_input_size'] = 224

    elif args['model_name'] == 'hit':  ##  ## deit tiny
        args['model_key'] = str(5)
        args['model_input_size'] = 32

    else:
        raise ValueError('model name = '+args['model_name']+' is not in the list')


    ## Get the epsilon key and eps stuff (This for creating directories). Remove args['eps_stack'] after checking.

    if args['dataset'] == 'CIFAR10':
        args['eps_stack_temp'] = [0.0, 0.5, 1.0]
        args['eps_key'] = str(args['eps_stack_temp'].index(args['epsilon'])+1)
        args['eps_stack'] = [args['epsilon']]
    elif args['dataset'] == 'IMAGENET12':
        args['eps_stack_temp'] = [0.0, 6.0, 12.0]
        args['eps_key'] = str(args['eps_stack_temp'].index(args['epsilon'])+1)
        args['eps_stack'] = [args['epsilon']]
        

    ## Make base directory if it doesn't exits
        
    if not os.path.exists(args['base_directory'] ) and args['colab'] is False:
        os.makedirs(args['base_directory'])

    ## Reedit and make new directories where we want to save

    args['directory'] = args['base_directory']+args['eps_key']+str(args['num_scales'])+"/"
    args['directory_net']=args['directory']+"net/"

    if args['colab'] is False and not os.path.exists(args['directory_net']):
        os.makedirs(args['directory_net'])

    
    ## Dataset stuff (need to add normalize for small imagenet)

    if args['dataset'] == 'CIFAR10':
        args['mean_cifar'] = (0.449, 0.449, 0.449)
        args['std_cifar'] = (0.226, 0.226, 0.226)
        args['normalize'] = transforms.Normalize(args['mean_cifar'], args['std_cifar'])

        args['num_classes'] = 10
        args['image_size'] = 32
        args['upsample'] = True

        args['total_train_size'] = 50000
        args['train_size'] = (1-args['val_split_size'])*args['total_train_size']
        args['val_size'] = args['val_split_size']*args['total_train_size']
        args['test_size'] = 10000
        
    elif args['dataset'] == 'IMAGENET12':
        args['mean_imgnet12'] = (0.5, 0.5, 0.5)
        args['std_imgnet12'] = (0.5, 0.5, 0.5)
        args['normalize'] = transforms.Normalize(args['mean_imgnet12'], args['std_imgnet12'])
        
        args['num_classes'] = 12
        args['image_size'] = 384
        args['upsample'] = False

        args['total_train_size'] = 76310
        args['train_size'] = (1-args['val_split_size'])*args['total_train_size']
        args['val_size'] = args['val_split_size']*args['total_train_size']
        args['test_size'] = 4550

    else:
         raise ValueError('dataset = '+args['dataset']+' is not in the list')

    ## Don't upsample if HiT and CIFAR10 (very ugly)
    if args['dataset'] == 'CIFAR10' and args['model_name'] == 'hit':
        args['upsample'] = False

    ## number of epochs
    args['num_epochs'] = int(args['train_steps']//(args['train_size']/args['batch_size']))


    ## Resuming training arguments
    args['resume_epoch'] = 0

    ## Save args
    if args['colab'] is False:
        with open(args['directory']+'args.pickle', 'wb') as handle:
            pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return args


def get_args_eval():
    parser = argparse.ArgumentParser()

    ## basic eval procedure arguments
    parser.add_argument("--model_name", type=str, default='hit', help="what model to PGD eval. Possible args are vit, resnet, efficientnet, deit_tiny, hit")
    parser.add_argument("--train_all_params", type=bool, default=True, help="Train the entire network or just the classifier")
    parser.add_argument("--learning_rate", type=float, default=0.03, help='learning rate to train the model')
    parser.add_argument("--num_workers", type=int, default=8, help='number of workers for the data loader')
    parser.add_argument("--pgd_steps", type=int, default=20, help='number of PGD steps')
    parser.add_argument("--epsilon", type=float, default=0.0, help='epsilon to attack the model')
    parser.add_argument("--batch_size", type=int, default=128, help='batch size')
    parser.add_argument("--number_of_gpus", type=int, default=8, help='number of GPUs to train the model with')
    parser.add_argument("--base_directory", type=str, default= "/cluster/scratch/pmayilvahana/deit_hr_week13/", help="directory to save everything") ## needs rework
    parser.add_argument("--base_directory_weights", type=str, default= "/cluster/scratch/pmayilvahana/deit_hr_week13/", help="directory with weights") ## needs rework
    
    parser.add_argument("--bounds", type=tuple, default=(0, 1), help='bounds for foolbox fmodel')
    parser.add_argument("--weight_decay", type=float, default=0.0,help='weight decay for optim')
    parser.add_argument("--optimizer", type=str, default='AdamW', help='SGD/Adam/AdamW')
    parser.add_argument("--momentum", type=float, default=0.9,help='momentum for optim')

    parser.add_argument("--colab", type=bool, default=False,help='marker for training on colab')

    ## eval.py arguments

    parser.add_argument("--eval_attack_keys", type=list, default=['L2PGD', 'LinfPGD', 'FGSM'], help='attacks you want to evaluate your model on')
    parser.add_argument("--eval_data_keys", type=list, default=['test'], help='evaluate on dataset type')
    parser.add_argument("--eval_sample_size", type=int, default=100000, help='evaluate on dataset size of 10k')
    parser.add_argument("--num_images", type=int, default=128, help='number of images and adversaries to save')
   
    #parser.add_argument("--eps_stack_L2PGD", type=list, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],help='epsilon stack for l2 pgd attacks')
    #parser.add_argument("--eps_stack_LinfPGD", type=list, default=[0.0, 0.002, 0.004, 0.006, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05],help='epsilon stack for linf pgd attacks')
    #parser.add_argument("--eps_stack_FGSM", type=list, default=[0.0, 0.002, 0.004, 0.006, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05],help='epsilon stack for FGSM  attacks')

    ## Dataset args
    parser.add_argument("--dataset", type=str, default='CIFAR10', help="the dataset you want to train or test on. Possible args are CIFAR10, IMAGENET12")
    parser.add_argument("--data_root", type=str, default='/cluster/project/infk/krause/pmayilvahana/datasets/CIFAR10/data', help="root location of dataset")
    parser.add_argument("--val_split_size", type=float, default=0.1, help="split size of validation")
    
    ## ViT args
    parser.add_argument("--num_scales", type=int, default='5', help="number of scales for multi-scale ViT")
    parser.add_argument("--ablation", type=bool, default=False, help="fix qk values for ablation study")
    parser.add_argument("--sa_stats", type=bool, default=False, help="retreive self attention stats like q, k, v, attn etc.")
    parser.add_argument("--interpolate_pos_embedding", type=bool, default=True, help="interpolate pos embedding or just take average")
    parser.add_argument("--rw_attn", type=str, default='hierarchical_peers', help="reweight attention, options: standard, hierarchical, hierarchical_peers, hierarchical_vertical")
    parser.add_argument("--rw_coeff", type=float, default=1, help="reweight attention, options: 2, 4. Note: if hierarchical, always give at least 1. If standard, give 2 or 4. ")

    ap = parser.parse_args()

    ## Create args and add some extra things 
    args = vars(ap)

    ## Getting the model key and model input size

    if args['model_name'] == 'vit': ## ViT
        args['model_key'] = str(1)
        args['model_input_size'] = 384

    elif args['model_name'] == 'resnet': ## Resnet
        args['model_key'] = str(2)
        args['model_input_size'] = 384

    elif args['model_name'] == 'efficientnet': ## Efficientnet
        args['model_key'] = str(3)
        args['model_input_size'] = 380

    elif args['model_name'] == 'deit_tiny':  ## deit tiny
        args['model_key'] = str(4)
        args['model_input_size'] = 224

    elif args['model_name'] == 'hit':  ##  ## deit tiny
        args['model_key'] = str(5)
        args['model_input_size'] = 32

    ## Get the epsilon key and eps stuff for training/directories

    if args['dataset'] == 'CIFAR10':
        args['eps_stack_temp'] = [0.0, 0.5, 1.0]
        args['eps_key'] = str(args['eps_stack_temp'].index(args['epsilon'])+1)
        args['eps_stack'] = [args['epsilon']]
    elif args['dataset'] == 'IMAGENET12':
        args['eps_stack_temp'] = [0.0, 6.0, 12.0]
        args['eps_key'] = str(args['eps_stack_temp'].index(args['epsilon'])+1)
        args['eps_stack'] = [args['epsilon']]


    ## Attack stack 
    if args['dataset'] == 'CIFAR10':
        args['eps_stack_L2PGD'] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        args['eps_stack_LinfPGD'] = [0.0, 0.002, 0.004, 0.006, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05]
        args['eps_stack_FGSM'] = [0.0, 0.002, 0.004, 0.006, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05]
    elif args['dataset'] == 'IMAGENET12':
        args['eps_stack_L2PGD'] = [0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0]
        args['eps_stack_LinfPGD'] = [0.0, 0.002, 0.004, 0.006, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05]
        args['eps_stack_FGSM'] = [0.0, 0.002, 0.004, 0.006, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05]


    ## Make parent directory if it doesn't exits
    if not os.path.exists(args['base_directory'] ) and args['colab'] is False:
        os.makedirs(args['base_directory'])

    ## Reedit and make new directories where we want to save

    args['directory'] = args['base_directory']+args['eps_key']+str(args['num_scales'])+"/eval/"
    args['directory_net'] = args['base_directory_weights']+args['eps_key']+str(args['num_scales'])+"/net/"
    if args['colab'] is False and not os.path.exists(args['directory']):
        os.makedirs(args['directory'])

    ## Dataset stuff (need to add small imagenet)
    if args['dataset'] == 'CIFAR10':
        args['mean_cifar'] = (0.449, 0.449, 0.449)
        args['std_cifar'] = (0.226, 0.226, 0.226)
        args['normalize'] = transforms.Normalize(args['mean_cifar'], args['std_cifar'])

        args['num_classes'] = 10
        args['image_size'] = 32
        args['upsample'] = True

        args['total_train_size'] = 50000
        args['train_size'] = (1-args['val_split_size'])*args['total_train_size']
        args['val_size'] = args['val_split_size']*args['total_train_size']
        args['test_size'] = 10000
        
    elif args['dataset'] == 'IMAGENET12':
        args['mean_imgnet12'] = (0.5, 0.5, 0.5)
        args['std_imgnet12'] = (0.5, 0.5, 0.5)
        args['normalize'] = transforms.Normalize(args['mean_imgnet12'], args['std_imgnet12'])

        args['num_classes'] = 12
        args['image_size'] = 384
        args['upsample'] = False

        args['total_train_size'] = 76310
        args['train_size'] = (1 - args['val_split_size']) * args['total_train_size']
        args['val_size'] = args['val_split_size'] * args['total_train_size']
        args['test_size'] = 4550

    else:
         raise ValueError('dataset = '+args['dataset']+' is not in the list')

    ## Don't upsample if HiT and CIFAR10 (very ugly)
    if args['dataset'] == 'CIFAR10' and args['model_name'] == 'hit':
        args['upsample'] = False

    ## Getting save length to save images and gradients
    args['save_length'] = args['num_images']//args['batch_size']

    ## Save args
    if args['colab'] is False:
        with open(args['directory']+'args.pickle', 'wb') as handle:
            pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return args

