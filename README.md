# Robust archs

This repo is for PGD training/finetuning architects like ViT, Resnet152 etc. 

## Things to do before running multiple jobs

### Packages needed

- `foolbox` and `timm`.

### Running multiple jobs

- When running `train.py` or `eval.py` (especially `eval.py`), always explicitly specify the arguments `--model_name` and `--epsilon` as they are used for directory creation. Other variables, give as you require.
- To run multiple jobs (for example), please have a look at the bash scripts `train.sh` and `eval.sh`. Please note that it is important to make the changes below in the `.py` files before you can run the jobs.

### Edits to do and variables to be wary of in config.py

- `--number_of_gpus` - default is 8, please adjust in case you are using more GPUs.
- `--dataset` - should be set to 'IMAGENET12' for Imagenet_small training. If changing default, change both in `get_args_train()` and `get_args_eval()`.
- `--data_root` - specify the root folder of Imagenet_small.  If changing default, change both in `get_args_train()` and `get_args_eval()`.
- `--base_directory` - specify the parent directory where you want to save your results. To train all 3 models (vit, eff, res) on Imagenet, specifying base directory is enough, children directory will be created automatically. If changing default, change both in `get_args_train()` and `get_args_eval()`.
- `--base_directory_weights` - specify the parent directory of model weights, of models you want to evaluate. It is best if you keep `--base_directory_weights` and `--base_directory` as the same.  This is only in `get_args_eval()`. 
- `--batch_size`: You might need to edit defaults in both `get_args_train()` and `get_args_eval()` based on your GPUs.
- `--train_steps`: You might need to edit defaults in`get_args_train()` based on the `batch_size` you give. The smaller the `batch_size` the higher the `train_steps` we need.
- Explicitly fill/edit lines `117-119` and `261-263` with mean, std, and normalization for Imagenet. Please note that this is necessary because we wrap the model with the normalization and do not specify normalization in the dataloader.
- The 3 epsilons for Imagenet that we will train on is `[0.0, 6.0, 12.0]`. If you want to change this then change it in line `78, 216, 227-229`. **Most importantly** change also the eps_array in the bash scripts `eval.sh` and `train.sh`.

### Edits to do in data.py

- `dataloader_IMAGENET12(args)` - This function needs to take in `args` and needs to return `tr_loader, va_loader, te_loader`. Mainly, `args['data_root']` will have the root location of the data. Everything else you might need should be `args`. **Most importantly** use `args['model_input_size']` when you resize the images. This is because efficientnet's input size is 380, and other models' input size is 384. 
- If you are entirely changing the above function (i.e. function name, input arguments), please make changes in `main()` in `train.py` and `eval.py`.
