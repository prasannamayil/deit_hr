"""
Changed attention module mainly, and other modules to accommodate higher scales and reweighting mechanisms.
Attention module with multiple scales needs rework because it is not general and it only works for ViT-B_16
"""


## model.py Imports
import timm
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np


from resnet import StdConv2d
from utils import (get_width_and_height_from_size, load_pretrained_weights,
                    get_model_params, edit_pos_embedding)

## import from timm_v2
from timm_v2 import *
##<====================== Edited ViT CLASS from tchzhangzi =====================>

VALID_MODELS = ('ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32', 'R50+ViT-B_16')


class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, emb_dim))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding

        if self.dropout:
            out = self.dropout(out)

        return out


class MlpBlock(nn.Module):
    """ Transformer Feed-Forward Block """
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):

        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        out = self.dropout2(out)
        return out


class LinearGeneral(nn.Module):
    def __init__(self, in_dim=(768, ), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()

        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return a


class SelfAttention(nn.Module):
    def __init__(self, in_dim, num_scales, heads=8, dropout_rate=0.1, ablation=False, sa_stats=False, rw_attn=None, rw_coeff=None):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim**0.5

        self.query = LinearGeneral((in_dim, ), (self.heads, self.head_dim))
        self.key = LinearGeneral((in_dim, ), (self.heads, self.head_dim))
        self.value = LinearGeneral((in_dim, ), (self.heads, self.head_dim))
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim, ))
        
        self.num_scales=num_scales ## num_scales       
        self.ablation=ablation ## Ablation model
        self.sa_stats=sa_stats ## If you need self attention stats
        self.rw_attn = rw_attn ## Reweight attention


        ### Getting the reweighting matrix (need to fix for the general case ViT)
        
        """ Note that in the standard attention matrix, the rows 'i' is obtained from
            querying 'i' and key-ing other tokens to get the attention weights for token 'i'. 
            Therefore the end of each row, with the higher scale attn wts need reweighting.
        """
        if rw_attn == 'standard':
            dim_rw = num_tokens = 576 ## This needs rework
            dim_rw += 1
            num_tokens_sqrt = np.sqrt(num_tokens)

            for i in range(num_scales-1): dim_rw += int(num_tokens_sqrt/(2**(i+1)))*int(num_tokens_sqrt/(2**(i+1)))  ## each row should be integerized

            self.reweighting_matrix = torch.nn.Parameter(torch.ones(dim_rw, dim_rw))
            self.reweighting_matrix.requires_grad = False  

            start_dim = num_tokens+1
            for i in range(num_scales-1):
                dim_length = int(num_tokens_sqrt/(2**(i+1)))*int(num_tokens_sqrt/(2**(i+1)))
                end_dim = int(start_dim+dim_length)
                self.reweighting_matrix[:,start_dim:end_dim] = rw_coeff**(i+1)
                start_dim = end_dim

        elif rw_attn == 'hierarchical':
            print("will code in a bit")

        ## Saving these stats for each encoder
        if self.sa_stats:
            self.q_val = None
            self.k_val = None
            self.v_val = None
            
            self.out_val = None
            self.attn_weights = None


        if self.ablation:
            self.q_val = None
            self.k_val = None
            self.v_val = None

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        b, n, _ = x.shape

        ## Ablation vs not ablation
        if self.ablation:
            ## 'if' for normal forward pass, 'else' for adv forward pass
            if self.q_val == None and self.k_val == None and self.v_val == None:
                q = self.query(x, dims=([2], [0]))
                k = self.key(x, dims=([2], [0]))
                v = self.value(x, dims=([2], [0]))
            else:
                ## Fixing just q and k
                q = self.q_val
                k = self.k_val
                v = self.value(x, dims=([2], [0]))


            ## Store values before permutatation
            self.q_val = q.detach()
            self.k_val = k.detach()
            self.v_val = v.detach()

        else: ## Standard net
            q = self.query(x, dims=([2], [0]))
            k = self.key(x, dims=([2], [0]))
            v = self.value(x, dims=([2], [0]))

            ## Store values before permutatation
            if self.sa_stats:
                self.q_val = q.detach()
                self.k_val = k.detach()
                self.v_val = v.detach()              


        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
               
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        ## Reweighting attention
        if self.rw_attn is not None:
            attn_weights = attn_weights*self.reweighting_matrix ## Softmax before because of some numerical instability issue
            attn_weights = attn_weights/torch.sum(attn_weights, dim = -1, keepdim=True)

        out = torch.matmul(attn_weights, v)

        ## Storing out and attn weights
        if self.sa_stats:
            self.attn_weights = attn_weights.detach()
            self.out_val = out.detach()

        out = out.permute(0, 2, 1, 3)

        out = self.out(out, dims=([2, 3], [0, 1]))

        return out



class EncoderBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 mlp_dim,
                 num_heads,
                 num_scales,
                 dropout_rate=0.1,
                 attn_dropout_rate=0.1, 
                 ablation=False, sa_stats=False, rw_attn=None, rw_coeff=None):
        super(EncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SelfAttention(in_dim,
                                  heads=num_heads,
                                  num_scales=num_scales,
                                  dropout_rate=attn_dropout_rate,
                                  ablation=ablation, sa_stats=sa_stats, rw_attn=rw_attn, rw_coeff=rw_coeff)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.attn(out)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out

        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out


class Encoder(nn.Module):
    def __init__(self,
                 num_patches,
                 emb_dim,
                 mlp_dim,
                 num_scales=1,
                 num_layers=12,
                 num_heads=12,
                 dropout_rate=0.1,
                 attn_dropout_rate=0.0,
                 ablation=False, sa_stats=False, rw_attn=None, rw_coeff=None):
        super(Encoder, self).__init__()

        # positional embedding
        self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate)

        # encoder blocks
        in_dim = emb_dim
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(in_dim, mlp_dim, num_heads, num_scales, dropout_rate,
                                 attn_dropout_rate, ablation=ablation, sa_stats=sa_stats, rw_attn=rw_attn, rw_coeff=rw_coeff)
            self.encoder_layers.append(layer)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):

        out = self.pos_embedding(x)

        for layer in self.encoder_layers:
            out = layer(out)

        out = self.norm(out)
        return out


class VisionTransformer(nn.Module):
    """ Vision Transformer.
        Most easily loaded with the .from_name or .from_pretrained methods.
        Args:
            params (namedtuple): A set of Params.
        References:
            [1] https://arxiv.org/abs/2010.11929 (An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale)
        Example:
            
            
            import torch
            >>> from vision_transformer_pytorch import VisionTransformer
            >>> inputs = torch.rand(1, 3, 256, 256)
            >>> model = VisionTransformer.from_pretrained('ViT-B_16')
            >>> model.eval()
            >>> outputs = model(inputs)
    """
    def __init__(self, params=None, num_scales=1, ablation=False, sa_stats=False, rw_attn=None, rw_coeff=None):
        super(VisionTransformer, self).__init__()
        self._params = params
        self.num_scales = num_scales
        self.ablation = ablation ##Added ablation
        self.sa_stats = sa_stats ##Added sa_stats
        if self._params.resnet:
            self.resnet = self._params.resnet()
            self.embedding = nn.Conv2d(self.resnet.width * 16,
                                       self._params.emb_dim,
                                       kernel_size=1,
                                       stride=1)
        else:
            self.embedding = nn.Conv2d(3,
                                       self._params.emb_dim,
                                       kernel_size=self.patch_size,
                                       stride=self.patch_size)
            
            self.embedding_higher_scales = nn.ModuleDict()
            for i in range(self.num_scales-1):
                scale_avgpool = 2**(i+1)


                self.embedding_higher_scales[str(i+1)] = nn.AvgPool2d(
                                      kernel_size=scale_avgpool,
                                      stride=scale_avgpool)

        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self._params.emb_dim))

        # transformer
        self.transformer = Encoder(
            num_patches=self.num_patches,
            emb_dim=self._params.emb_dim,
            mlp_dim=self._params.mlp_dim,
            num_scales=self.num_scales,
            num_layers=self._params.num_layers,
            num_heads=self._params.num_heads,
            dropout_rate=self._params.dropout_rate,
            attn_dropout_rate=self._params.attn_dropout_rate,
            ablation=self.ablation,
            sa_stats=self.sa_stats, rw_attn=rw_attn, rw_coeff=rw_coeff)

        # classfier
        self.classifier = nn.Linear(self._params.emb_dim,
                                    self._params.num_classes)

    @property
    def image_size(self):
        return get_width_and_height_from_size(self._params.image_size)

    @property
    def patch_size(self):
        return get_width_and_height_from_size(self._params.patch_size)

    @property
    def num_patches(self):
        h, w = self.image_size
        fh, fw = self.patch_size
        if hasattr(self, 'resnet'):
            gh, gw = h // fh // self.resnet.downsample, w // fw // self.resnet.downsample
        else:
            gh, gw = h // fh, w // fw
        return gh * gw

    def extract_features(self, x):
        if hasattr(self, 'resnet'):
            x = self.resnet(x)

        emb = self.embedding(x)  # (n, c, gh, gw)
        emb = emb.permute(0, 2, 3, 1)  # (n, gh, hw, c)
        b, h, w, c = emb.shape
        emb = emb.reshape(b, h * w, c)

        ## higher scale embeddings
        for i in range(self.num_scales-1):
            new_emb = self.embedding(self.embedding_higher_scales[str(i+1)](x))  # (n, c, gh, gw)
            new_emb = new_emb.permute(0, 2, 3, 1)  # (n, gh, hw, c)
            b, h, w, c = new_emb.shape
            new_emb = new_emb.reshape(b, h * w, c)
            emb = torch.cat([emb, new_emb], dim=1) ## keep concatenating them

        # prepend class token
        cls_token = self.cls_token.repeat(b, 1, 1)
        emb = torch.cat([cls_token, emb], dim=1)

        # transformer
        feat = self.transformer(emb)
        return feat

    def forward(self, x):
        feat = self.extract_features(x)

        # classifier
        logits = self.classifier(feat[:, 0])
        return logits

    @classmethod
    def from_name(cls, model_name, num_scales=1, ablation=False, sa_stats=False, rw_attn=None, rw_coeff=None, in_channels=3, **override_params):
        """create an vision transformer model according to name.
        Args:
            model_name (str): Name for vision transformer.
            in_channels (int): Input data's channel number.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'image_size', 'patch_size',
                    'emb_dim', 'mlp_dim',
                    'num_heads', 'num_layers',
                    'num_classes', 'attn_dropout_rate',
                    'dropout_rate'
        Returns:
            An vision transformer model.
        """
        cls._check_model_name_is_valid(model_name)
        params = get_model_params(model_name, override_params)
        model = cls(params, num_scales, ablation, sa_stats, rw_attn, rw_coeff)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls,
                        model_name,
                        num_scales=1,
                        ablation=False,
                        sa_stats=False,
                        rw_attn=None,
                        rw_coeff=None,
                        weights_path=None,
                        in_channels=3,
                        num_classes=1000,
                        **override_params):
        """create an vision transformer model according to name.
        Args:
            model_name (str): Name for vision transformer.
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.
            in_channels (int): Input data's channel number.
            num_classes (int):
                Number of categories for classification.
                It controls the output size for final linear layer.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'image_size', 'patch_size',
                    'emb_dim', 'mlp_dim',
                    'num_heads', 'num_layers',
                    'num_classes', 'attn_dropout_rate',
                    'dropout_rate'
        Returns:
            A pretrained vision transformer model.
        """
        model = cls.from_name(model_name,
                              num_scales,
                              ablation,
                              sa_stats,
                              rw_attn,
                              rw_coeff,
                              num_classes=num_classes,
                              **override_params)
        load_pretrained_weights(model,
                                model_name,
                                weights_path=weights_path,
                                load_fc=(num_classes == 1000))
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name.
        Args:
            model_name (str): Name for vision transformer.
        Returns:
            bool: Is a valid name or not.
        """
        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' +
                             ', '.join(VALID_MODELS))

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.
        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            if hasattr(self, 'resnet'):
                self.resnet.root['conv'] = StdConv2d(in_channels,
                                                     self.resnet.width,
                                                     kernel_size=7,
                                                     stride=2,
                                                     bias=False,
                                                     padding=3)
            else:
                self.embedding = nn.Conv2d(in_channels,
                                           self._params.emb_dim,
                                           kernel_size=self.patch_size,
                                           stride=self.patch_size)



##<====================== MODEL WRAPPERS and GET MODEL=====================>

## Class that wraps upsampler (if any) and the given model. Note that normalize is wrapped with the model and not in the data loader for foolbox to give real advs from real image.

class wrap_model(nn.Module):
    def __init__(self, model, args):
        super(wrap_model, self).__init__()
        self.normalize = args['normalize']
        self.model = model
        if args['upsample']:
            self.upsampler = nn.Upsample(scale_factor=args['model_input_size']/args['image_size'], mode = 'bilinear', align_corners=True) ## Using bilinear because in the ASYML repo they use Resize in the dataloader which has default mode as bilinear
            self.upsample = True
        else:
            self.upsample = False

    
    def forward(self, x):
        x = self.normalize(x)
        if self.upsample:
            x = self.upsampler(x)
        x = self.model(x)
        return x

### Obtaining the model and parallelizing it 

def get_model(args):
    model_name = args['model_name']
    train_all_params = args['train_all_params']
    num_classes=args['num_classes'] ## this needs fix
    
    ## Getting the pretrained 384 vit base model or resnet151
    if model_name == 'vit':
        model = VisionTransformer.from_pretrained('ViT-B_16', num_scales = args['num_scales'], ablation = args['ablation'], sa_stats = args['sa_stats'], rw_attn=args['rw_attn'], rw_coeff=args['rw_coeff'])
        if num_classes != 1000:
            model.classifier = nn.Linear(768, num_classes, bias = True)
        print(f"Loaded vit with classes = {num_classes}")

        ## Finetuning only the classifier
        if not train_all_params:
            for param in model.parameters():
                param.requires_grad = False

            model.classifier.weight.requires_grad = True
            model.classifier.bias.requires_grad = True
            print("Training only the classifier")

        ## Editing the position embedding if num_scales > 1
        if args['num_scales'] > 1:
            edit_pos_embedding(model, interpolate=args['interpolate_pos_embedding'])

    elif model_name =='resnet':
        model = models.resnet152(pretrained = True)

        if num_classes != 1000:
            model.fc = nn.Linear(2048, num_classes, bias = True)
        print(f"Loaded resnet with classes = {num_classes}")

        ## Finetuning only the classifier
        if not train_all_params:
            for param in model.parameters():
                param.requires_grad = False

            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
            print("Training only the classifier")

    elif model_name == 'efficientnet':
        model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=num_classes)
        print(f"Loaded efficientnet with classes = {num_classes}")

        ## Finetuning only the classifier
        if not train_all_params:
            for param in model.parameters():
                param.requires_grad = False

            model.classifier.weight.requires_grad = True
            model.classifier.bias.requires_grad = True
            print("Training only the classifier")

    elif model_name == 'deit_tiny':
        model = create_model('vit_deit_tiny_patch16_224', num_scales = args['num_scales'], attn_stats = args['sa_stats'], rw_attn=args['rw_attn'], rw_coeff=args['rw_coeff'], pretrained=True)
        if num_classes != 1000:
            model.classifier = nn.Linear(192, num_classes, bias = True)
        print(f"Loaded vit with classes = {num_classes}")

        ## Finetuning only the classifier
        if not train_all_params:
            for param in model.parameters():
                param.requires_grad = False

            model.classifier.weight.requires_grad = True
            model.classifier.bias.requires_grad = True
            print("Training only the classifier")

        ## Editing the position embedding if num_scales > 1
        if args['num_scales'] > 1:
            edit_pos_embedding(model, img_size=224, patch_size=16, embedding_size=192, timm=True, interpolate=args['interpolate_pos_embedding'])

    else:
        print("Model doesn't exist")

    return model

