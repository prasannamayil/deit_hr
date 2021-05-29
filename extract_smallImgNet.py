import numpy as np
import os
from shutil import copy2
import random

random.seed(10)
np.random.seed(10)

### Create Raw miniImgNet dataset ###
cateIdx = dict()
cateIdx['dog'] = list(range(151,269))
cateIdx['cat'] = list(range(281,286))
cateIdx['monkey'] = list(range(364,383))
cateIdx['car'] = [468, 656, 717, 751, 817]
cateIdx['truck'] = [555, 569, 675, 867]
cateIdx['ship'] = [472, 510, 625, 675, 814, 871]
cateIdx['plane'] = [404, 895]
cateIdx['ball'] = [522, 574, 722, 768, 805, 852]
cateIdx['windInst'] = [432, 558,  683, 776]
cateIdx['strInst'] = [402, 486, 546, 889]
cateIdx['fish'] = list(range(7)) + list(range(389,396))
cateIdx['elephant'] = [385, 386]
sourceParentDir = '/home/cjsimon/datasets/imgnet/'
targetParentDir = '/home/cjsimon/datasets/smallImgNet/'
va_counts = []
tr_counts = []
for key, idxs in cateIdx.items():
    files = dict() # Dictionary for train and val
    for name in ['train/', 'val/']:
        files[name] = []
        sourceDir = os.path.join(sourceParentDir, name)
        cateDirs = os.listdir(sourceDir)
        cateDirs.sort()
        for idx in idxs:
            sourceCateDir = os.path.join(sourceDir, cateDirs[idx])
            files[name] += [os.path.join(sourceCateDir, imgName)
                      for imgName in os.listdir(sourceCateDir)]
        random.shuffle(files[name])
  
    files_len = len(files['train/'])+len(files['val/']) 
    va_len = int(np.min([max(files_len/10, 200), 1000]))
    va_len = int(np.ceil(va_len / 10.0)) * 10 # rounding to near 10
    tr_len = min(files_len-va_len, 10000)
    tr_len = int(np.floor(tr_len / 10.0)) * 10 # rounding to lower 10
    va_counts.append(va_len)
    tr_counts.append(tr_len)
    fileDict = dict()
    fileDict['val/'] = files['val/'][:va_len]
    fileDict['train/'] = files['train/'][:tr_len]
    print(key, len(fileDict['train/']), len(fileDict['val/']))
    # Copy to training folder
    for name in ['train/', 'val/']:
        targetDir = os.path.join(targetParentDir, name)
        targetCateDir = os.path.join(targetDir, key)
        if not os.path.exists(targetCateDir):
            os.makedirs(targetCateDir)
        for file in fileDict[name]:
            copy2(file, targetCateDir)
