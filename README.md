## SRGFormer

### Overview

![Overview of LGMRec](img/framework.jpg)

### Environment

pip install -r requirements.txt

<!-- ### Data

Download from Google Drive: [Baby/Sports/Clothing](https://drive.google.com/drive/folders/1BxObpWApHbGx9jCQGc8z52cV3t9_NE0f?usp=sharing).
The data contains text and image features extracted from Sentence-Transformers and VGG-16 and has been publiced in [MMRec](https://github.com/enoche/MMRec) framework. -->

### Run

1. Put your downloaded data (e.g. baby) under `data/` dir.
2. Run `train.sh` to train SRGFormer:
    `bash train.sh`
You may specify other parameters in CMD or config with `configs/model/*.yaml` and `configs/dataset/*.yaml`. 
