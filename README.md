# zheng_cv_project
config.yml
```
MODE: 1             # 1:train,2:test
MODEL: 2            # 1:VGG,2:RESNET
SEED: 10            # random seed
GPU: [1]            # list of gpu ids

TRAIN_PATH: /home/ouc/Documents/dataset/train/
TEST_PATH: /home/ouc/Documents/dataset/val/

LR: 0.1                    # learning rate
BETA1: 0.5                  # adam optimizer beta1
BETA2: 0.9                  # adam optimizer beta2
BATCH_SIZE: 24              # input batch size for training
RESIZE: 96                  # vgg:256,resnet:96
CROP: 96                    # vgg:224,resnet:96
MAX_ITERS: 400000           # maximum number of iterations to train the model
DEVICE: 1
MILESTONES: [30,60,80]
WARM: 1
SAVE_EPOCH: 1
```
## folder---classifier
To achieve images classification. If you want to train this model, please run
```
cd classifier
mkdir checkpoints  # put config.yml to this folder
python train ./checkpoints/
```
if you want to test model which has trained. You can:
```
python test.py 
```
what's more, please note modification points in codes.
## folder --- deploy
To achieve web deploy. Please run:
```
python app.py runserver -h <your ip-address> -p 8000 -d 
```
Then input <your ip-address> :8000 in web
