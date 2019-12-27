import torch
from torch import optim
from torch import nn
import argparse
from model.RESNET import resnet18,resnet34,resnet50,resnet101,resnet152
from model.vgg import VGG16
from model.mobilenet import mobilenet
from model.shufflenet import shufflenet
import os
import numpy as np
import random
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import _LRScheduler
# from tensorboardX import SummaryWriter
from torch.autograd import Variable
from utils.utils import WarmUpLR,get_acc,load_config,train_tf,test_tf

def main(mode=None):
    
    config = load_config(mode)
    
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    
    train_set = ImageFolder(config.TRAIN_PATH,transform=train_tf)
    length1 = len(train_set)
    train_data=torch.utils.data.DataLoader(train_set,batch_size=config.BATCH_SIZE,shuffle=True)
    iter_per_epoch = len(train_data)

    test_set = ImageFolder(config.TEST_PATH,transform=test_tf)
    test_data=torch.utils.data.DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # INIT GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        print('\nGPU IS AVAILABLE')
        torch.backends.cudnn.benchmark = True
    else:
        config.DEVICE = torch.device("cpu")

     # choose network
    if config.MODEL == 1:
        net = VGG16().to(config.DEVICE)
        print('The Model is VGG\n')
    if config.MODEL == 2:
        net = resnet34().to(config.DEVICE)
        print('The Model is ResNet34\n')  
    if config.MODEL == 3:
        net = mobilenet().to(config.DEVICE)
        print('The Model is mobilenet\n')
    if config.MODEL == 4:
        net = shufflenet().to(config.DEVICE)
        print('The Model is shufflenet\n')
#     print(dir(net))
#     # choose train or test
#     if config.MODE == 1:
#         print("Start Training...\n")
#         net.train()
#     if config.MODE == 2:
#         print("Start Testing...\n")
#         net.test()

    optimizer = optim.SGD(net.parameters(),lr=config.LR,momentum=0.9,weight_decay=5e-4)
    loss_function = nn.CrossEntropyLoss()
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.MILESTONES,gamma=0.2)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * config.WARM)
#     optimizer = optim.Adam(net.parameters(),lr=float(config.LR),betas=(config.BETA1, config.BETA2))

# use tensorboard
    runs_path = os.path.join(config.PATH,'runs')
    if not os.path.exists(runs_path):
        os.mkdir(runs_path)
                 
#     writer=SummaryWriter(log_dir=runs_path)
#     input_tensor = torch.Tensor(12, 3, 32, 32).cuda()
#     writer.add_graph(net, Variable(input_tensor, requires_grad=True))

#create checkpoint folder to save model
    model_path = os.path.join(config.PATH,'model')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    checkpoint_path = os.path.join(model_path,'{epoch}-{type}.pth')
                 
    best_acc = 0.0
    for epoch in range(1, 100):
        if epoch > config.WARM:
            train_scheduler.step(epoch)
    
        ### train ###
        net.train()   
        train_loss = 0.0 # cost function error
        train_correct = 0.0

        for i, data in enumerate(train_data):

            if epoch <= config.WARM:
                warmup_scheduler.step()

            length = len(train_data)
            image, label = data
            image, label = image.to(config.DEVICE),label.to(config.DEVICE)

            output = net(image)
            train_correct += get_acc(output, label)
            loss = loss_function(output, label)
            train_loss +=loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tAcc: {:0.4f}\tLR: {:0.6f}'.format(
                train_loss/(i+1),
                train_correct/(i+1),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=i * 24 + len(image),
                total_samples=len(train_data.dataset)
            ))
        
       ##eval 
        net.eval()
        test_loss = 0.0 # cost function error
        test_correct = 0.0

        for i, data in enumerate(test_data):
            images, labels = data
            images, labels = images.to(config.DEVICE),labels.to(config.DEVICE)

            outputs = net(images)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()
            test_correct += get_acc(outputs, labels)
            
            
            print('Test set: [{test_samples}/{total_samples}]\tAverage loss: {:.4f}, Accuracy: {:.4f}'.format(
            test_loss /(i+1),
            test_correct / (i+1),
            test_samples=i * 24 + len(images),
            total_samples=len(test_data.dataset)
        ))
        print()
        
        acc = test_correct/(i+1)  
      #start to save best performance model after learning rate decay to 0.01 
        if epoch > config.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % config.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(epoch=epoch, type='regular'))
                 
#     writer.close()

# def training(epoch,net):
# #     net.train()   
#     train_loss = 0.0 # cost function error
#     train_correct = 0.0
                 
#     for i, data in enumerate(train_data):
                 
#         if epoch <= args.warm:
#             warmup_scheduler.step()
                 
#         length = len(train_data)
#         image, label = data
#         image, label = image.to(config.DEVICE),label.to(config.DEVICE)

#         output = net(image)
#         train_correct += get_acc(output, label)
#         loss = loss_function(output, label)
#         train_loss +=loss.item()
                 
#         # backward
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
                 
#         n_iter = (epoch - 1) * len(train_data) + i + 1

# #         last_layer = list(net.children())[-1]
# #         for name, para in last_layer.named_parameters():
# #             if 'weight' in name:
# #                 writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
# #             if 'bias' in name:
# #                 writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

#         print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tAcc: {:0.4f}\tLR: {:0.6f}'.format(
#             train_loss/len(train_data),
#             optimizer.param_groups[0]['lr'],
#             train_correct/len(train_data),
#             epoch=epoch,
#             trained_samples=i * config.BATCHSIZE + len(image),
#             total_samples=len(train_data.dataset)
#         ))

#         #update training loss for each iteration
# #         writer.add_scalar('Train/loss', loss.item(), n_iter)

# #         for name, param in net.named_parameters():
# #             layer, attr = os.path.splitext(name)
# #             attr = attr[1:]
# #             writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
       
                 
# def eval_training(epoch,net):
#     net.eval()

#     test_loss = 0.0 # cost function error
#     test_correct = 0.0

#     for i, data in enumerate(test_data):
#         images, labels = data
#         images, labels = images.to(config.DEVICE),labels.to(config.DEVICE)

#         outputs = net(images)
#         loss = loss_function(outputs, labels)
#         test_loss += loss.item()
#         test_correct += get_acc(outputs, labels)

#     print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
#         test_loss / len(test_loader.dataset),
#         test_correct / len(test_data.dataset)
#     ))
#     print()

#     #add informations to tensorboard
#     writer.add_scalar('Test/Average loss', test_loss / len(test_data.dataset), epoch)
#     writer.add_scalar('Test/Accuracy', test_correct.float() / len(test_data.dataset), epoch)

#     return test_correct.float() / len(test_data.dataset)                 

if __name__ == "__main__":
    main()
