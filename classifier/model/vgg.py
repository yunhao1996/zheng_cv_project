import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self,num_classes=121,grayscale=False):
        dim = None
        if grayscale==True:
            dim = 1
        else :
            dim = 3
 
        super(VGG16,self).__init__()
        self.vgg_bone = nn.Sequential(
          nn.Conv2d(dim,64,kernel_size=3,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(64,64,kernel_size=3,padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
          nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(128,128,kernel_size=3, padding=1 ),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
          nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
          nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
        )
        self.vgg_logit = nn.Sequential(
        nn.Linear(7*7*512,4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096,4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096,num_classes),
        )
        for m in self.modules():
            if isinstance(m ,nn.Conv2d ):
                m.weight.detach().normal_(0,0.05)
                if m.bias is not None :
                    m.bias.data.detach().zero_()
            elif isinstance(m,nn.Linear):
                m.weight.detach().normal_(0,0.05)
                m.bias.detach().detach().zero_()
    def forward(self,x):
        x =self.vgg_bone(x)
        x =x.view(x.size(0),-1)
        logit  = self.vgg_logit(x)
        prob = F.softmax(logit)

