# -*- coding:utf-8 -*-
from torch import nn
from models.BasicModule import BasicModule

def conv_block(in_channels,out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
class protoNet(BasicModule):

    def __init__(self,x_dim=3,hid_dim=64,z_dim=64):
        super(protoNet,self).__init__()
        print("进入初始化网络!")
        self.model_name='protonet'

        self.encoder=nn.Sequential(
            conv_block(x_dim,hid_dim),
            conv_block(hid_dim,hid_dim),
            conv_block(hid_dim,hid_dim),
            conv_block(hid_dim,z_dim),
        )

    def forward(self,x):
        #print("刚开始的x的size",x.size())  [200,3,84,84]
        x=self.encoder(x)
        #print("encoder之后x的size",x.size())  [200,64,5,5]
        #print("x.size(0):",x.size(0)) 200
        x=x.view(x.size(0),-1)
        #print("view之后x的size",x.size())  [200,1600]
        return x





