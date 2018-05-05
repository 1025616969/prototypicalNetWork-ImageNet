# -*- coding:utf-8 -*-
import numpy as np
import torch

from data.dataset import OmniglotDataset


class PrototypicalBatchSampler(object):
    def __init__(self,labels,classes_per_it,num_samples,iterations):
        super(PrototypicalBatchSampler,self).__init__()

        self.labels=labels  #y的集合
        self.classes_per_it=classes_per_it  #train:60  test:5
        self.sample_per_class=num_samples# train:5+5   test:5+15
        self.iterations=iterations #100

        #self.classes为0-4112的tensor   self.counts返回【20，20，20，。。。】，即每个类样本出现的次数，均为20
        self.classes,self.counts=np.unique(self.labels,return_counts=True)
        print("总类别数：",len(self.classes))




        #self.classes=0-4111的LongTensor
        self.classes=torch.LongTensor(self.classes)

        self.idxs=range(len(self.labels)) #【0-82240】

        # 4112行，20列的array
        self.label_tens=np.empty((len(self.classes),max(self.counts)),dtype=int)*np.nan
        self.label_tens=torch.Tensor(self.label_tens)
        self.label_lens=torch.zeros_like(self.classes) #，每个class拥有的样本数量

        #print("self.labels",self.labels) [0,1,2,3,0,1,2,3,----]




        for idx,label in enumerate(self.labels):

            #print("idx,label",idx,label)# 0,0
            #print("没取【0，0】之前：",np.argwhere(self.classes==label)) #0 LongTensor 1*1
            #print("取了0，0之后：",np.argwhere(self.classes==label)[0,0])# 0 一个数
            label_idx=np.argwhere(self.classes==label)[0,0]
            #print("np.where输出：",np.where(np.isnan(self.label_tens[label_idx])))#[0,1,2...,19]array
            #print("label_tens:",self.label_tens)
            self.label_tens[label_idx, np.where(np.isnan(self.label_tens[label_idx]))[0][0]] = idx
            self.label_lens[label_idx] += 1


        #print("最后结果：",self.label_tens) #这个数组第i行表示，label为i的样本在self.label中的下标为多少，如：class=0的样本是第0，4，8，12。。。号样本，
        #print("label_lens",self.label_lens)

    def __iter__(self):
        """
        产生a batch of indexes
        :return:
        """
        spc=self.sample_per_class #10
        cpi=self.classes_per_it #60


        for it in range(self.iterations):
            batch_size=spc*cpi
            batch=torch.LongTensor(batch_size)
            #print("batch:",batch)
            c_idxs=torch.randperm(len(self.classes))[:cpi] #随机选60个类
            for i,c in enumerate(self.classes[c_idxs]):
                s=slice(i*spc,(i+1)*spc)# 10个的切片
                label_idx = np.argwhere(self.classes == c)[0, 0] #c这个label的索引
                sample_idxs = torch.randperm(self.label_lens[label_idx])[:spc] #第c行取10个样本出来，取出的是10个样本在总样本中的索引
                batch[s] = self.label_tens[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        #返回每个epoch的episode数量 100
        return self.iterations


