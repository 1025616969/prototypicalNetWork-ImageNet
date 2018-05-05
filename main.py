import os
import json
import torch
from tqdm import tqdm

import models.prototypicalNet
from data.batchSampler import PrototypicalBatchSampler
from data.dataset import OmniglotDataset
from torch.autograd import Variable
from lossCal import lossCal
import numpy as np

def train(opt, trainLoader, model, optimizer, lr_schedule,trace_file):


    best_acc=0.0
    if os.path.isfile(trace_file):
        os.remove(trace_file)

    for epoch in range(opt['train.epoches']):
        train_loss = []
        train_acc = []
        print("----epoch: %2d-------"%epoch)
        model.train()
        for data,label in tqdm(trainLoader):

            data=Variable(data)
            label=Variable(label)

            optimizer.zero_grad()
            if(opt['data.cuda']):
                data=data.cuda()
                label=label.cuda()
           # print("data的size:",data.size())
            output=model(data)
           # print("output:",output) # 600*64(200*1600)
            #print("label.size",label.size()) #600
            loss,acc=lossCal(output,label,opt['data.shot'])

            loss.backward()
            optimizer.step()
            train_loss.append(loss)
            train_acc.append(acc)

        avg_loss=np.mean(train_loss)
        avg_acc=np.mean(train_acc)
        avg_loss = avg_loss.cpu() if avg_loss.is_cuda else avg_loss
        avg_acc=avg_acc.cpu() if avg_acc.is_cuda else avg_acc
        lr_schedule.step()
        print("epoch %2d 训练结束：avg_loss:%.4f , avg_acc:%.4f"%(epoch,avg_loss,avg_acc))
        with open(trace_file,'a') as f:
            f.write('epoch:{:2d} 训练结束：avg_loss:{:.4f} , avg_acc:{:.4f}'.format(epoch,avg_loss.data.numpy()[0],avg_acc.data.numpy()[0]))
            f.write('\n')
        avg_acc2=avg_acc.data.numpy()[0]
        if(avg_acc2>best_acc):
            print("产生目前最佳模型，正在保存。。。。。。")
            name=model.save(path=opt['log.exp_dir'])
            best_acc=avg_acc2
            print("保存成功！，保存在：",name)
    return best_acc





def test(opt,testLoader,model):

    model.eval()
    test_acc=[]
    for data,label in tqdm(testLoader):
        data=Variable(data)
        label=Variable(label)
        if opt['data.cuda']:
            data=data.cuda()
            label=label.cuda()

        output=model(data)
        _,acc=lossCal(output,label,opt['data.test_shot'])
        test_acc.append(acc)
    print("输出个结果看看：",test_acc)

    avg_acc = np.mean(test_acc)
    avg_acc=avg_acc.cpu() if avg_acc.is_cuda else avg_acc
    print("%4d 测试结束：avg_acc:%.4f" % (opt['data.test_episodes'], avg_acc))
    return avg_acc.data.numpy()[0]




def main(opt):
    if(opt['run_mode']=='train'):
        if not os.path.isdir(opt['log.exp_dir']):
            os.makedirs(opt['log.exp_dir'])

        with open(os.path.join(opt['log.exp_dir'],'opt.json'),'w') as f:
            json.dump(opt,f)
            f.write('\n')

        trace_file=os.path.join(opt['log.exp_dir'],'trace.txt')

        #设定随机数种子
        torch.manual_seed(1234)
        if opt['data.cuda']:
            torch.cuda.manual_seed(1234)



        #step1: 设置模型
        model=models.prototypicalNet.protoNet()

        if opt['data.cuda']:
            model.cuda()

        #step2: 加载数据
        trainLoader=init_dataset(opt,mode='train')

        #step3: 目标函数和优化器
        optimizer=torch.optim.Adam(model.parameters(),lr=opt['train.learning_rate'])
        lr_schedule=torch.optim.lr_scheduler.StepLR(optimizer=optimizer,gamma=0.5,step_size=opt['train.decay_every'])

        res=train(opt,trainLoader,model,optimizer,lr_schedule,trace_file)
        print("训练结束，训练过程中最好精度为：",res)

    elif opt['run_mode']=='test':
        path=os.path.join(opt['log.exp_dir'],'best_model.pth')
        if not os.path.isfile(path):
            print("还没有模型保存呢，先训练好吗！！！")
            return


        # 设定随机数种子
        torch.manual_seed(1234)
        if opt['data.cuda']:
            torch.cuda.manual_seed(1234)
        #step1: 加载模型
        model=models.prototypicalNet.protoNet()
        model.load(path)
        if opt['data.cuda']:
            model.cuda()

        #step2:加载数据
        testLoader=init_dataset(opt,mode='test')

        res=test(opt,testLoader,model)












def init_dataset(opt,mode='train'):
    '''
    Initialize the datasets, samplers and dataloaders
    '''
    if mode=='train':
        train_dataset = OmniglotDataset(mode='train',root='./data/images/')

        tr_sampler = PrototypicalBatchSampler(labels=train_dataset.y,
                                              classes_per_it=opt['data.way'],
                                              num_samples=opt['data.shot'] + opt['data.query'],
                                              iterations=opt['data.train_episodes'])
        tr_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_sampler=tr_sampler)
        return tr_dataloader

    elif mode=='test':
        test_dataset = OmniglotDataset(mode='test', root='./data/imgtest/')
        test_sampler = PrototypicalBatchSampler(labels=test_dataset.y,
                                                classes_per_it=opt['data.test_way'],
                                                num_samples=opt['data.test_shot'] + opt['data.test_query'],
                                                iterations=opt['data.test_episodes'])

        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_sampler=test_sampler)
        return test_dataloader



