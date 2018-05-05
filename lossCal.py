import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

def lossCal(output,label,n_support):
    cpuOutput=output.cpu() if output.is_cuda else output

    cpuLabel=label.cpu() if label.is_cuda else label
    cpuLabel=cpuLabel.data

    def supp_idxs(c):
        return torch.nonzero(cpuLabel.eq(int(c)))[:n_support].squeeze()

    classes=np.unique(cpuLabel)
    n_classes=len(classes)
    #print("n_classes应该是60",n_classes)
    n_query=len(torch.nonzero(cpuLabel.eq(int(classes[0]))))-n_support
    #print("n_query应该是5",n_query)

    os_idxs=list(map(supp_idxs,classes))  #os_idxs为每个way的support样本的index

    prototypes=torch.stack([cpuOutput[i].mean(0) for i in os_idxs]) #计算出属于各个way的原型，并且拼起来，这样每一行就是一个way的原型了
    #所以size应该为60*64
    #print("原型的尺寸 60*64",prototypes.size())
    prototypes=prototypes.cuda() if label.is_cuda else prototypes

    #query集的index  尺寸为300
    oq_idxs_0 = torch.stack(list(map(lambda c: torch.nonzero(cpuLabel.eq(int(c)))[n_support:], classes))).view(-1)
    oq_idxs_0 = oq_idxs_0.cuda() if label.is_cuda else oq_idxs_0
    oq = output[oq_idxs_0] #300*64 且按照5个同类一组串起来了
    dists = euclidean_dist(oq, prototypes) #300*60

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1) #(300,60)->(60,5,60)
    print("样本属于各个类的概率如下：",F.log_softmax(-dists, dim=1).exp()) #75*60

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    #print("target——indes",target_inds) #60*5*1
    target_inds = target_inds.long()
    target_inds = Variable(target_inds, requires_grad=False)
    target_inds = target_inds.cuda() if label.is_cuda else target_inds

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean() #计算300个query的平均loss
    _, y_hat = log_p_y.max(2) #值和索引，这里只需要索引就够了

    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean() #计算300个query的正确率

    return loss_val, acc_val


def euclidean_dist(x,y):
    """
    :param x: N*D  300*64
    :param y: M*D  60*64
    :return:
    """
    n=x.size(0) # 300
    m=y.size(0) #60
    d=x.size(1) #64

    assert d==y.size(1)

    x=x.unsqueeze(1).expand(n,m,d)  #（300，1，64）-》(300,60,64)
    y=y.unsqueeze(0).expand(n,m,d)  #(1,60,64)->(300,60,64)

    return torch.pow(x-y,2).sum(2) #将index为2的维度变为0  300*60