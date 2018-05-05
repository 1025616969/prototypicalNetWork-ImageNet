import torch.utils.data as data
import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms as T

IMG_CACHE={}
def get_current_classes(fpath):
    a=os.listdir(fpath)

    classes=[]
    items=[]
    for i in range(len(a)):
        if(a[i].endswith("jpg")):
            items.append(a[i])
            classes.append(a[i].split('0000')[0])

    #print("classes",classes)
    classes=np.unique(classes)
    return classes,items
    # with open(fname) as f:
    #     classes=f.read().splitlines()
    # return classes


# def find_items(root_dir, classes):
#     ret=[]
#     #rots=['/rot000','/rot090','/rot180','/rot270']
#     #root: ./omniglot/data
#     #dirs:[文件夹名字。里面是文字的名字]
#     #files:除了文件夹之外的文件
#     for (root,dirs,files) in os.walk(root_dir):
#
#         for f in files:
#             r=root.split('/')
#             lr=len(r)
#             #在文件夹下：root=./omniglot/data/字符名/characterXX
#             #因此此时label=字符名/characterXX
#             label=r[lr-2]+"/"+r[lr-1]
#             for rot in rots:
#                 if label+rot in classes and (f.endswith("png")):
#                     ret.extend([(f,label,root,rot)])
#                     #ret.append((f,label,root,rot)) 效果一样
#
#     print("Dataset: Found %d items"%len(ret)) #82240 items 每4个items是一张图片旋转而成，所以有20560张图片
#     #ret的格式为：[('0459_14.png','Gujiaef/character42','./omniglot/data/Gujiaef/character42','/rot000'),(...),()]
#     return ret


def index_classes(classes):
    idx={}
    for i,value in enumerate(classes):
        #i[1]为Gujarati/character42   i[-1]为/rot000
        #因此将字符/characNum/rot 作为一个类，给他赋一个数作为label

        if(not value in idx):
            idx[value]=len(idx)
   # print("Dataset: Found %d classes"%len(idx)) #82240个图片，20一类，所以应该有4112个classes
    #print("idx:",idx) # {'n01546543':0, 'n0193535':1 ,...}
    return idx


def load_img(path):
    #path,rot=path.split('/rot')
    if path in IMG_CACHE:
        x=IMG_CACHE[path]
    else:
        x=Image.open(path)
        IMG_CACHE[path]=x
    x=x.resize((28,28))

    shape=1,x.size[0],x.size[1]
    x=np.array(x,np.float32,copy=False)
    x=1.0-torch.from_numpy(x)
    x=x.transpose(0,1).contiguous().view(shape)

    return x


def makeLabel(root,idx_classes, items):
    path_List=[]
    y=[]
    for i in range(len(items)):
        path=os.path.join(root,items[i])
        path_List.append(path)
        className=items[i].split('0000')[0]
        label=idx_classes[className]
        y.append(label)
    return path_List,y





class OmniglotDataset(data.Dataset):
    # vinyal_split_sizes={
    #     'test':"./omniglot/splits/vinyals/test.txt",
    #     'train':"./omniglot/splits/vinyals/train.txt"
    # }
    def __init__(self,mode='train',root='./imgtest/',transforms=None,target_transforms=None):

        self.root=root
        self.transforms=transforms
        self.target_transforms=target_transforms
        self.classes,self.items=get_current_classes(self.root) #返回class名字，前面9个字符  每个图片的全名


       # self.all_items=find_items(self.root,self.classes,self.items)

        self.idx_classes=index_classes(self.classes)

        #paths,self.y=zip(*[self.get_path_label(pl) for pl in range(len(self))])
        self.paths,self.y=makeLabel(root,self.idx_classes,self.items)
        #print("path:",paths) #['./img/n017734350000123.jpg','..',..]
        #print("y:",self.y) #[2,1,1,1,2,0]

        if mode=='train':
            # self.paths=self.paths[:int(0.75*len(self.paths))]
            # self.y=self.y[:int(0.75*len(self.y))]
            print("图片总数:",len(self.paths))


        elif mode=='test':
            # self.paths=self.paths[int(0.75*len(self.paths)):]
            # self.y=self.y[int(0.75*len(self.y)):]
            print("图片总数:", len(self.paths))




        # self.x=map(load_img,paths)
        # self.x=list(self.x)
        # print(self.x)
        if transforms is None:
            normalize=T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

            if mode=='train':
                self.transforms=T.Compose([
                    T.Scale(100),
                    T.RandomSizedCrop(84),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])
            elif mode=='test':
                self.transforms=T.Compose([
                    T.Scale(100),
                    T.CenterCrop(84),
                    T.ToTensor(),
                    normalize
                ])


    # def get_path_label(self, index):
    #     #all_items格式：[('0459_14.png','Gujiaef/character42','./omniglot/data/Gujiaef/character42','/rot000')】
    #     filename=self.all_items[index][0] #235_14.png
    #     rot=self.all_items[index][-1]  #/rot 000
    #     img=str.join('/',[self.all_items[index][2],filename])+rot
    #     #print("img:",img) ./omniglot/data/Guajian/charactere32/0459_14.png/rot000
    #     target=self.idx_classes[self.all_items[index][1]+self.all_items[index][-1]]
    #     #print("target:",target) 0
    #
    #     if self.target_transform is not None:
    #         target=self.target_transform(target)
    #
    #     return img,target

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        x=self.paths[index]
        data=Image.open(x)
        if self.transforms:
            data=self.transforms(data)
        return data,self.y[index]

#a=OmniglotDataset()
