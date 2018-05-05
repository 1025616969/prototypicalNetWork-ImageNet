import os

import torch as t
import time

class BasicModule(t.nn.Module):
    """
    封装了nn.Module,主要提供save和load方法
    """
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))

    def load(self,path):
        self.load_state_dict(t.load(path))

    def save(self,path=None):
        if path is None:
            prefix='results/'+self.model_name+'_'
            name=time.strftime(prefix+'%m%d_%H:%M:%S.pth')
            t.save(self.state_dict(),name)
            return name
        name=os.path.join(path,'best_model.pth')
        t.save(self.state_dict(),name)
        return name
