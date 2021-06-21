import time

__all__ = ['Progbar']

class Param:
    pass

class Progbar():
    def __init__(self, target, width=30, verbose=1, unit_name='step'):
        self.param = Param
        self.param.width = 30
        self.param.target = target
        self.param.time = time.time()
        self.param.n = 0
        self.param.unit_name = unit_name
        self.param.verbose = verbose
        self.param.current = 0

    def add(self, current, values=None):
        self.param.n += 1
        self.param.current += current
        if self.param.target is not None:
            percent = int(self.param.current/self.param.target*self.param.width)
            msg = f"{self.param.current}/{self.param.target} "
            msg = msg+f"[{('='*percent+'>'+'.'*(self.param.width-percent))[:self.param.width]}] "
        else:
            msg = f"{self.param.current}/Unknown "
        
        time_diff = time.time()-self.param.time
        if self.param.target is not None:
            if current<self.param.target:
                msg = msg+f"- EAT: {int(time_diff/self.param.current*(self.param.target-self.param.current))}s"
            else:
                msg = msg+f"- {int(time_diff)}s {int(time_diff/self.param.n*1000)}ms/{self.param.unit_name}"
        else:
            msg = msg+f"- {int(time_diff)}s {int(time_diff/self.param.n*1000)}ms/{self.param.unit_name}"
        if values is not None:
            msg = msg+' - '+''.join([f"{i[0]}: {i[1]} " for i in values])
        if self.param.verbose:
            print(msg, end='\r')
            
    def update(self, current, values=None):
        self.param.n += 1
        if self.param.target is not None:
            if current>self.param.target:
                raise
        if self.param.target is not None:
            percent = int(current/self.param.target*self.param.width)
            msg = f"{current}/{self.param.target} "
            msg = msg+f"[{('='*percent+'>'+'.'*(self.param.width-percent))[:self.param.width]}] "
        else:
            msg = f"{current}/Unknown "
        
        time_diff = time.time()-self.param.time
        if self.param.target is not None:
            if current<self.param.target:
                msg = msg+f"- EAT: {int(time_diff/current*(self.param.target-current))}s"
            else:
                msg = msg+f"- {int(time_diff)}s {int(time_diff/self.param.n*1000)}ms/{self.param.unit_name}"
        else:
            msg = msg+f"- {int(time_diff)}s {int(time_diff/self.param.n*1000)}ms/{self.param.unit_name}"
        if values is not None:
            msg = msg+' - '+''.join([f"{i[0]}: {i[1]} " for i in values])
        if self.param.verbose:
            print(msg, end='\r')
