__all__ = ['config']

def config(**kwargs):
    class Config:
        for i, j in kwargs.items():
            locals()[i] = j
        del i,j
    return Config()
