__all__ = ['Config']

def config(**kwargs):
    class Config:
        for i, j in kwargs.items():
            locals()[i] = j
        try:
            del i,j
        except:
            pass
    return Config()
