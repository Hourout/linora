__all__ = ['Config']

def Config(**kwargs):
    class config:
        for i, j in kwargs.items():
            locals()[i] = j
        try:
            del i,j
        except:
            pass
    return config()
