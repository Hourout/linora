__all__ = ['Config']

def Config(**kwargs):
    class Configs:
        for i, j in kwargs.items():
            locals()[i] = j
        del i,j
    return Configs()
