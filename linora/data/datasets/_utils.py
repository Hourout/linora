from linora import gfile
from linora.utils._progbar import Progbar
from linora.data._utils import assert_dirs, get_file
from linora.data.datasets._config_path import Param

__all__ = ['download', 'list_datasets']


def list_datasets():
    """Support datasets list."""
    return list(Param.__dict__.keys())


def download(data_id, folder=None):
    """"Download datasets.
    
    Args:
        data_id: datasets id. see api: la.data.datasets.list_datasets()
        folder: str, Store the absolute path of the data directory.
    Returns:
        Store the absolute path of the data directory, example: `folder/mnist`.
    """
    task_path = assert_dirs(folder, data_id)
    try:
        data_list = Param.__getattribute__(data_id)
    except:
        raise ValueError(f'Not support datasets `{data_id}`.')
    
    if len(data_list)>1:
        bar = Progbar(len(data_list))
        for r, url in enumerate(data_list):
            root_file = gfile.path_join(task_path, url.split('/')[-1])
            get_file(url, root_file, verbose=0, retries=3, chunk_size=5120)
            bar.update(r+1)
    else:
        url = data_list[0]
        root_file = gfile.path_join(task_path, url.split('/')[-1])
        get_file(url, root_file, verbose=1, retries=3, chunk_size=5120)
    return task_path