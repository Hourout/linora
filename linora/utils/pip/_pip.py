import itertools
import subprocess

from linora.utils._config import Config

__all__ = ['freeze', 'upgrade', 'upgradeable', 'install', 'uninstall', 
           'mirror', 'download', 'show', 'mirror_set', 'mirror_get',
           'env_view', 'env_clone', 'env_create', 'env_export', 'env_import', 'env_remove']


mirror = Config()
mirror.pip = "https://pypi.org/simple"
mirror.tsinghua = "https://pypi.tuna.tsinghua.edu.cn/simple"
mirror.aliyun = "https://mirrors.aliyun.com/pypi/simple"
mirror.ustc = "https://mirrors.ustc.edu.cn/pypi/web/simple"
mirror.tencent = 'https://mirrors.cloud.tencent.com/pypi/simple'
mirror.douban = 'https://pypi.doubanio.com/simple/'

def libraries_name(name):
    for i in ['==', '>=', '<=', '>', '<']:
        if i in name:
            return name.split(i)+[i]
    for i,j in enumerate(name):
        if j=='-' and name[i+1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return [name[:i], name[i+1:], '-']
    return [name, '', '']

def download(root, name, mirror=mirror.aliyun, py=''):
    """Download python libraries to the specified folder.
    
    Args:
        root: str, dirs.
        name: str or list. libraries name.
        mirror: pip install libraries mirror,
                default official https://pypi.org/simple.
                or you can set mirror='https://pypi.tuna.tsinghua.edu.cn/simple'.
                mirror=la.utils.pip.mirror.pip
        py: python environment.one of ['', '3'].
    Return:
        root: libraries path list.
    """
    if isinstance(name, str):
        if name.endswith('.txt'):
            cmd = f"pip{py} download -r {name} -d {root} -i {mirror}"
            s = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
            s = [i.split(': ') for i in s.decode('utf-8').split('\n')[:-1]]
            s = [i[6:] for i in itertools.chain.from_iterable(s) if 'Saved ' in i]
            return s
        name = [name]
    cmd = f"pip{py} download {' '.join(name)} -d {root} -i {mirror}"
    s = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    s = [i.split(': ') for i in s.decode('utf-8').split('\n')[:-1]]
    s = [i[6:] for i in itertools.chain.from_iterable(s) if 'Saved ' in i]
    return s

def freeze(name=None, py=''):
    """List all python libraries.
    
    Args:
        name: str or list. libraries name.
        py: python environment.one of ['', '3'].
    Return:
        a dict of python libraries version.
    """
    cmd = f"pip{py} list"
    s = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    s = [i.split(' ') for i in s.decode('utf-8').split('\n')[2:-1]]
    s = {i[0]: i[-1] for i in s}
    if not name:
        return s
    if isinstance(name, str):
        name = [name]
    name = [libraries_name(i)[0] for i in name]
    s = {i:s.get(i, '') for i in name}
    return s

def mirror_get(py=''):
    """Get up pip mirrors on your machine.
    
    Args:
        py: python environment.one of ['', '3'].
    Return:
        mirror path.
    """
    cmd = f"pip{py} config list"
    s = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return s.decode('utf-8').strip().split('global.index-url=')[1][1:-1]

def install(name, mirror=mirror.aliyun, py=''):
    """Install python libraries.
    
    Args:
        name: str or list. libraries name. 
              eg. name = 'numpy' or 'numpy==1.0.0' or ['numpy', 'pandas>1.0.0']
        mirror: pip install libraries mirror,
                default official https://pypi.org/simple.
                or you can set mirror='https://pypi.tuna.tsinghua.edu.cn/simple'.
                or mirror=la.utils.pip.mirror.pip
        py: python environment.one of ['', '3'].
    Return:
        a dict of python libraries version.
    """
    if isinstance(name, str):
        if name.startswith('https://github.com/'):
            cmd = f"pip{py} install git+{name}.git"
            s = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
            s = s.decode('utf-8').strip().split('\n')[-1].split(' ')[2:]
            if len(s)>1:
                return 'Install Failed'
            return freeze(name=s, py=py)
        elif name.endswith('.whl'):
            cmd = f"pip{py} install {name}"
            s = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
            s = s.decode('utf-8').strip().split('\n')[-1].split(' ')[2:]
            return freeze(name=s, py=py)
        elif name.endswith('.txt'):
            cmd = f"pip{py} install -r {name}"
            s = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
            s = s.decode('utf-8').strip().split('\n')[-1].split(' ')[2:]
            return freeze(name=s, py=py)
        name = [name]
    name = [libraries_name(i) for i in name]
    name1 = [i[0] for i in name]
    name2 = [i[0] for i in name if i[2]=='']
    name2 = name2 + [i[0]+'=='+i[1] for i in name if i[2]=='-']
    name2 = name2 + [i[0]+i[2]+i[1] for i in name if i[2] not in ['', '-']]
    name2 = ['"'+i+'"' for i in name2]
    cmd = f"pip{py} install {' '.join(name2)} -i {mirror}"
    subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return freeze(name=name1, py=py)

def search(name, mirror=mirror.aliyun, py=''):
    """Show python libraries.
    
    Args:
        name: str. libraries name.
        mirror: pip install libraries mirror,
                default official https://pypi.org/simple.
                or you can set mirror='https://pypi.tuna.tsinghua.edu.cn/simple'.
                or mirror=la.utils.pip.mirror.pip
        py: python environment.one of ['', '3'].
    Return:
        a dict of python libraries version information.
    """
    assert isinstance(name, str), "`name` should be str."
    cmd = f"pip{py} install pip-search -U -i {mirror}"
    subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    cmd = f"pip_search{py} {name}"
    s = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    s = [i.strip() for i in s.decode('utf-8').split('\n')[:-1]]
    s = [s[r:-1] for r, i in enumerate(s) if 'numpy' in i][1]
    s = [[i.strip() for i in j[3:-1].split('│')] for j in s]
    result = {}
    for i in s:
        if i[0]!='':
            result[i[0]] = {'version':i[1], 'released':i[2], 'description':i[3]}
            t = i[0]
        else:
            result[t]['description'] += ' '+i[3]
    return result

def mirror_set(mirror, py=''):
    """Set up pip mirrors on your machine.
    
    Args:
        mirror: pip install libraries mirror,
                default official https://pypi.org/simple.
                or you can set mirror='https://pypi.tuna.tsinghua.edu.cn/simple'.
                or mirror=la.utils.pip.mirror.pip
        py: python environment.one of ['', '3'].
    Return:
        mirror file path.
    """
    subprocess.Popen(f"pip{py} install pip -U -i {mirror}", stdout=subprocess.PIPE, shell=True).communicate()[0]
    cmd = f"pip{py} config set global.index-url {mirror}"
    s = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return s.decode('utf-8').strip()

def show(name, py=''):
    """Show python libraries.
    
    Args:
        name: str or list. libraries name.
        py: python environment.one of ['', '3'].
    Return:
        a dict of python libraries version information.
    """
    if isinstance(name, str):
        name = [name]
    name = [libraries_name(i)[0] for i in name]
    name = freeze(name, py)
    t = {}
    for lib in name:
        if name[lib]=='':
            t[lib] = {}
        else:
            cmd = f"pip{py} show {lib}"
            s = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
            s = [i.split(': ') for i in s.decode('utf-8').split('\n')[:-1]]
            s = {i[0]: i[1] for i in s}
            t[lib] = s
    return t

def uninstall(name, py=''):
    """Uinstall python libraries.
    
    Args:
        name: str or list. libraries name.
        py: python environment.one of ['', '3'].
    Return:
        uninstall log dict.
    """
    if isinstance(name, str):
        name = [name]
    name = [libraries_name(i)[0] for i in name]
    name = freeze(name, py)
    name1 = {i:'Notexist' for i in name if name[i]==''}
    name = [i for i in name if name[i]!='']
    cmd = f"pip{py} uninstall {' '.join(name)} -y"
    s = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    s = ' '.join([i.strip() for i in s.decode('utf-8').strip().split('\n') if 'Successfully uninstalled' in i])
    s = {i:'Success' if i in s else 'Failed' for i in name}
    return {**s, **name1}

def upgrade(name, version=None, mirror=mirror.aliyun, py='', logger=False):
    """Upgrade python libraries.
    
    Args:
        name: str or list. libraries name.
        version: str or list. libraries version.
        mirror: pip install libraries mirror,
                default official https://pypi.org/simple.
                or you can set eg. mirror='https://pypi.tuna.tsinghua.edu.cn/simple'.
                or eg. mirror=la.utils.pip.mirror.tsinghua
        py: python environment.one of ['', '3'].
        logger: whether to print the log.
    Return:
        a dict of python libraries version.
    """
    assert version is None or isinstance(version, (str, list)), "`version` should be None or str or list."
    if isinstance(name, str):
        name = [name]
    old_lib = freeze(name=name, py=py)
    if version is not None:
        if isinstance(version, str):
            version = [version]
        assert len(name)==len(version), "`name` and `version` should be same number."
        for (dist, ver) in zip(name, version):
            if ver=='':
                cmd = f"pip{py} install --upgrade {dist} -i {mirror}"
            else:
                if len([i for i in ['==', '>', '<', '>=', '<='] if i in ver])==0:
                    ver = '=='+ver
                cmd = f"pip{py} install --upgrade {dist}{ver} -i {mirror}"
            if logger:
                subprocess.call(cmd, shell=True)
            else:
                subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    else:
        for dist in old_lib:
            cmd = f"pip{py} install --upgrade {dist} -i {mirror}"
            if logger:
                subprocess.call(cmd, shell=True)
            else:
                subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    new_lib = freeze(name=name, py=py)
    lib = {i:{'old_version': old_lib[i], "new_version":new_lib[i]} for i in new_lib}
    return lib

def upgradeable(mirror=mirror.aliyun, py=''):
    """Veiw upgradeable python libraries.
    
    Args:
        mirror: pip install libraries mirror,
                default official https://pypi.org/simple.
                or you can set mirror='https://pypi.tuna.tsinghua.edu.cn/simple'.
                or mirror=la.utils.pip.mirror.pip
        py: python environment.one of ['', '3'].
    Return:
        a dict of python libraries version.
    """
    cmd = f"pip{py} list -o -i {mirror}"
    s = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    s = [i.strip().split(' ') for i in s.decode('utf-8').split('\n')[2:-1]]
    s = [list(filter(lambda x:len(x)>0, i)) for i in s]
    s = {i[0]:{'version':i[1], 'latest':i[2], 'type':i[-1]} for i in s}
    return s

def env_view():
    """Get virtual environment info."""
    cmd = f"conda info -e"
    s = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    s = s.decode('utf-8').strip().split('\n')[2:]
    s = [i.split(' ') for i in s]
    return {i[0]:i[-1] for i in s}

def env_clone(new_name, old_name):
    """copy already exists virtual environment.
    
    Args:
        new_name: new virtual environment.
        old_name: already exists virtual environment.
    Return:
        log info.
    """
    cmd = 'conda update -n base -c defaults conda'
    subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    s = view_env()
    if old_name not in s:
        return f'Virtual environment {old_name} not exists.'
    cmd = f"conda create -n {new_name} --clone {old_name}"
    subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    s = view_env()
    if new_name in s:
        return 'Virtual environment successfully created.'
    return 'Virtual environment failed created.'

def env_create(name, version):
    """Create virtual environment.
    
    Args:
        name: virtual environment.
        version: python version.
    Return:
        log info.
    """
    cmd = 'conda update -n base -c defaults conda'
    subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    s = view_env()
    if name in s:
        return 'Virtual environment already exists.'
    cmd = f"conda create -n {name} python={version} -y"
    subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    s = view_env()
    if name in s:
        return 'Virtual environment successfully created.'
    return 'Virtual environment failed created.'

def env_export(yml):
    """Export virtual environment to `.yml` format file.
    
    Args:
        yml: `.yml` format file.
    Return:
        log info.
    """
    cmd = 'conda update -n base -c defaults conda'
    subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    cmd = f"conda env export > {yml}"
    subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return yml

def env_import(yml):
    """Import virtual environment from `.yml` format file.
    
    Args:
        yml: `.yml` format file.
    Return:
        log info.
    """
    cmd = 'conda update -n base -c defaults conda'
    subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    s = view_env()
    cmd = f"conda env create -f {yml}"
    subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    s1 = view_env()
    if len(s1)>len(s):
        return 'Virtual environment successfully created.'
    return 'Virtual environment failed created.'

def env_remove(name):
    """Remove virtual environment.
    
    Args:
        name: virtual environment.
    Return:
        log info.
    """
    s = view_env()
    if name not in s:
        return 'Virtual environment not exists.'
    cmd = f'conda remove -n {name} --all'
    subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    s = view_env()
    if name not in s:
        return 'Virtual environment successfully removed.'
    return 'Virtual environment failed removed.'