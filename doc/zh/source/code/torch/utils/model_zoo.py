import torch

import hashlib
import os
import re
import shutil
import sys
import tempfile

try:
    from requests.utils import urlparse
    import requests.get as urlopen
    requests_available = True
except ImportError:
    requests_available = False
    if sys.version_info[0] == 2:
        from urlparse import urlparse  # noqa f811
        from urllib2 import urlopen  # noqa f811
    else:
        from urllib.request import urlopen
        from urllib.parse import urlparse
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # defined below

# matches bfd8deac from resnet18-bfd8deac.pth
HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')


def load_url(url, model_dir=None, map_location=None):
    r"""从给定的 URL 处加载 Torch 序列化对象.
    如果该对象已经存在于 `model_dir` 中, 则将被反序列化并返回. 
    URL 的文件名部分应该遵循命名约定
    ``filename-<sha256>.ext`` 其中 ``<sha256>`` 是文件内容的 SHA256 哈希的前八位或更多位数.
    哈希用于确保唯一的名称并验证文件的内容.

    `model_dir` 的默认值为 ``$TORCH_HOME/models`` 其中
    ``$TORCH_HOME`` 默认值为 ``~/.torch``. 可以使用
    ``$TORCH_MODEL_ZOO`` 环境变量来覆盖默认目录.

    Args:
        url (string): 需要下载对象的 URL
        model_dir (string, optional): 保存对象的目录
        map_location (optional): 一个函数或者一个字典,指定如何重新映射存储位置 (详情查阅 torch.load)

    Example:
        >>> state_dict = torch.utils.model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    """
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = HASH_REGEX.search(filename).group(1)
        _download_url_to_file(url, cached_file, hash_prefix)
    return torch.load(cached_file, map_location=map_location)


def _download_url_to_file(url, dst, hash_prefix):
    u = urlopen(url)
    if requests_available:
        file_size = int(u.headers["Content-Length"])
        u = u.raw
    else:
        meta = u.info()
        if hasattr(meta, 'getheaders'):
            file_size = int(meta.getheaders("Content-Length")[0])
        else:
            file_size = int(meta.get_all("Content-Length")[0])

    f = tempfile.NamedTemporaryFile(delete=False)
    try:
        sha256 = hashlib.sha256()
        with tqdm(total=file_size) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        digest = sha256.hexdigest()
        if digest[:len(hash_prefix)] != hash_prefix:
            raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                               .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


if tqdm is None:
    # fake tqdm if it's not installed
    class tqdm(object):

        def __init__(self, total):
            self.total = total
            self.n = 0

        def update(self, n):
            self.n += n
            sys.stderr.write("\r{0:.1f}%".format(100 * self.n / float(self.total)))
            sys.stderr.flush()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stderr.write('\n')
