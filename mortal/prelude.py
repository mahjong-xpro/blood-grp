import sys
import os
import logging
import warnings
import torch
import numpy as np

sys.stdin.reconfigure(encoding='utf-8')

# 加载 libblood 模块（macOS 上需要特殊处理）
try:
    import libblood_loader  # 这会自动加载 libblood
except ImportError:
    # 如果 libblood_loader 不存在，尝试直接导入（Linux/其他平台）
    pass

logging.basicConfig(
    stream = sys.stderr,
    level = logging.INFO,
    format = '%(asctime)s %(levelname)8s %(filename)12s:%(lineno)-4s %(message)s',
)

warnings.simplefilter('ignore')

# "The given NumPy array is not writeable"
dummy = np.array([])
dummy.setflags(write=False)
torch.as_tensor(dummy)

# "distutils Version classes are deprecated"
import torch.utils.tensorboard

warnings.simplefilter('default')
