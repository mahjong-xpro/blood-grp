"""
libblood 模块加载器

在 macOS 上，pyo3 生成的模块是 .dylib 文件，需要使用特殊方式加载。
"""
import sys
import os
import importlib.machinery
import importlib.util

def load_libblood():
    """加载 libblood 模块"""
    # 如果已经加载，直接返回
    if 'libblood' in sys.modules:
        return sys.modules['libblood']
    
    # 查找 libblood.dylib 文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    libblood_dylib_paths = [
        os.path.join(project_root, 'target', 'release', 'libblood.dylib'),
        os.path.join(project_root, 'target', 'debug', 'libblood.dylib'),
    ]
    
    for dylib_path in libblood_dylib_paths:
        if os.path.exists(dylib_path):
            # 使用 ExtensionFileLoader 加载 .dylib 文件
            loader = importlib.machinery.ExtensionFileLoader('libblood', dylib_path)
            spec = importlib.machinery.ModuleSpec('libblood', loader, origin=dylib_path)
            libblood = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(libblood)
            sys.modules['libblood'] = libblood
            return libblood
    
    raise ImportError("libblood.dylib not found. Please build libblood first: cargo build --lib --release")

# 自动加载
load_libblood()
