"""
libblood 模块加载器

跨平台支持：在 macOS 上，pyo3 生成 .dylib 文件；在 Linux 上，生成 .so 文件。
需要使用特殊方式加载这些扩展模块。
"""
import sys
import os
import importlib.machinery
import importlib.util
import platform

def load_libblood():
    """加载 libblood 模块"""
    # 如果已经加载，直接返回
    if 'libblood' in sys.modules:
        return sys.modules['libblood']
    
    # 查找 libblood 模块文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 根据平台确定文件扩展名
    system = platform.system()
    if system == 'Darwin':  # macOS
        extensions = ['.dylib']
        search_paths = [
            os.path.join(project_root, 'target', 'release', 'deps'),
            os.path.join(project_root, 'target', 'debug', 'deps'),
            os.path.join(project_root, 'target', 'release'),
            os.path.join(project_root, 'target', 'debug'),
        ]
    elif system == 'Linux':  # Linux
        extensions = ['.so']
        search_paths = [
            os.path.join(project_root, 'target', 'release'),
            os.path.join(project_root, 'target', 'debug'),
            os.path.join(project_root, 'target', 'release', 'deps'),
            os.path.join(project_root, 'target', 'debug', 'deps'),
        ]
    else:  # Windows 或其他
        extensions = ['.dll', '.pyd']
        search_paths = [
            os.path.join(project_root, 'target', 'release'),
            os.path.join(project_root, 'target', 'debug'),
        ]
    
    # 查找 libblood 模块文件
    for search_path in search_paths:
        for ext in extensions:
            # 尝试不同的文件名格式
            possible_names = [
                f'libblood{ext}',
                f'blood{ext}',
            ]
            for name in possible_names:
                libblood_path = os.path.join(search_path, name)
                if os.path.exists(libblood_path):
                    # 使用 ExtensionFileLoader 加载扩展模块
                    loader = importlib.machinery.ExtensionFileLoader('libblood', libblood_path)
                    spec = importlib.machinery.ModuleSpec('libblood', loader, origin=libblood_path)
                    libblood = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(libblood)
                    sys.modules['libblood'] = libblood
                    return libblood
    
    # 如果找不到，尝试直接导入（可能已经安装为 Python 包）
    try:
        import libblood
        sys.modules['libblood'] = libblood
        return libblood
    except ImportError:
        pass
    
    raise ImportError(
        f"libblood module not found. Please build libblood first:\n"
        f"  cargo build --lib --release\n"
        f"Searched in: {', '.join(search_paths)}\n"
        f"Looking for extensions: {', '.join(extensions)}"
    )

# 自动加载
load_libblood()
