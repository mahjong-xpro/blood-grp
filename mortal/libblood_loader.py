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
    # 如果已经加载，直接返回（兼容两种模块名：blood / libblood）
    if 'blood' in sys.modules:
        sys.modules.setdefault('libblood', sys.modules['blood'])
        return sys.modules['blood']
    if 'libblood' in sys.modules:
        sys.modules.setdefault('blood', sys.modules['libblood'])
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
    
    # 查找扩展模块文件
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
                    # IMPORTANT:
                    # The actual pyo3 module name is `blood` (exports PyInit_blood).
                    # Some parts of this repo import `libblood`, so we load as `blood`
                    # and then alias it to `libblood` for compatibility.
                    loader = importlib.machinery.ExtensionFileLoader('blood', libblood_path)
                    spec = importlib.machinery.ModuleSpec('blood', loader, origin=libblood_path)
                    blood = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(blood)
                    sys.modules['blood'] = blood
                    sys.modules['libblood'] = blood
                    return blood
    
    # 如果找不到，尝试直接导入（可能已经安装为 Python 包）
    try:
        import blood
        sys.modules['blood'] = blood
        sys.modules['libblood'] = blood
        return blood
    except ImportError:
        try:
            import libblood
            sys.modules['libblood'] = libblood
            sys.modules.setdefault('blood', libblood)
            return libblood
        except ImportError:
            pass
    
    raise ImportError(
        f"libblood/blood module not found. Please build it first:\n"
        f"  cargo build -p libblood --release --lib\n"
        f"Searched in: {', '.join(search_paths)}\n"
        f"Looking for extensions: {', '.join(extensions)}"
    )

# 自动加载
load_libblood()
