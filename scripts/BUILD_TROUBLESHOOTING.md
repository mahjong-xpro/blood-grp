# 编译问题排查指南

## macOS 链接错误

### 问题描述

在 macOS 上运行 `cargo build` 时，可能出现以下错误：

```
error: linking with `cc` failed: exit status: 1
ld: symbol(s) not found for architecture x86_64
```

### 原因

这是 macOS 上构建包含 pyo3 的二进制文件时的常见问题。二进制文件（如 `stat`、`validate_logs`）需要链接 Python 库，但在 macOS 上可能没有正确配置。

### 解决方案

#### 方案 1：只构建库（推荐）

如果只需要 Python 模块，可以只构建库：

```bash
cargo build --lib
# 或
cargo build --lib --release
```

**优点**：
- 避免链接问题
- 构建更快
- 对于 Python 训练来说已经足够

#### 方案 2：配置 Python 环境变量

设置 Python 库路径：

```bash
# 获取 Python 库路径
PYTHON_LIB_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))")

# 设置环境变量
export LIBRARY_PATH="$PYTHON_LIB_DIR:$LIBRARY_PATH"
export C_INCLUDE_PATH="$PYTHON_INCLUDE_DIR:$C_INCLUDE_PATH"

# 然后构建
cargo build
```

#### 方案 3：使用 conda 环境

如果使用 conda：

```bash
# 激活 conda 环境
conda activate your_env

# 设置环境变量
export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"
export C_INCLUDE_PATH="$CONDA_PREFIX/include/python3.x:$C_INCLUDE_PATH"

# 构建
cargo build
```

#### 方案 4：跳过二进制文件构建

如果不需要二进制文件，可以在 `Cargo.toml` 中注释掉：

```toml
# [[bin]]
# name = "stat"
# path = "src/bin/stat.rs"
```

### 验证

构建成功后，验证 Python 模块：

```bash
cd mortal
python3 -c "import libblood; print('libblood available')"
```

### 训练时的影响

**重要**：对于训练来说，只需要库（`libblood`），不需要二进制文件。因此：

- ✅ `cargo build --lib` 成功即可
- ❌ `cargo build` 失败不影响训练（如果只是二进制文件链接问题）

### 当前状态

根据检查：
- ✅ `cargo build --lib` 成功（只有警告）
- ❌ `cargo build` 失败（二进制文件链接问题）

**结论**：可以正常进行训练，因为只需要库。

---

## 其他常见问题

### 问题：找不到 Python 头文件

**错误**：
```
error: failed to run custom build command for `pyo3-build-config`
```

**解决**：
```bash
# 安装 Python 开发头文件
# macOS (Homebrew)
brew install python3

# 或设置环境变量
export C_INCLUDE_PATH="/path/to/python/include:$C_INCLUDE_PATH"
```

### 问题：找不到 Python 库

**错误**：
```
ld: library not found for -lpython3.x
```

**解决**：
```bash
# 设置库路径
export LIBRARY_PATH="/path/to/python/lib:$LIBRARY_PATH"
```

### 问题：架构不匹配

**错误**：
```
ld: warning: object file was built for newer 'macOS' version than being linked
```

**解决**：
- 更新 Xcode Command Line Tools
- 或设置最低 macOS 版本：
  ```bash
  export MACOSX_DEPLOYMENT_TARGET=10.15
  cargo build
  ```

---

## 快速检查

```bash
# 1. 检查库是否可以构建
cargo build --lib

# 2. 检查 Python 模块是否可用
cd mortal
python3 -c "import libblood; print('OK')"

# 3. 如果库构建成功，可以忽略二进制文件的链接错误
```

---

## 总结

对于训练来说：
- ✅ **只需要库**：`cargo build --lib` 成功即可
- ❌ **不需要二进制文件**：`stat` 和 `validate_logs` 是工具，不是训练必需的

如果 `cargo build --lib` 成功，就可以正常进行训练了。
