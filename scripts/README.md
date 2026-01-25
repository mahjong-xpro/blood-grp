# Bloody Battle Mahjong Server Scripts

这个目录包含了用于管理 Bloody Battle Mahjong 训练服务器的脚本。

## 文件说明

### `blood-server.sh`
主要的服务器管理脚本，用于启动、停止、重启和查看服务器状态。

**使用方法：**
```bash
./scripts/blood-server.sh {start|stop|restart|status|logs}
```

**命令说明：**
- `start` - 启动训练服务器
- `stop` - 停止训练服务器
- `restart` - 重启训练服务器
- `status` - 显示服务器状态和进程信息
- `logs` - 实时查看服务器日志

**环境变量：**
- `MORTAL_CFG` - 配置文件路径（默认：`mortal/config.toml`）
- `PYTHON_CMD` - Python 命令（默认：`python3`）

**示例：**
```bash
# 启动服务器
./scripts/blood-server.sh start

# 查看状态
./scripts/blood-server.sh status

# 查看日志
./scripts/blood-server.sh logs

# 停止服务器
./scripts/blood-server.sh stop
```

### `blood-mortal.sh`
用于启动 Mortal AI 玩家的脚本。

**使用方法：**
```bash
./scripts/blood-mortal.sh <player_id>
```

**参数：**
- `player_id` - 玩家ID，必须是 0、1、2 或 3

**示例：**
```bash
# 启动玩家 0
./scripts/blood-mortal.sh 0
```


## 配置要求

在使用这些脚本之前，请确保：

1. **配置文件存在**：创建 `mortal/config.toml` 或设置 `MORTAL_CFG` 环境变量指向配置文件

2. **Python 环境**：确保 Python 3 已安装，并且所有依赖已安装

3. **libblood 模块**：确保 `libblood` Python 模块可以正常导入

4. **端口配置**：在配置文件中设置正确的服务器端口（默认：5000）

## 日志

- 脚本日志：`/tmp/blood-server.log`
- Systemd 日志：使用 `journalctl -u blood-server` 查看

## 故障排除

### 服务器无法启动
1. 检查配置文件是否存在且格式正确
2. 检查 Python 环境和依赖
3. 查看日志文件：`/tmp/blood-server.log`

### 端口被占用
1. 检查端口是否被其他进程占用：
   ```bash
   lsof -i :5000
   ```
2. 修改配置文件中的端口号

### 权限问题
确保脚本有执行权限：
```bash
chmod +x scripts/*.sh
```

## 训练相关

### 开始训练

**没有人类对战数据？** 系统支持自博弈训练（从零开始）！

- **快速开始**：`scripts/QUICK_START_SELF_PLAY.md` ⭐
- **详细指南**：`scripts/SELF_PLAY_GUIDE.md`（自博弈）
- **通用指南**：`scripts/TRAINING_GUIDE.md`（所有训练模式）
- **完整训练流程**：`scripts/COMPLETE_TRAINING_GUIDE.md` 📘 **从零开始的完整训练流程（推荐）**
- **GRP 训练**：`scripts/GRP_TRAINING_GUIDE.md` ⚠️ **重要：GRP 需要先训练**
- **GRP 原理**：`scripts/GRP_PRINCIPLE.md` 📖 **了解 GRP 的技术原理和作用**
- **GRP 分析**：`scripts/GRP_ANALYSIS.md` 🔍 **深度分析 GRP 实现的有效性**
- **GRP 优化**：`scripts/GRP_FURTHER_OPTIMIZATIONS.md` 🚀 **进一步优化建议和方向**
- **无数据训练**：`scripts/NO_DATA_TRAINING.md` 🎮 **没有数据如何开始训练（自博弈）**

**快速开始（离线训练）：**
```bash
# 1. 准备配置文件
cd mortal
cp config.example.toml config.toml
# 编辑 config.toml 设置数据路径等

# 2. 开始训练
./scripts/blood-train.sh offline
```

**在线训练：**
```bash
# 1. 启动训练服务器
./scripts/blood-server.sh start

# 2. 启动训练客户端（多个终端）
cd mortal
python client.py

# 3. 启动训练主程序
./scripts/blood-train.sh online
```

## 注意事项

- 服务器默认监听 `127.0.0.1:5000`（可在配置文件中修改）
- 确保防火墙允许相应端口的连接
- 定期检查日志文件大小，避免占用过多磁盘空间
- 服务器以后台进程运行，PID 保存在 `/tmp/blood-server.pid`

## 相关文档

- `scripts/TRAINING_GUIDE.md` - 详细训练指南 ⭐
- `scripts/MORTAL_EXPLANATION.md` - Mortal AI 玩家说明
- `mortal/config.example.toml` - 完整配置示例
