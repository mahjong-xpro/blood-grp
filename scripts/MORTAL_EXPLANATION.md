# blood-mortal.sh 业务逻辑详解

## 概述

`blood-mortal.sh` 是一个用于启动 **Mortal AI 玩家** 的脚本。它启动一个 AI 玩家实例，该玩家可以通过 **mjai 协议** 与其他玩家（人类或其他 AI）进行血战到底麻将对局。

## 核心概念

### 1. mjai 协议
mjai 是一个用于麻将 AI 对局的 JSON 协议，类似于国际象棋的 UCI 协议。它通过标准输入/输出（stdin/stdout）进行通信：
- **输入**：从 stdin 读取 JSON 格式的游戏事件（如摸牌、打牌、和牌等）
- **输出**：向 stdout 输出 JSON 格式的动作响应（如打哪张牌、是否和牌等）

### 2. Player ID（玩家ID）
- 血战到底麻将需要 4 个玩家
- Player ID 范围：0, 1, 2, 3
- 每个玩家实例需要知道自己的 ID，以便正确理解游戏状态

## 工作流程

### 启动流程

```
blood-mortal.sh <player_id>
    ↓
mortal.py <player_id>
    ↓
1. 加载 AI 模型（Brain + DQN）
2. 创建 MortalEngine（AI 决策引擎）
3. 创建 Bot（mjai 协议接口）
4. 进入事件循环
```

### 运行时流程

```
标准输入 (stdin)
    ↓
读取 JSON 事件（如：{"type":"tsumo","pai":"1m"}）
    ↓
Bot.react() 处理事件
    ↓
更新游戏状态 (PlayerState)
    ↓
判断是否需要响应（can_act）
    ↓
如果需要响应：
    ↓
MortalEngine 计算最佳动作
    ↓
输出 JSON 响应到标准输出 (stdout)
```

## 详细代码解析

### 1. 脚本部分 (`blood-mortal.sh`)

```bash
# 验证玩家 ID（必须是 0-3）
PLAYER_ID=$1
if ! [[ "$PLAYER_ID" =~ ^[0-3]$ ]]; then
    echo "Error: Invalid player ID"
    exit 1
fi

# 运行 mortal.py，传入玩家 ID
exec python3 mortal.py "$PLAYER_ID"
```

**作用**：简单的包装脚本，验证参数并启动 Python 程序。

### 2. Python 主程序 (`mortal.py`)

#### 2.1 初始化阶段

```python
# 1. 解析玩家 ID
player_id = int(sys.argv[-1])  # 0, 1, 2, 或 3

# 2. 加载 AI 模型
state = torch.load(config['control']['state_file'])
mortal = Brain(...).eval()  # 神经网络模型（观察 → 特征）
dqn = DQN(...).eval()        # Q 网络（特征 → 动作价值）

# 3. 创建决策引擎
engine = MortalEngine(mortal, dqn, ...)

# 4. 创建 mjai Bot
bot = Bot(engine, player_id)
```

**说明**：
- `Brain`：将游戏观察（手牌、场况等）编码为特征向量
- `DQN`：根据特征向量计算每个动作的 Q 值（预期收益）
- `MortalEngine`：封装决策逻辑，选择最佳动作
- `Bot`：mjai 协议接口，处理输入/输出

#### 2.2 事件循环

```python
for line in filtered_trimmed_lines(sys.stdin):
    # 从标准输入读取 JSON 事件
    # 例如：{"type":"tsumo","pai":"1m","actor":0}
    
    if reaction := bot.react(line):
        # 如果需要响应，输出动作
        print(reaction, flush=True)
        # 例如：{"type":"dahai","pai":"1m","actor":0}
```

**说明**：
- 持续从 stdin 读取游戏事件
- `bot.react()` 处理事件并返回响应（如果需要）
- 将响应输出到 stdout

### 3. Bot 处理逻辑 (`libblood/src/mjai/bot.rs`)

```rust
fn react(&mut self, line: &str, can_act: bool) -> Result<Option<String>> {
    // 1. 解析 JSON 事件
    let event: Event = json::from_str(line)?;
    
    // 2. 更新游戏状态
    self.state.update(&event)?;
    
    // 3. 检查是否需要响应
    if !can_act || !cans.can_act() {
        return Ok(None);  // 不需要响应
    }
    
    // 4. 获取 AI 决策
    let reaction = self.agent.get_reaction(...)?;
    
    // 5. 返回 JSON 响应
    Ok(Some(json::to_string(&reaction)?))
}
```

**说明**：
- 解析输入事件（摸牌、打牌、和牌等）
- 更新内部游戏状态
- 如果需要响应（轮到自己行动），调用 AI 决策
- 返回动作响应

## 使用场景

### 场景 1：本地测试对局

```bash
# 终端 1：启动玩家 0
./scripts/blood-mortal.sh 0

# 终端 2：启动玩家 1
./scripts/blood-mortal.sh 1

# 终端 3：启动玩家 2
./scripts/blood-mortal.sh 2

# 终端 4：启动玩家 3
./scripts/blood-mortal.sh 3

# 然后通过游戏服务器或对局程序连接这 4 个玩家
```

### 场景 2：在线对局平台

许多在线麻将平台支持 mjai 协议，可以：
1. 配置平台使用 `mortal.py` 作为 AI 玩家
2. 平台通过 stdin/stdout 与 AI 通信
3. AI 自动响应游戏事件

### 场景 3：训练数据收集

在训练模式下，AI 玩家会：
- 记录对局日志
- 提交到训练服务器（`blood-server.sh`）
- 用于改进 AI 模型

## 输入/输出示例

### 输入事件（从 stdin 读取）

```json
// 游戏开始
{"type":"start_game","kyoku_first":0,"aka_nashi":true}

// 摸牌
{"type":"tsumo","pai":"1m","actor":0}

// 其他玩家打牌
{"type":"dahai","pai":"2p","actor":1}

// 可以响应（如可以吃、碰、杠、和）
{"type":"hora","pai":"3s","actor":2,"target":1}
```

### 输出响应（向 stdout 输出）

```json
// 打牌
{"type":"dahai","pai":"1m","actor":0}

// 和牌
{"type":"hora","pai":"3s","actor":0,"target":1}

// 碰
{"type":"pon","pai":"2p","actor":0,"target":1,"consumed":["2p","2p"]}
```

## 关键配置

### 配置文件 (`mortal/config.toml`)

```toml
[control]
state_file = '/path/to/mortal.pth'  # AI 模型文件路径
version = 4                          # 模型版本
```

### 环境变量

```bash
MORTAL_CFG=/path/to/config.toml  # 配置文件路径
MORTAL_REVIEW_MODE=1             # 启用复盘模式（用于分析）
```

## 与其他组件的关系

```
┌─────────────────┐
│ blood-mortal.sh │  ← 启动脚本
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   mortal.py     │  ← Python 主程序
└────────┬────────┘
         │
         ├──→ libblood.mjai.Bot  ← mjai 协议接口
         │
         ├──→ MortalEngine       ← AI 决策引擎
         │
         └──→ Brain + DQN         ← 神经网络模型
```

## 总结

`blood-mortal.sh` 的作用是：

1. **启动一个 AI 玩家实例**
2. **通过 mjai 协议与其他玩家对局**
3. **自动响应游戏事件，做出最优决策**

它是一个**客户端程序**，需要配合：
- 游戏服务器（管理对局流程）
- 或其他玩家（人类或其他 AI）

才能进行完整的对局。
