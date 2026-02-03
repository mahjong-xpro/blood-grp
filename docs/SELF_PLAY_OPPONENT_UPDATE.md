# 自博弈对手更新说明（整个更新）

本文说明当前自博弈训练中**对手（champion/baseline）**与**被训练方（challenger/trainee）**的更新方式：均为**整模型全量更新**，无增量或差分更新。

---

## 1. 角色与数据流

- **被训练方（trainee / challenger）**：1v3 中“1”的那一方，使用当前要训练的模型。
- **自博弈对手（champion / baseline）**：1v3 中“3”的那三方，使用同一份 baseline 模型。

在线模式下：训练器（trainer）更新模型并提交到服务器；客户端（client）从服务器拉取参数、用拉到的模型做 1v3 自对局，对局中己方是 trainee、对方三人是 baseline。

---

## 2. 服务端：参数是整份替换

- 服务器只保存**一份**当前全局参数：`mortal_param`、`dqn_param`、`param_version`。
- 训练器调用 `submit_param(mortal, dqn)` 时：
  - 发送的是**完整** `mortal.state_dict()` 和 `dqn.state_dict()`。
  - 服务端在 `handle_submit_param` 里**整份覆盖**：
    - `S.mortal_param = msg['mortal']`
    - `S.dqn_param = msg['dqn']`
    - `S.param_version += 1`
- 没有差分、增量或合并逻辑；每次提交都是**整个更新**。

代码位置：`mortal/server.py` 中 `handle_submit_param`。

---

## 3. 客户端：trainee 从服务器整模型拉取

- 客户端每轮自对局前会 `get_param`，收到服务器返回的 `mortal`、`dqn`、`param_version`。
- 本地用**全量**覆盖当前模型：
  - `mortal.load_state_dict(rsp['mortal'])`
  - `dqn.load_state_dict(rsp['dqn'])`
- 没有“只更新部分参数”或“在旧参数上打补丁”；每次拉取后 trainee 都是**整模型更新**。

代码位置：`mortal/client.py` 中 `get_param` 成功后对 `mortal`/`dqn` 的 `load_state_dict`。

---

## 4. 自博弈对手（baseline）：从文件整模型加载

- 1v3 中的三个对手统一使用 **baseline 引擎**（`TrainPlayer.baseline_engine`）。
- baseline 在 **TrainPlayer 初始化时** 从配置中的 `baseline.train.state_file`（如 `/data/mortal/baseline.pth`）**整模型加载**一次：
  - `stable_mortal.load_state_dict(state['mortal'])`
  - `stable_dqn.load_state_dict(state['current_dqn'])`
- 之后在该客户端进程内，这三个对手**不再**从服务器更新；若要让对手更新，需要：
  - 在别处（如训练器或运维脚本）把最新/最佳模型**整份**写入 baseline 文件（例如 `cp mortal.pth baseline.pth`），然后
  - 新起客户端进程（或重新创建 TrainPlayer），才会重新从文件**整模型加载**新 baseline。

因此，自博弈对手的更新方式也是**整个更新**：每次更新都是整份替换 baseline 文件中的模型，加载时整份 `load_state_dict`，无增量。

代码位置：`mortal/player.py` 中 `TrainPlayer.__init__` 对 `baseline_file` 的加载。

---

## 5. 小结

| 角色           | 更新来源       | 更新方式                         |
|----------------|----------------|----------------------------------|
| 服务端全局参数 | 训练器 submit  | 整份覆盖 `mortal_param`/`dqn_param` |
| Trainee        | 服务器 get_param | 整份 `load_state_dict`          |
| 自博弈对手     | baseline 文件  | 进程启动时整份 `load_state_dict`；文件更新需整份替换 |

当前实现中**没有**：  
- 参数差分/增量同步  
- 只更新部分层或部分参数  
- 对手在单次客户端会话内从服务器实时更新  

如需让自对局对手跟踪最新模型，需通过“整份更新 baseline 文件 + 新进程/新 TrainPlayer”的方式，仍属于**整个更新**。

---

## 6. 更新对手：是否需要停训、复制、重启？

**结论：训练不用停；复制可随时做；要让对手生效，必须重启客户端。**

| 步骤 | 是否需要 | 说明 |
|------|----------|------|
| **停止训练** | **不需要** | 训练器（trainer）在在线模式下不做 1v3，不读 baseline。复制 `mortal.pth` / `best_state_file` 到 baseline 文件时，训练可照常运行。 |
| **复制到 baseline** | **需要** | 把当前或最佳模型整份写到 `baseline.train.state_file`（如 `/data/mortal/baseline.pth`），例如：`cp /data/mortal/mortal.pth /data/mortal/baseline.pth` 或复制 `best_state_file`。可在训练运行时执行（如定时任务或每次 save 后）。 |
| **重启** | **仅客户端需要** | 每个 client 在启动时只创建一次 `TrainPlayer()`，baseline 只在此时从文件加载并常驻内存。不重启 client，已运行的 client 会一直用**旧** baseline。要让自对局对手变成新模型，必须**重启所有做自对局的 client 进程**。训练器和服务器不必重启。 |

**推荐流程（不中断训练）：**

1. 训练照常跑，按需在合适时机把模型复制到 baseline，例如：
   - 每次 `save_every` 存盘后：`cp /data/mortal/mortal.pth /data/mortal/baseline.pth`
   - 或仅在打破最佳记录时：`cp /data/mortal/best.pth /data/mortal/baseline.pth`
2. 当希望自对局对手更新到这份新 baseline 时，**只重启 client 进程**（如 `blood-client.sh` 启动的那些），训练器和 server 保持运行。
