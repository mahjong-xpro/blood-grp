# 前端重构说明（解决界面不显示）

## 问题原因简要

1. **in-DOM 模板**：Vue 挂载到 `#app` 时，用 `#app` 的 innerHTML 作为模板编译。易出现「模板里用到的属性在实例上未定义」等解析/生命周期命名冲突（如 `mounted`）。
2. **依赖外链 CSS**：若 `/css/style.css` 未加载，仅靠 body 背景时，内容可能不可见或布局错乱。
3. **无兜底内容**：脚本报错或未加载时，`#app` 为空，用户只看到黑屏。

## 重构内容

### 1. index.html

- **仅保留**：`<div id="app"></div>` 作为挂载点，**不再在 HTML 里写任何 Vue 模板**。
- **内联关键样式**：在 `<style>` 中写 body / #app 的基础样式（背景、文字色、最小高度、flex），保证即使 `/css/style.css` 未加载也能看到页面骨架。
- **#app 为空时的兜底**：用 `#app:empty::before { content: '加载中…'; ... }`，在 Vue 未渲染前至少显示「加载中…」。

### 2. app.js

- **字符串模板**：用 `template: APP_TEMPLATE`，在 JS 里定义完整模板字符串（`APP_TEMPLATE`），**不再使用 in-DOM 模板**，避免：
  - 与 Vue 生命周期/保留属性命名冲突；
  - 服务端或代理修改 HTML 导致的解析问题。
- **移除 appReady**：不再需要「先显示加载再切主界面」的分支，逻辑更简单。
- **WebSocket 拼接**：用字符串拼接代替模板字符串，避免在 createApp 配置里出现 `${...}` 被误解析。

### 3. style.css

- **#app**：改为 `min-height: 100vh`，保证空状态时也有高度。
- **.app-root**：为 Vue 根节点增加与 #app 一致的 flex 布局，保证整页布局正确。
- **删除 .load-fallback**：已无此节点。

## 自检

1. 从 **blood-arena** 目录启动：`uvicorn backend.main:app --reload`。
2. 打开页面：应先看到「加载中…」或直接看到「血战到底」顶栏与主内容。
3. 若仍无内容：F12 → Console 看报错；Network 确认 `/js/app.js`、`/css/style.css` 为 200。
