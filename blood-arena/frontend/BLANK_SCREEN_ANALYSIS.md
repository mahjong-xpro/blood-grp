# 界面全黑/无显示 — 深度分析

## 可能原因

### 1. 后端静态路径依赖 CWD（已修复）
- `FileResponse("frontend/index.html")` 和 `StaticFiles(directory="frontend/...")` 使用**相对路径**，相对于**进程启动时的当前工作目录**。
- 若从 `blood`（仓库根）或其它目录启动 uvicorn，`frontend/` 会指向错误位置，导致 404 或返回错误/空文件。
- **修复**：在 `main.py` 中用 `arena_root = os.path.dirname(current_dir)` 得到 blood-arena 目录，再用 `os.path.join(arena_root, "frontend", ...)` 提供 index.html 和静态目录，与 CWD 无关。

### 2. v-cloak 在 Vue 未挂载时隐藏整页（已修复）
- 此前有 `[v-cloak] { display: none }`，Vue 挂载前 `#app` 被隐藏。
- 若 `/js/app.js` 加载失败或运行报错，Vue 永不挂载，页面会一直空白。
- **修复**：已移除 v-cloak 及相关样式。

### 3. Vue 挂载前无任何占位
- 若脚本加载慢或报错，用户会长时间只看到黑屏，无法区分“加载中”还是“出错”。
- **修复**：增加 `mounted` 状态与“加载中”占位：首屏显示“加载中…”，`onMounted` 后再渲染完整界面；并为 `.load-fallback` 设置可见样式。

### 4. 脚本报错导致 Vue 未挂载
- 若 `app.js` 中有运行时错误（如未定义变量、模板中访问不存在的属性），Vue 可能在 mount 阶段抛错，导致界面不渲染。
- **建议**：打开开发者工具 (F12) → Console，查看是否有红色报错；根据报错修正代码。

### 5. 静态资源 404
- 若 `/js/app.js` 或 `/css/style.css` 返回 404，样式或脚本不加载，可能只看到空白或未样式化内容。
- **建议**：在 F12 → Network 中确认 `app.js`、`style.css` 是否 200，以及是否从正确 host 加载。

## 已做修改汇总

1. **backend/main.py**：用基于 `__file__` 的 `arena_root` 计算前端目录，用绝对路径提供 `index.html` 和 `/static`、`/js`、`/css`。
2. **index.html**：增加 `v-if="!mounted"` 的“加载中”占位，其余内容包在 `v-else` 的 `<template>` 中。
3. **app.js**：引入 `ref`，增加 `mounted = ref(false)`，在 `onMounted` 中设 `mounted.value = true`，并暴露 `mounted`。
4. **style.css**：增加 `.load-fallback` 样式，保证占位文字可见。

## 自检步骤

1. 从 **blood-arena** 目录启动后端，例如：`cd blood-arena && uvicorn backend.main:app --reload`。
2. 浏览器访问根路径，先看是否出现“加载中…”；若出现，再等 1～2 秒看是否变为正常界面。
3. 若始终只有“加载中”或仍全黑：打开 F12 → Console 看报错，Network 看 `/js/app.js`、`/css/style.css` 是否 200。
