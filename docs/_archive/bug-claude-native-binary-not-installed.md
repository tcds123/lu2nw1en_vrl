# Bug: claude CLI "native binary not installed" + find 命令被污染

**日期**: 2026-08-20
**状态**: ✅ 已修复并验证(三层锁 + 装同步版本),已固化修复脚本到持久层
**环境**: AIOS notebook 容器 (root@aimp-notebook-zhuldz), Linux x86_64
**同源记录**: `docs/phase2_issues.md` 中 2026-08-20 issue 条目(本文件是独立整理版,方便下次修理)

---

## 1. 症状

```bash
root@...:/data# claude
Error: claude native binary not installed. Either postinstall did not run
(--ignore-scripts, some pnpm configs) or the platform-native optional dependency
was not downloaded (--omit=optional).
Run the postinstall manually (adjust path for local vs global install):
  node node_modules/@anthropic-ai/claude-code/install.cjs
Or reinstall without --ignore-scripts / --omit=optional.
```

附带现象:`find` / `tar` / `grep` 等命令的输出里**也混进了这段报错文字**,干扰诊断(很多 `npm view` / `tar` / `file` 输出看着像失败,其实是被污染了)。

---

## 2. 根因(三层)

### 直接层:claude.exe 是 500B 报错 stub
- 真二进制应在 `/usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe`
- 损坏时该文件只有 **500 字节**,是 ASCII 占位 stub(就是上面那段报错文本)
- 正常应是 **~320MB 的 ELF 64-bit x86-64 可执行文件**(md5 `069650dfb9f1...`)
- **最快的判别器:看文件大小**。500B = stub 坏,320M = 真二进制 OK

### 深层:npm 发布失配(真正的根因)
- Claude Code 主包 `@anthropic-ai/claude-code` 有 `optionalDependencies` 指向各平台原生子包(`@anthropic-ai/claude-code-linux-x64` 等)
- `bin/claude.exe` 本身只是占位符;`install.cjs`(postinstall)**只负责把下载好的原生子包二进制拷贝覆盖 stub,不负责下载**
- **问题**: npm 上主包 `@anthropic-ai/claude-code@2.1.237` 已发布,但对应的原生子包 `@anthropic-ai/claude-code-linux-x64@2.1.237` **返回 404(不存在)**,最新原生子包只到 `2.1.236`
- 后果: 装 2.1.237 时原生子包没下来,postinstall 无东西可拷,stub 原地保留 → 报错
- **这是 Anthropic 官方 npm 发布失配,不是本地环境问题**

### 反复失败层:autoupdater 反复回退(第一次修了又坏的原因)
- Claude Code **autoupdater 默认 enabled**:每次 `claude` 启动都查 npm 最新版本并 `npm install -g` 升级
- 最新 = 2.1.237 = 原生子包缺失 = stub
- 所以**手动装好的 2.1.236 真二进制,会被 autoupdater 在下次启动时打回 2.1.237 stub**
- 这就是"修了又不行"的真正机制

### find 命令污染的成因
- Claude Code 的 bash 工具注入了一个 `find` **shell function**(覆盖 `/usr/bin/find`),用 `CLAUDE_CODE_EXECPATH` 把 claude 二进制当 `bfs` find 后端跑
- 二进制是 stub 时,每次 `find` 调用都打印报错文本 → 污染输出
- 注:该 function **不在 `.bashrc` 里**(是父进程注入的),一旦真二进制就位会自动消失
- 诊断期可用 `command find` / `command grep` 绕过

---

## 3. 修复(三层锁 + 装同步版本)

### 一键脚本(已固化到持久层)
```bash
bash /data/zhuldz/lunwen/scripts/fix_claude_install.sh
```

### 手动步骤(脚本做了什么)
1. **装同步存在的版本**: `npm install -g @anthropic-ai/claude-code@2.1.236`(主包 + linux-x64 原生子包都存在于 2.1.236,postinstall 能正常拷贝真二进制)
2. **锁 autoupdater —— `/root/.claude.json`**:
   ```json
   "autoUpdates": false,
   "autoUpdaterStatus": "disabled"
   ```
3. **锁 autoupdater —— `/root/.claude/settings.json`**: `"autoUpdates": false`
4. **锁 autoupdater —— `/root/.bashrc`**(环境级):
   ```bash
   export DISABLE_AUTOUPDATER=1
   ```

三层冗余是因为 autoupdater 有多个读取入口,任一未关都可能触发升级回退。

### 验证
- `claude.exe` 大小 ~320M 且 `claude --version` 正常输出 → OK
- 多次启动 claude 后 size+md5 不变 → autoupdater 已锁死,不再回退

---

## 4. 持久性 & 容器重建

修复的三个产物**全在 Docker overlay(根 `/`,容器可写层)**:
| 产物 | 位置 | 文件系统 |
|---|---|---|
| `claude.exe` 真二进制 | `/usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/` | overlay `/` |
| `.claude.json` 锁 | `/root/` | overlay `/` |
| `settings.json` 锁 | `/root/.claude/` | overlay `/` |
| `.bashrc` DISABLE_AUTOUPDATER | `/root/` | overlay `/` |

- **容器重启**(进程重启,overlay 保留):修复保留。`startup.sh`(PID1)**不重装/不重置 claude**,且 `.bashrc` 的 aios_profile 段是幂等追加(只在无 `addProfileByAiosEnd` 标记时追加,不覆盖我的 DISABLE_AUTOUPDATER)→ **重启无需重跑脚本**
- **容器/镜像重建 / 换新实例**:overlay 上层清空 → **三个产物全丢**,claude 回到镜像里坏 stub → **需重跑脚本**

所以修复脚本本身必须放在**持久层** `/data/zhuldz`(JuiceFS,跨重建保留),而不是 overlay。脚本已固化在 `/data/zhuldz/lunwen/scripts/fix_claude_install.sh`。

---

## 5. 经验教训

1. **文件大小是最快的 stub 判别器**(500B vs 320M),不要被错误文本误导去查 install.cjs
2. **第一次修了又坏 = 有自动回退机制**,修任何"反复失败"的 bug 都要先排除 autoupdater / 守护进程 / 镜像重置这类回写源
3. **npm optionalDependencies 发布失配**:主包发了但平台原生子包没同步发,是 Anthropic 官方问题,本地无解,只能降到同步存在的版本
4. **`find` 被污染时用 `command find` 绕过**,或直接 curl npm registry HTTP API 查版本是否存在(避免被 find function 坑)
5. **容器环境的修复要落持久层**:overlay 上的修复在镜像重建时会丢,脚本/配置必须放 JuiceFS(`/data/zhuldz`)

---

## 6. 相关文件

- 修复脚本: `/data/zhuldz/lunwen/scripts/fix_claude_install.sh`
- 同源 issue 条目: `docs/phase2_issues.md`(2026-08-20)
- 锁配置: `/root/.claude.json`, `/root/.claude/settings.json`, `/root/.bashrc`
- 容器启动脚本(确认不重置 claude): `/root/.init/startup.sh`
