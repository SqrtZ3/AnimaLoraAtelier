# 在 scnet 上打包训练镜像 —— 规范与流程

> 适用平台：国家超算互联网 scnet（<https://www.scnet.cn>）容器服务 / Notebook。
> 平台规则原文：[构建镜像规则](https://www.scnet.cn/help/docs/mainsite/container-service/image-management/build-rules/)
>
> 证据等级同 `hygon-dcu.md`：**【实测】**=在本项目机器上跑过；**【资料】**=平台文档口径；
> **【推断】**=外推。本文除标注外均为【实测】，来自 2026-08-18 的三次真实构建
> （一次失败、一次"假成功"、v0.1.1 通过）。

---

## 0. 30 秒版

```bash
# 1) 把构建上下文摆到平台要求的位置（幂等，可反复跑）
cd /root/private_data/anima-lora-train
bash AnimaLoraToolkit/tools/scnet_stage_dockerfile.sh            # 只带代码
bash AnimaLoraToolkit/tools/scnet_stage_dockerfile.sh --weights  # 连权重一起

# 2) 控制台 → 镜像管理 → 构建镜像 → Dockerfile
#    「Dockerfile文件路径」填 /public/home/${username}/dockerFileTemp/Dockerfile

# 3) 用新镜像建实例，进去验收（★ 不做这步等于没验）
. /opt/dtk/env.sh && python /opt/anima-lora-train/AnimaLoraToolkit/tools/dcu_image_selfcheck.py --runtime
```

当前可用镜像：`image.ac.com:5000/dcu/sqrtz/base/anima-lora-dcu:v0.1.1`（运行期 7/7 通过）。

---

## 1. 先搞清楚"什么进镜像"

| 路径 | 是什么 | 随镜像走？ | 关机/重启后 |
|---|---|---|---|
| `/`（含 `/usr/local/lib/python3.11/site-packages`、`/opt`、`/etc`） | 系统盘 | ✅ | **回退到基础镜像**（除非"保存开发环境"） |
| `/root/private_data`（= `/public/home/${username}`） | 平台单独分配的持久盘（可调容量） | ❌ | **独立保留，不会被清除** |
| `/root/public_data`、`/root/group_data` | 平台共享目录，只读 | ❌ | 保留 |

由此推出三条铁律：

1. **依赖装进系统 python，绝不建 venv。** venv 建在持久盘上→不进镜像；建在 `/` 上→
   重启就没。而且 venv 不带 `--system-site-packages` 会把镜像的 das torch 整个隔离掉
   （见 §5.1）。
2. **代码要进镜像就得复制到 `/`**（本项目放 `/opt/anima-lora-train`）。
3. **权重默认不进镜像**：平台共享目录里每个节点都能读，塞进去只让每次拉取多传几个 GB。
   要做完全自包含的参赛产物时才 `--weights`。

**体积限制是 docker 单层 ≤ 15GB，不是镜像总大小。** 所以按「依赖 / 代码 / 权重」拆成
不同的 `RUN`/`COPY` 层，就不必为省空间牺牲自包含性。参考量级：基础镜像 62 层、
压缩后 6.54GB、最大单层 3.57GB；成品 v0.1.1 未压缩 22.95GB。

---

## 2. 平台的硬性要求（缺了实例起不来）

【资料，已逐项在机器上核对】

| 要求 | 基础镜像的实际情况 |
|---|---|
| 预装 SSH 服务 | `openssh-server 1:8.9p1-3ubuntu0.14` ✅ |
| 预装 SUDO | `sudo 1.9.9-1ubuntu2.6` ✅ |
| Notebook 需 Jupyter，启动路径 `/opt/conda/bin/jupyter` | 存在（软链到 `/usr/local/bin/jupyter`，jupyterlab 4.0.0）✅ |
| 基础 OS 在兼容列表（CentOS 7/8、Ubuntu 14.04–22.04） | Ubuntu 22.04.5 LTS ✅ |

`FROM` 平台基础镜像即全部继承。**因此：**

* **绝不覆盖 `CMD` / `ENTRYPOINT`。** 平台按启动路径扫描 IDE，覆盖掉会导致创建实例失败。
  基础镜像是 `Entrypoint=docker-entrypoint.sh`、`Cmd=/bin/bash`。
* 别 `apt remove` 掉 sshd/sudo，别动 `/opt/conda`。
* `tools/dcu_image_selfcheck.py` 会在构建末尾检查这三样，缺任何一个 **build 失败**。

---

## 3. 基础镜像地址怎么查（控制台不展示）

公共镜像列表和镜像详情页**都不给拉取地址**，容器内部也读不到（环境变量、
`/proc/<jupyter-pid>/environ`、`.SothisAI/` 全找过）。走光源 harbor 的 registry API：

```
1) GET https://image.sourcefind.cn:5000/v2/          → 401，读 WWW-Authenticate
   realm=https://image.sourcefind.cn:5000/service/token   service=harbor-registry
2) GET {realm}?service=harbor-registry&scope=repository:<repo>:pull   → 匿名可拿 token
3) GET /v2/<repo>/tags/list      带 Authorization: Bearer <token>
```

* `image.sourcefind.cn:5000` 通；`.com` 不通；443 端口上没有 registry。
* `registry:catalog:*` 这个 scope **拿不到 token**，所以列不出全量目录，只能按已知 repo
  名查。已知：`dcu/admin/base/jupyterlab-pytorch`（71 tags）、`dcu/admin/base/pytorch`
  （91 tags）。命名规律 `<torch>-<os>-<dtk>-<py>-devel`，另有 `-shca` 变体（用途不明）。
* 查这个 API **必须先有出网代理**，见 §7。

### ★ 怎么确认查到的就是本机跑的那个

**不要看 tag 像不像。** 拉该 tag 的 manifest → config blob，把 `config.Env` 跟运行中的
实例逐项对拍。DCU 镜像里这几项有区分度：
`AMDGPU_TARGETS` / `MOFED_VERSION` / `MIOPEN_FIND_MODE` / `DTKROOT` / `HF_ENDPOINT`。
（`JUPYTER_ENABLE_LAB`、`NB_USER` 是平台运行时注入的，镜像里没有，**不能**当判据。）

本项目核实结果：

```
image.sourcefind.cn:5000/dcu/admin/base/jupyterlab-pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.11-devel
```

五项全同。其中 `AMDGPU_TARGETS=gfx906;gfx926;gfx928;gfx936;gfx938` 含 gfx936，
正好解释了 das torch 为什么能在这张卡（gfx936）上跑。

---

## 4. 构建上下文的摆放

**平台约定：`Dockerfile` 里 `COPY`/`ADD` 的源路径相对于家目录下的 `dockerFileTemp`**
（`/public/home/${username}/dockerFileTemp`，在 Notebook 里就是
`/root/private_data/dockerFileTemp` —— 同一个挂载的两个名字），**不是**构建上下文根目录。

`tools/scnet_stage_dockerfile.sh` 负责这件事：

* 把源码 tar 过去，排除 `.git` / `venv` / `output` / `Dataset` / `monitor_data` /
  `*.safetensors` / `*.log` / `__pycache__` → 约 44MB；
* 写 `.image_commit` 记下 commit；
* 复制 `Dockerfile.dcu` 为 `dockerFileTemp/Dockerfile`；
* `--weights` 时把 `anima_models/` 一并复制（约 5.6GB 单层）；
* 逐项报体积并对照 15GB 单层上限；
* **`chown` 给家目录属主** —— root 建的文件平台的文件管理/构建服务可能读不到
  （motd 里那句"可以使用 chown 命令赋权"说的就是这件事）。

---

## 5. ★ 三类会**静默**出错的坑

这一节是本文的核心。它们的共同点是：**不会让构建失败，只会让你拿到一个没验过、
或者根本坏掉的镜像。**

### 5.1 依赖：镜像里没有 torchvision

仓库原 `requirements.txt` 写着 `torchvision>=0.15.0`。镜像里**没有** torchvision，
于是 pip 从 PyPI 装它，**连带把 das 版 torch 换成 CUDA 构建**，
`torch.cuda.is_available()` 变 False，之后所有报错都指向奇怪的地方。

→ 用 `requirements-dcu.txt`（故意不列 torch 系）+ `tools/dcu_gen_constraints.py`
生成的 constraints 把 `torch`/`numpy` 钉死。钉死后任何想换它们的包会**报错**而不是
静默替换。`numpy` 必须留在 1.25.0：das torch 是按 1.x ABI 编的，未做 numpy2 对拍。

**另一个同族坑（2026-08-18 修复）**：flash_attn / triton 在 PyPI 上只有 NVIDIA CUDA 版，
但 DCU 的 SDPA flash 后端与 torch.compile 都需要它们（flash_attn 缺了，DTK 上无 mask
SDPA 直接抛异常，仓库只能关掉 flash 退回 O(S²) 的 math；triton 还是 flash_attn 的 import
硬依赖）。→ 装海光光源 DAS1.8 的预编译轮子（`Dockerfile.dcu` 层 3.5 直接从
`download.sourcefind.cn:65024/directlink/4/{flash_attn,triton}/DAS1.8/` 下载，
`torch290`/`cp311` 段与本镜像的 torch 2.9.0+das.opt1.dtk2604 / py3.11 配套），
另需 `pytest`（flash_attn import 链硬依赖）。装后 SDPA 自动走 flash：无 mask 峰值
32.57 → 0.34 GiB（S=16384），真机验证通过（详见 `dcu-attn-backend-research.md`）。

### 5.2 legacy builder：heredoc 直接失效

平台用 **docker legacy builder**（日志里会打 `DEPRECATED: The legacy builder is deprecated`）。
它把 `RUN` 的多行命令压成一行，于是：

```
Step 19/22 : RUN set -e;     python - <<'PY'
 ---> /bin/bash: line 1: warning: here-document at line 1 delimited by end-of-file (wanted `PY')
```

**校验一行都不跑，而这一步报告成功。** v0.1.0 的"双重硬门槛"就是这样整个空转的。

→ 校验必须写成仓库里的**真实脚本文件**（`tools/dcu_image_selfcheck.py`），`RUN` 只调用它。

### 5.3 构建期没有 DTK 环境 + 管道吞退出码

DTK 的环境是基础镜像的 `docker-entrypoint.sh` 在**容器启动时**设的，`RUN` **不走
entrypoint**，所以构建期裸 `import torch` 必然报
`librocm_smi64.so.2: cannot open shared object file`。

→ 每个需要 torch 的 `RUN` 先 `. /opt/dtk/env.sh`。

而且 v0.1.0 里这个错误**没有让构建失败**，因为写成了：

```dockerfile
RUN python -c "..." | tee /opt/anima-build/base_torch.txt && 下一步
```

**管道的退出码取的是 `tee` 的，永远成功**，`&&` 照走。
→ 别在 `RUN` 里用管道接收命令的成败；先写文件再 `cat`，并加 `set -eu`。

---

## 6. Dockerfile 的分层约定

见 `Dockerfile.dcu`。每层的存在理由：

| 层 | 内容 | 为什么单独一层 |
|---|---|---|
| 1 | 记录 base 的 torch/pip 基线到 `/opt/anima-build/` | "有没有被污染"的判据，必须在装任何东西**之前** |
| 2 | 生成 constraints | 只读包元数据，不 import torch |
| 3 | `pip install -c constraints -r requirements-dcu.txt` + wandb/accelerate | 改 requirements 时只有这层缓存失效 |
| 3.5 | DAS 轮子：flash_attn / triton（+ pytest） | 从 download.sourcefind.cn 下载、约 750MB，与 requirements 解耦 |
| 4 | `scnet_env.sh` → `/etc/profile.d/zz-scnet-env.sh` + 挂进 `.bashrc` | 见 §7 |
| 5 | 代码 → `/opt/anima-lora-train`（44MB） | 改代码只失效这层 |
| 6 | 权重 → `/opt/anima_models`（**默认注释掉**） | 5.6GB，opt-in |
| 7 | `dcu_image_selfcheck.py` 硬门槛 | 必须在所有安装**之后** |

另外：构建期代理写在 `ARG BUILD_PROXY` 里，**末尾用 `ENV http_proxy=` 清掉**，
不留进运行期镜像（凭据随实例变，运行期由 `scnet_env.sh` 动态读）。

---

## 7. 运行期环境：`scnet_env.sh` 为什么必须进镜像

容器缺两样东西，都不是"没适配"，是环境没接上：

1. **DTK 的 `env.sh` 不在 `/etc/profile.d` 里** → 不 source 就 `import torch` 报
   `libgalaxyhip.so.5` 找不到。
2. **出网代理只存在于 jupyter 进程的环境里**（`/proc/<pid>/environ`），sshd 由 pid 1
   拉起、不继承 → **SSH 会话里 pip / git / wandb 全部超时，看起来像"整个容器断网"**，
   而 Jupyter 终端里一切正常。诊断一行：

   ```bash
   tr '\0' '\n' < /proc/$(pgrep -f jupyter-lab | head -1)/environ | grep -i proxy
   ```

`tools/scnet_env.sh` 一次补齐两者，代理凭据**优先从正在跑的 `jupyter-lab` 进程动态读**
（重启后凭据可能变，写死会让脚本悄悄失效），读不到才用兜底值。

Dockerfile 把它装到 `/etc/profile.d/zz-scnet-env.sh`（登录 shell 自动生效）并挂进
`/root/.bashrc`（交互式非登录 shell 也生效）。**非交互的 `ssh host "命令"` 两者都不走，
要显式 source。**

---

## 8. 验收标准

`tools/dcu_image_selfcheck.py`。构建期（层 7）跑 5 项，`--runtime` 跑 7 项：

| # | 检查 | 构建期 | 运行期 |
|---|---|:--:|:--:|
| 1 | torch 仍是 HIP 构建（`version.hip` 非空且 `version.cuda` 为空） | ✅ | ✅ |
| 2 | numpy 仍是 1.x | ✅ | ✅ |
| 3 | 训练必需依赖可 import（含 flash_attn / triton，2026-08-18 起） | ✅ | ✅ |
| 4 | 平台必需组件 `sshd`/`sudo`/`/opt/conda/bin/jupyter` 齐全 | ✅ | ✅ |
| 5 | `/etc/profile.d/zz-scnet-env.sh` 已安装 | ✅ | ✅ |
| 6 | **DCU 真的可见** | ❌ | ✅ |
| 7 | **SDPA flash 后端激活**（`configure_sdpa_backends` 什么都不关；不再校验"关闭 flash 生效"） | ❌ | ✅ |

**6/7 构建期验不了** —— 构建节点上没有 DCU（日志里 `get hyhal driver version error!`）。
第 7 项语义（2026-08-18 更新）：v0.1.1 时期它是"无 mask SDPA 能跑 + flash 被关闭"
（flash_attn 轮子缺失时的修复证据）；镜像带上 DAS flash_attn 轮子后，它升级为
"flash 后端激活"（`sdpa.flash=True` 且 `disabled_by_anima` 为空）—— 这才是这个镜像
存在的主要理由：无 mask SDPA 显存 O(S) 级（S=16384 峰值 0.34 GiB），NaViT 大 token
预算可用。**所以"构建成功"不等于"验收通过"，必须用新镜像建实例跑一次 `--runtime`。**

正常输出里这条**不是错误**：

* `Torch was not compiled with memory efficient attention`、`WARNING: /opt/hyhal/lib/cmake/rocm_smi doesn't exist` ← 基础镜像固有

---

## 9. 构建服务的行为【实测】

* **`FROM` 的镜像拉取由 docker daemon 负责，`--build-arg` 的代理对它无效。**
* 到 `image.sourcefind.cn:5000` **不稳定**：第一次 `TLS handshake timeout` 失败，
  **原样重试第二次成功**。挂在 `FROM` 就直接重试，不用改任何东西。
* 到 pypi（tuna）**直接可达**，构建不需要 `BUILD_PROXY`。
* 平台内部 registry 是 `image.ac.com:5000`，产物形如
  `image.ac.com:5000/dcu/<user>/base/<name>:<tag>`。**从 Notebook 连不上**
  （DNS/证书都不通），所以没法从实例里远程验产物。
* 日志里 `Deleting test image … Image deleted successfully` 是健康检查节点清本地副本，
  **不是删仓库里的镜像**。

### 进度卡住 / 想看详细日志

UI 的百分比读的是 `~/images_build/SothisAI/images_build/<id>.progress`：

```
path=image.ac.com:5000/dcu/sqrtz/base/anima-lora-dcu:v0.1.1
progress=100
size=22953526490
harbor_hash=sha256:ce0fabb4f18b…
jupyterlab=true
```

遇到过日志已 `制作镜像成功`、UI 还停在 40% 的情况 —— 直接 `cat` 这个文件看
`progress=` 到底多少。**完整构建日志在同目录的 `<id>.log`，比 UI 上能看到的详细得多**，
每个 Step 的真实 stdout/stderr 都在里面。v0.1.0 那两个静默失效就是靠它发现的。

---

## 10. 表单怎么填（Notebook / DCU）

| 字段 | 值 | 依据 |
|---|---|---|
| 镜像名称 | `anima-lora-dcu`（**创建后不可改**，别带版本号） | 规则：60 位小写字母/数字/点/中划线/下划线 |
| 适配加速卡 | DCU | |
| 使用区域 | 与跑训练的实例**同区** | |
| 版本名称 | `v0.1.1` 之类 | 版本走这里，不走镜像名 |
| 镜像来源 | Dockerfile | |
| Dockerfile文件路径 | `/public/home/${username}/dockerFileTemp/Dockerfile` | §4 |
| SSH服务 | 已安装 | §2 已核对 |
| 开发工具 | Jupyter → **默认路径** `/opt/conda/bin/jupyter` | §2 已核对 |
| 框架 / 版本 | PyTorch / `2.9.0` | `torch.__version__`（完整串 `2.9.0+das.opt1.dtk2604`） |
| Python版本 | `3.11` | 实测 3.11.9 |
| DTK版本 | `26.04` | `/opt/dtk -> /opt/dtk-26.04` |
| 操作系统 | Ubuntu / `22.04` | 实测 22.04.5 LTS |
| 推荐配置 | 开；型号 `BW`、最小显存 `64GB`、最小卡数 `1` | `get_device_name(0)`=`BW`，65520MB |
| 部署服务 | 关 | 训练镜像不是推理服务 |

---

## 11. 迭代与版本

* 镜像里的 `/opt/anima-lora-train` 是**随镜像固化的只读参考副本**；
  日常开发用持久盘上的 git 工作副本 `/root/private_data/anima-lora-train`。
* 出新版本：工作副本 `git pull` → `scnet_stage_dockerfile.sh` → 控制台构建新 tag。
* **没通过 `--runtime` 验收的版本不要用**（如 v0.1.0，硬门槛空转）。
* 备用路线：控制台「关机 → 保存开发环境」直接把当前容器存成镜像，不需要拉基础镜像。
  存之前跑 `tools/scnet_pack_image.sh --code` 做同样的检查。
  代价是不可复现（评审看不到构建过程），只在 Dockerfile 路线走不通时用。
