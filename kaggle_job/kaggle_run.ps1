<#
.SYNOPSIS
    把一个 Kaggle script kernel 推上去、轮询到结束、把产物拉回本地。

.DESCRIPTION
    这是"用 CLI 当工作台"的驱动脚本 —— 全程不碰 notebook 网页界面。
    代码在本地 git 里正常管理，这个脚本负责 push / 轮询 / 取回。

    前置：先登录一次（只需一次，凭据会缓存在本机）
        python -m kaggle auth login

.PARAMETER Username
    你的 Kaggle 用户名。会自动写进 kernel-metadata.json 的 id 字段。

.PARAMETER JobDir
    包含 kernel-metadata.json 与代码文件的目录。默认 splash_probe。

.PARAMETER Accelerator
    加速器类型。TPU 用 TpuV5E8。传 "none" 则不带 --accelerator（纯 CPU，
    用来先零配额地验证整条 CLI 链路是否通）。

.PARAMETER TimeoutSec
    传给 Kaggle 的运行时长上限（秒）。探针几分钟就够，设小一点省配额。

.EXAMPLE
    # 第一步：不带加速器跑一次，零 TPU 配额验证 CLI 链路
    .\kaggle_run.ps1 -Username yourname -Accelerator none -TimeoutSec 900

.EXAMPLE
    # 第二步：真正上 TPU
    .\kaggle_run.ps1 -Username yourname -Accelerator TpuV5E8 -TimeoutSec 2700
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Username,
    [string]$JobDir = "splash_probe",
    [string]$Accelerator = "TpuV5E8",
    [int]$TimeoutSec = 2700,
    [int]$PollSec = 30,
    [switch]$NoPull,
    [switch]$PullOnly,
    # 默认只拉报告和日志。探针会把 jax 编译缓存也写进 /kaggle/working，
    # 那是几十个文件、每次全下很慢。传 -FilePattern "" 可拉全部（含缓存）。
    [string]$FilePattern = '(.*_probe.*|.*\.log)$'
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$jobPath = Join-Path $here $JobDir
$metaPath = Join-Path $jobPath "kernel-metadata.json"

if (-not (Test-Path $metaPath)) { throw "找不到 $metaPath" }

# 用 `python -m kaggle` 而非 `kaggle`：pip 装完 Scripts 目录未必在 PATH 上。
# PYTHONUTF8=1 是必须的：中文 Windows 默认 GBK，kernels output 把内核日志写盘时
# 遇到非 ASCII 字符会崩（实测报 'gbk' codec can't encode character '✗'），
# 结果日志文件落成 0 字节。UTF-8 模式同时管住 stdio 与文件默认编码。
function Invoke-Kaggle {
    param([string[]]$KaggleArgs)
    Write-Host "  > python -m kaggle $($KaggleArgs -join ' ')" -ForegroundColor DarkGray
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    & python -m kaggle @KaggleArgs 2>&1
}

# ── 1. 把用户名写进 metadata 的 id ────────────────────────────────────────────
$meta = Get-Content $metaPath -Raw | ConvertFrom-Json
$slug = ($meta.id -split '/')[-1]
$ref = "$Username/$slug"
if ($meta.id -ne $ref) {
    $meta.id = $ref
    $meta | ConvertTo-Json -Depth 10 | Set-Content $metaPath -Encoding UTF8
    Write-Host "已把 kernel id 改写为 $ref" -ForegroundColor Yellow
}

# ── PullOnly：只重新拉一次产物/日志。不 push、不跑、**不消耗任何配额**。
# 上一次拉取因编码崩掉导致日志落成 0 字节时，用这个补救即可，不必重跑。
function Get-KernelOutput {
    param([string]$Ref, [string]$Dir)
    $a = @("kernels", "output", $Ref, "-p", $Dir, "-o")
    if ($FilePattern) { $a += @("--file-pattern", $FilePattern) }
    Invoke-Kaggle $a
}

if ($PullOnly) {
    $outDir = Join-Path $here "output/$slug"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    Write-Host "=== 仅重新拉取 $ref 的产物到 $outDir（不消耗配额）===" -ForegroundColor Cyan
    Get-KernelOutput $ref $outDir | Write-Host
    $rep = Get-ChildItem $outDir -Filter "*_probe_report.txt" -Recurse -File -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($rep) {
        Write-Host "`n=== 探针报告 ===" -ForegroundColor Cyan
        Get-Content $rep.FullName -Encoding UTF8 | Write-Host
    }
    $log = Get-ChildItem $outDir -Filter "*.log" -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($log -and $log.Length -gt 0) {
        Write-Host "`n=== 内核日志 ($($log.Name), $($log.Length) 字节) ===" -ForegroundColor Cyan
        Get-Content $log.FullName -Encoding UTF8 | Write-Host
    }
    else {
        Write-Host "`n日志仍为空。去网页看：https://www.kaggle.com/code/$ref" -ForegroundColor Yellow
    }
    exit 0
}

# ── 2. 先看配额（回答"一次能跑多久 / 还剩多少"）────────────────────────────────
Write-Host "`n=== 当前配额 ===" -ForegroundColor Cyan
Invoke-Kaggle @("quota") | Write-Host

# ── 3. 推送并启动 ─────────────────────────────────────────────────────────────
Write-Host "`n=== 推送 $ref ===" -ForegroundColor Cyan
$pushArgs = @("kernels", "push", "-p", $jobPath, "-t", "$TimeoutSec")
if ($Accelerator -and $Accelerator -ne "none") {
    $pushArgs += @("--accelerator", $Accelerator)
}
$pushOut = Invoke-Kaggle $pushArgs
$pushOut | Write-Host
if ($pushOut -match "error|Error|ERROR") {
    Write-Host "推送疑似失败，停在这里（上面是原始输出）" -ForegroundColor Red
    exit 1
}

# ── 4. 轮询到结束 ─────────────────────────────────────────────────────────────
Write-Host "`n=== 轮询状态（每 $PollSec 秒，Ctrl+C 可随时退出，不影响云端运行）===" -ForegroundColor Cyan
$start = Get-Date
$terminal = @("complete", "error", "cancelAcknowledged", "cancelRequested")
while ($true) {
    Start-Sleep -Seconds $PollSec
    $st = (Invoke-Kaggle @("kernels", "status", $ref)) -join " "
    $elapsed = [int]((Get-Date) - $start).TotalSeconds
    Write-Host ("[{0,5}s] {1}" -f $elapsed, $st.Trim())
    $hit = $terminal | Where-Object { $st -match $_ }
    if ($hit) {
        Write-Host "`n运行结束（状态含：$($hit -join ',')），本次墙钟 ${elapsed}s" -ForegroundColor Green
        break
    }
}

# ── 5. 取回产物 ───────────────────────────────────────────────────────────────
if (-not $NoPull) {
    $outDir = Join-Path $here "output/$slug"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    Write-Host "`n=== 拉取产物到 $outDir ===" -ForegroundColor Cyan
    Get-KernelOutput $ref $outDir | Write-Host
    Write-Host "`n产物文件（不含编译缓存）：" -ForegroundColor Cyan
    Get-ChildItem $outDir -Recurse -File |
        Where-Object { $_.DirectoryName -notmatch 'jax_cache' } |
        Select-Object Name, Length | Format-Table

    # 探针会把纯文本报告写到 /kaggle/working，优先读它——比内核日志更结构化
    $report = Get-ChildItem $outDir -Filter "*_probe_report.txt" -Recurse -File -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($report) {
        Write-Host "`n=== 探针报告 ===" -ForegroundColor Cyan
        Get-Content $report.FullName -Encoding UTF8 | Write-Host
    }
    else {
        $log = Get-ChildItem $outDir -Filter "*.log" -File -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($log -and $log.Length -gt 0) {
            Write-Host "`n=== 内核日志 ===" -ForegroundColor Cyan
            Get-Content $log.FullName -Encoding UTF8 | Write-Host
        }
        else {
            Write-Host "没拿到报告/日志。网页：https://www.kaggle.com/code/$ref" -ForegroundColor Yellow
        }
    }
}

Write-Host "`n=== 结束后再看一次配额（对比可算出本次消耗）===" -ForegroundColor Cyan
Invoke-Kaggle @("quota") | Write-Host
