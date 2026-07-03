// gpu_hog.cu — 动态最大化 GPU 压力测试工具
// -----------------------------------------------------------------------------
// 相比旧版（一次性 cudaMalloc 95% 显存 + 一个 `i++` 空转内核）的改进：
//
//   1. 动态、灵敏地占满显存：后台轮询 cudaMemGetInfo，把空闲显存按块抢到只剩
//      安全余量（--keep-free）。显存被别的进程释放出来时，下一个轮询周期立刻补占；
//      当别的进程逼近硬下限（--hard-min-free）时，按块让出，避免把驱动/他人挤爆。
//
//   2. 真正的计算压力：旧版的 `while(true) i++` 只占着 SM、几乎不产生 FLOPs 和功耗。
//      这里换成 grid-stride 的 FP32 FMA 重载内核（多独立累加器打满 FP 流水线）叠加
//      对显存的读写（带宽压力），由独立线程连续投放，保持 ~100% 计算利用率。
//
//   3. 可配置、可优雅退出（Ctrl+C / SIGTERM 释放全部显存后退出）。
//
// 注意：这是给自有/自己有权使用的机器做压力测试与占位用的工具，默认行为是激进占用。
//       想温和地“留出余量给训练”，用同目录的 gpu_vram_guard.py 更合适。
//
// 编译（在目标机器本地编译，产物平台相关；云端 Linux 上重新编译才能在云端运行）：
//   nvcc -O3 -arch=native gpu_hog.cu -o gpu_hog          # 单机自动匹配本机架构
//   nvcc -O3 -gencode arch=compute_80,code=sm_80 gpu_hog.cu -o gpu_hog   # 显式指定
//
// 用法示例：
//   ./gpu_hog                                 # 默认：设备0，抢到只剩 512MiB 空闲
//   ./gpu_hog --device 0 --keep-free 1024 --interval 100 --fma 1024
//   ./gpu_hog --max-reserve 76000 --no-touch  # 最多占 76000MiB，跳过初始写入
// -----------------------------------------------------------------------------

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr size_t MiB = 1024ull * 1024ull;

// ---- 可配置参数（命令行解析后填充）----------------------------------------
struct Config {
    int    device        = 0;      // CUDA 设备号
    size_t keep_free_mb  = 512;    // 目标保留空闲显存（MiB）；抢到只剩这么多为止
    size_t hard_min_mb   = 0;      // 空闲低于此值立即让出一块（MiB）；0 → keep_free/2
    size_t block_mb      = 512;    // 每次抢占/让出的块粒度（MiB）
    size_t max_reserve_mb = 0;     // 本进程最多占用（MiB）；0 → 不限
    int    interval_ms   = 200;    // 显存轮询间隔（毫秒），越小越灵敏
    size_t scratch_mb    = 1024;   // 计算内核工作缓冲大小（MiB）
    int    fma_iters     = 512;    // 每个元素的内层 FMA 迭代数，越大计算占比越高
    int    blocks_per_sm = 32;     // 每个 SM 的线程块数（网格规模）
    bool   touch         = true;   // 抢占后是否写入（让显存计入“已用”，更真实）
    bool   quiet         = false;  // 只打印关键事件
};

std::atomic<bool> g_stop{false};

void handle_signal(int) { g_stop.store(true); }

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr, "[gpu_hog] CUDA error %s at %s:%d: %s\n",      \
                         cudaGetErrorName(_err), __FILE__, __LINE__,            \
                         cudaGetErrorString(_err));                             \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ---- 计算重载内核 -----------------------------------------------------------
// 对每个元素：读一次 + 若干独立累加器的 FMA 链 + 写一次。
// 多个独立累加器隐藏 FMA 延迟、打满 FP32 吞吐；读写提供显存带宽压力；
// 写回结果依赖全部累加器，阻止编译器把计算优化掉（dead-code elimination）。
__global__ void stress_kernel(float* __restrict__ data, size_t n, int inner) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float a = data[i];
        float b = a * 1.0000011f + 0.5f;
        float c = a * 0.9999989f + 0.25f;
        float d = a + 1.0f;
        float e = a - 0.5f;
        float f = a * 0.5f + 0.125f;
        float g = a * 2.0f - 0.75f;
        float h = a + 0.333f;
#pragma unroll 8
        for (int k = 0; k < inner; ++k) {
            a = fmaf(a, 1.0000001f, 0.0000103f);
            b = fmaf(b, 0.9999999f, 0.0000107f);
            c = fmaf(c, 1.0000002f, 0.0000109f);
            d = fmaf(d, 0.9999998f, 0.0000113f);
            e = fmaf(e, 1.0000003f, 0.0000119f);
            f = fmaf(f, 0.9999997f, 0.0000121f);
            g = fmaf(g, 1.0000004f, 0.0000127f);
            h = fmaf(h, 0.9999996f, 0.0000131f);
        }
        data[i] = a + b + c + d + e + f + g + h;
    }
}

// ---- 已占用的显存块 ---------------------------------------------------------
struct Block {
    void*  ptr;
    size_t bytes;
};

void print_usage(const char* prog) {
    std::printf(
        "用法: %s [选项]\n"
        "  --device N          CUDA 设备号 (默认 0)\n"
        "  --keep-free MB      目标保留空闲显存 MiB (默认 512)\n"
        "  --hard-min-free MB  空闲低于此值立即让出一块 MiB (默认 keep-free/2)\n"
        "  --block MB          抢占/让出块粒度 MiB (默认 512)\n"
        "  --max-reserve MB    本进程最多占用 MiB, 0=不限 (默认 0)\n"
        "  --interval MS       显存轮询间隔毫秒, 越小越灵敏 (默认 200)\n"
        "  --scratch MB        计算工作缓冲大小 MiB (默认 1024)\n"
        "  --fma N             每元素内层 FMA 迭代数, 越大计算越重 (默认 512)\n"
        "  --blocks-per-sm N   每个 SM 的线程块数 (默认 32)\n"
        "  --no-touch          抢占后不写入 (启动更快, 显存统计可能偏保守)\n"
        "  --quiet             只打印关键事件\n"
        "  --help              显示本帮助\n",
        prog);
}

bool parse_size(const char* s, size_t* out) {
    char* end = nullptr;
    long long v = std::strtoll(s, &end, 10);
    if (end == s || v < 0) return false;
    *out = static_cast<size_t>(v);
    return true;
}

bool parse_args(int argc, char** argv, Config* cfg) {
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "[gpu_hog] %s 需要一个参数\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        const char* a = argv[i];
        size_t tmp;
        if (!std::strcmp(a, "--device"))            cfg->device = std::atoi(need(a));
        else if (!std::strcmp(a, "--keep-free"))    { parse_size(need(a), &cfg->keep_free_mb); }
        else if (!std::strcmp(a, "--hard-min-free")){ parse_size(need(a), &cfg->hard_min_mb); }
        else if (!std::strcmp(a, "--block"))        { parse_size(need(a), &cfg->block_mb); }
        else if (!std::strcmp(a, "--max-reserve"))  { parse_size(need(a), &cfg->max_reserve_mb); }
        else if (!std::strcmp(a, "--interval"))     cfg->interval_ms = std::atoi(need(a));
        else if (!std::strcmp(a, "--scratch"))      { parse_size(need(a), &cfg->scratch_mb); }
        else if (!std::strcmp(a, "--fma"))          cfg->fma_iters = std::atoi(need(a));
        else if (!std::strcmp(a, "--blocks-per-sm"))cfg->blocks_per_sm = std::atoi(need(a));
        else if (!std::strcmp(a, "--no-touch"))     cfg->touch = false;
        else if (!std::strcmp(a, "--quiet"))        cfg->quiet = true;
        else if (!std::strcmp(a, "--help") || !std::strcmp(a, "-h")) { print_usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "[gpu_hog] 未知参数: %s\n", a); print_usage(argv[0]); return false; }
        (void)tmp;
    }
    if (cfg->block_mb == 0)   cfg->block_mb = 256;
    if (cfg->scratch_mb == 0) cfg->scratch_mb = 256;
    if (cfg->fma_iters < 1)   cfg->fma_iters = 1;
    if (cfg->blocks_per_sm < 1) cfg->blocks_per_sm = 1;
    if (cfg->interval_ms < 1) cfg->interval_ms = 1;
    if (cfg->hard_min_mb == 0)
        cfg->hard_min_mb = cfg->keep_free_mb / 2 > 256 ? cfg->keep_free_mb / 2 : 256;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    if (!parse_args(argc, argv, &cfg)) return 2;

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    CUDA_CHECK(cudaSetDevice(cfg.device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));

    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    std::printf(
        "[gpu_hog] 设备 %d: %s | 显存 total=%zu MiB, free=%zu MiB\n"
        "          keep_free=%zu MiB, hard_min=%zu MiB, block=%zu MiB, interval=%dms\n"
        "          scratch=%zu MiB, fma=%d, blocks_per_sm=%d (SM=%d), max_reserve=%zu MiB\n",
        cfg.device, prop.name, total_b / MiB, free_b / MiB,
        cfg.keep_free_mb, cfg.hard_min_mb, cfg.block_mb, cfg.interval_ms,
        cfg.scratch_mb, cfg.fma_iters, cfg.blocks_per_sm, prop.multiProcessorCount,
        cfg.max_reserve_mb);
    std::fflush(stdout);

    // ---- 计算工作缓冲 -------------------------------------------------------
    const size_t scratch_bytes = cfg.scratch_mb * MiB;
    const size_t scratch_n = scratch_bytes / sizeof(float);
    float* scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&scratch, scratch_bytes));
    CUDA_CHECK(cudaMemset(scratch, 1, scratch_bytes));

    cudaStream_t compute_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));

    const int threads = 256;
    const int grid = prop.multiProcessorCount * cfg.blocks_per_sm;
    const int fma_iters = cfg.fma_iters;

    // ---- 计算线程：连续投放重载内核，保持 GPU ~100% 利用率 -----------------
    // 不逐次同步；驱动的启动队列填满后 launch 会自然阻塞，形成自节流并持续占满设备。
    std::thread compute_thread([&]() {
        int since_sync = 0;
        while (!g_stop.load()) {
            stress_kernel<<<grid, threads, 0, compute_stream>>>(scratch, scratch_n, fma_iters);
            // 周期性同步：既能及时暴露内核错误，又避免启动队列无界堆积。
            if (++since_sync >= 64) {
                since_sync = 0;
                if (cudaStreamSynchronize(compute_stream) != cudaSuccess) break;
            }
        }
        cudaStreamSynchronize(compute_stream);
    });

    // ---- 主线程：动态占用/让出显存 -----------------------------------------
    std::vector<Block> blocks;
    size_t reserved_b = 0;
    const size_t keep_free_b = cfg.keep_free_mb * MiB;
    const size_t hard_min_b  = cfg.hard_min_mb * MiB;
    const size_t block_b     = cfg.block_mb * MiB;
    const size_t max_reserve_b = cfg.max_reserve_mb * MiB;
    size_t grab_block_b = block_b;  // 抢占块粒度，OOM 时自适应收缩
    bool announced_full = false;

    while (!g_stop.load()) {
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));

        // 1) 空闲低于硬下限：让出一块，把显存还给逼近的进程。
        if (free_b < hard_min_b && !blocks.empty()) {
            Block b = blocks.back();
            blocks.pop_back();
            cudaFree(b.ptr);
            reserved_b -= b.bytes;
            grab_block_b = block_b;  // 让出后恢复默认粒度
            announced_full = false;
            if (!cfg.quiet)
                std::printf("[gpu_hog] 空闲 %zu MiB < 硬下限，让出 %zu MiB，现占用 %zu MiB\n",
                            free_b / MiB, b.bytes / MiB, reserved_b / MiB);
            std::fflush(stdout);
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg.interval_ms / 2 + 1));
            continue;
        }

        // 2) 空闲高于目标余量：抢占——灵敏地把新释放出来的显存补占回来。
        bool cap_ok = (max_reserve_b == 0) || (reserved_b < max_reserve_b);
        if (free_b > keep_free_b + grab_block_b && cap_ok) {
            size_t want = grab_block_b;
            if (free_b - keep_free_b < want) want = free_b - keep_free_b;
            if (max_reserve_b != 0 && reserved_b + want > max_reserve_b)
                want = max_reserve_b - reserved_b;

            void* ptr = nullptr;
            if (want > 0 && cudaMalloc(&ptr, want) == cudaSuccess) {
                if (cfg.touch) cudaMemset(ptr, 1, want);  // 写入使其计入“已用”显存
                blocks.push_back({ptr, want});
                reserved_b += want;
                announced_full = false;
                if (!cfg.quiet)
                    std::printf("[gpu_hog] 抢占 %zu MiB（抢占前空闲 %zu MiB），现占用 %zu MiB\n",
                                want / MiB, free_b / MiB, reserved_b / MiB);
                std::fflush(stdout);
                continue;  // 不睡眠，尽快抢下一块（灵敏占满）
            }
            // 抢占失败（碎片/竞争）：收缩块粒度重试，直到 block/8。
            cudaGetLastError();  // 清掉 OOM 粘性错误
            if (grab_block_b > block_b / 8 + MiB)
                grab_block_b /= 2;
        } else if (!announced_full) {
            announced_full = true;
            std::printf("[gpu_hog] 已占满：占用 %zu MiB，空闲 %zu MiB（持续压测中，Ctrl+C 退出）\n",
                        reserved_b / MiB, free_b / MiB);
            std::fflush(stdout);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(cfg.interval_ms));
    }

    // ---- 优雅退出：停计算、释放全部显存 ------------------------------------
    std::printf("[gpu_hog] 收到退出信号，正在释放显存...\n");
    std::fflush(stdout);
    g_stop.store(true);
    compute_thread.join();
    for (auto& b : blocks) cudaFree(b.ptr);
    cudaFree(scratch);
    cudaStreamDestroy(compute_stream);
    std::printf("[gpu_hog] 已释放 %zu MiB 显存并退出。\n", reserved_b / MiB);
    return 0;
}
