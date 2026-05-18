#!/usr/bin/env python3
import numpy as np
import struct
import sys
import matplotlib.pyplot as plt

def read_rsse_binary(filename):
    """读取RSSE V2格式的binary文件"""
    with open(filename, 'rb') as f:
        magic = f.read(4).decode('ascii')
        assert magic == 'RSSE', f"Invalid magic: {magic}"

        version = struct.unpack('<i', f.read(4))[0]
        lx, ly, nn, nb, mm = struct.unpack('<5i', f.read(20))
        beta, surface_n = struct.unpack('<2d', f.read(16))

        nh_list, K_list, parity_list = [], [], []

        while True:
            data = f.read(8)
            if len(data) < 8: break

            nh, K = struct.unpack('<2i', data)
            nh_list.append(nh)
            K_list.append(K)

            # Read parity_prefix and get final parity
            if nh > 0:
                parity_prefix = np.frombuffer(f.read(nh), dtype=np.int8)
                parity_list.append(parity_prefix[-1])
                f.read(nh * 4)  # skip opstring
            else:
                parity_list.append(0)

        return np.array(nh_list), np.array(K_list), np.array(parity_list), beta, nn

def autocorr(x, max_lag=None):
    """计算自相关函数（直接法）"""
    x = np.asarray(x)
    x = x - x.mean()
    c0 = np.dot(x, x) / len(x)

    if max_lag is None:
        max_lag = len(x) // 4

    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        acf[lag] = np.dot(x[:-lag or None], x[lag:]) / (len(x) - lag) / c0

    return acf

def autocorr_fft(x):
    """FFT法计算自相关函数（O(N log N)）"""
    x = np.asarray(x)
    x = x - x.mean()
    n = len(x)

    f = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    acf /= acf[0]

    return acf

def blocking_analysis(x, max_block_size=None):
    """Blocking/binning法估计误差

    返回 block_vars[i] = Var(block_means) / n_blocks
    即对整体均值方差的估计（Var(mean)），不是 Var(block means)
    """
    x = np.asarray(x)
    n = len(x)

    if max_block_size is None:
        max_block_size = n // 4

    block_sizes = []
    block_vars = []

    B = 1
    while B <= max_block_size:
        n_blocks = n // B
        if n_blocks < 2:
            break

        blocks = x[:n_blocks*B].reshape(n_blocks, B).mean(axis=1)
        var = blocks.var(ddof=1) / n_blocks  # Var(mean) estimate

        block_sizes.append(B)
        block_vars.append(var)

        B *= 2

    return np.array(block_sizes), np.array(block_vars)

def detect_plateau(block_sizes, block_vars, n_samples, min_blocks=32):
    """检测 blocking curve 的 plateau 区域

    策略：在 n_blocks >= min_blocks 的区域内，找到最平坦的连续区域
    """
    n_blocks_arr = n_samples // block_sizes
    valid_mask = n_blocks_arr >= min_blocks

    if np.sum(valid_mask) < 3:
        return None, None, valid_mask

    valid_idx = np.where(valid_mask)[0]
    valid_vars = block_vars[valid_mask]

    # 计算相对变化率
    rel_changes = np.abs(np.diff(valid_vars)) / valid_vars[:-1]

    # 找到最平坦的连续区域（相对变化 < 5%）
    flat_threshold = 0.05
    is_flat = rel_changes < flat_threshold

    if np.any(is_flat):
        # 找最后一个平坦区域
        flat_regions = []
        start = None
        for i, flat in enumerate(is_flat):
            if flat and start is None:
                start = i
            elif not flat and start is not None:
                flat_regions.append((start, i))
                start = None
        if start is not None:
            flat_regions.append((start, len(is_flat)))

        if flat_regions:
            # 取最后一个平坦区域
            last_start, last_end = flat_regions[-1]
            plateau_idx = valid_idx[last_start:last_end+1]
            return plateau_idx, valid_idx, valid_mask

    # Fallback: 找最小方差附近的点（排除最后2个噪声点）
    search_end = max(3, len(valid_vars) - 2)
    min_idx = np.argmin(valid_vars[:search_end])

    # 取 min_idx 附近的 3 个点
    plateau_start = max(0, min_idx - 1)
    plateau_end = min(len(valid_vars), min_idx + 2)
    plateau_idx = valid_idx[plateau_start:plateau_end]

    return plateau_idx, valid_idx, valid_mask

def integrated_time(acf, c=5.0):
    """Madras-Sokal自适应窗口法计算积分自相关时间

    截断到第一个非正值，避免噪声累积
    使用相对收敛判据
    """
    # 截断到第一个非正值
    cutoff = len(acf)
    for i in range(1, len(acf)):
        if acf[i] <= 0:
            cutoff = i
            break

    acf_trunc = acf[:cutoff]

    tau = 0.5
    for _ in range(10):  # 迭代收敛
        W = int(c * tau)
        if W >= len(acf_trunc):
            W = len(acf_trunc) - 1

        tau_new = 0.5 + np.sum(acf_trunc[1:W+1])

        # 相对收敛判据
        if tau > 0 and abs(tau_new - tau) / tau < 0.01:
            break
        tau = tau_new

    return tau

def plot_acf(acf_energy, acf_sign, filename):
    """绘制能量和sign的ACF曲线（半对数图）"""
    max_lag = min(1000, len(acf_energy))
    plt.figure(figsize=(8, 5))
    plt.semilogy(range(max_lag), np.abs(acf_energy[:max_lag]), 'o-', markersize=2, label='Energy', alpha=0.7)
    plt.semilogy(range(max_lag), np.abs(acf_sign[:max_lag]), 's-', markersize=2, label='Sign', alpha=0.7)
    plt.xlabel('lag')
    plt.ylabel('|acf|')
    plt.title('ACF Comparison (semilogy)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_blocking(block_sizes, block_vars, filename):
    """绘制Blocking curve（log-log）"""
    plt.figure(figsize=(6, 4))
    plt.loglog(block_sizes, block_vars, 'o-', label='raw')
    plt.loglog(block_sizes, block_vars / block_vars[0], 's-', label='normalized')
    plt.xlabel('block size')
    plt.ylabel('variance')
    plt.title('Blocking Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_tau_vs_window(acf, filename):
    """绘制tau vs window（window stability）"""
    max_W = min(2000, len(acf))
    Ws = np.arange(1, max_W)
    tau_list = [0.5 + np.sum(acf[1:W]) for W in Ws]

    plt.figure(figsize=(6, 4))
    plt.plot(Ws, tau_list)
    plt.xlabel('window size W')
    plt.ylabel('tau_int(W)')
    plt.title('Tau vs Window (Madras-Sokal stability)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python autocorr_analysis.py <rsse_binary_file>")
        sys.exit(1)

    filename = sys.argv[1]
    nh_series, K_series, parity_series, beta, nn = read_rsse_binary(filename)

    energy = -nh_series / (beta * nn)
    n_samples = len(energy)

    print(f"# File: {filename}")
    print(f"# Samples: {n_samples}")
    print(f"# beta={beta:.3f}, nn={nn}")
    print()

    # 方法1: 直接法自相关
    print("=== Method 1: Direct ACF ===")
    acf_e = autocorr(energy, max_lag=min(1000, len(energy)//4))
    tau_e = integrated_time(acf_e)
    print(f"tau_int: {tau_e:.2f}")
    print(f"Effective samples: {n_samples / (2*tau_e):.1f}")
    print()

    # 方法2: FFT自相关
    print("=== Method 2: FFT ACF ===")
    acf_e_fft = autocorr_fft(energy)
    tau_e_fft = integrated_time(acf_e_fft)
    print(f"tau_int: {tau_e_fft:.2f}")
    print(f"Effective samples: {n_samples / (2*tau_e_fft):.1f}")
    print()

    # 方法3: Blocking分析
    print("=== Method 3: Blocking Analysis ===")
    block_sizes, block_vars = blocking_analysis(energy)

    n_blocks_arr = n_samples // block_sizes

    # 使用新的 plateau 检测
    plateau_idx, valid_idx, valid_mask = detect_plateau(block_sizes, block_vars, n_samples)

    if plateau_idx is not None and len(plateau_idx) > 0:
        plateau_sizes = block_sizes[plateau_idx]
        plateau_vars = block_vars[plateau_idx]

        plateau_var = np.mean(plateau_vars)
        plateau_B = np.mean(plateau_sizes)

        # 对每个 plateau 点分别估计 tau_int，再取平均
        tau_block_vals = 0.5 * plateau_vars / block_vars[0]
        tau_block = np.mean(tau_block_vals)

        print(f"Plateau variance: {plateau_var:.6e} (avg of {len(plateau_idx)} plateau points)")
        print(f"Plateau block size: {plateau_B:.1f}")
        print(f"tau_int: {tau_block:.2f}")
        print(f"Plateau tau estimates: {tau_block_vals}")
        print(f"Plateau indices: {plateau_idx}")
        print(f"Plateau n_blocks: {n_blocks_arr[plateau_idx]}")

        # 诊断：检查 tau_block 是否异常大
        if tau_block > 10 * tau_e_fft:
            print(f"WARNING: tau_block ({tau_block:.2f}) >> tau_fft ({tau_e_fft:.2f})")
    else:
        tau_block = np.nan
        print("Not enough blocks (n_blocks >= 32) for reliable plateau estimation")

    print(f"Block sizes: {block_sizes}")
    print(f"Block vars: {block_vars}")
    print(f"n_blocks: {n_blocks_arr}")
    print(f"Valid mask (n_blocks >= 32): {valid_mask}")
    print()

    # Energy均值和误差条
    print("=== Energy Statistics ===")
    mean_e = np.mean(energy)
    var_e = np.var(energy)
    stderr_naive = np.sqrt(var_e / n_samples)
    stderr_corrected = np.sqrt(2 * tau_e_fft * var_e / n_samples)

    print(f"Mean energy: {mean_e:.6f}")
    print(f"Naive stderr: {stderr_naive:.6e}")
    print(f"Corrected stderr (using FFT tau_int): {stderr_corrected:.6e}")
    print(f"Correction factor: {stderr_corrected / stderr_naive:.2f}")
    print()

    # K的自相关（FFT法）
    print("=== K autocorrelation (FFT) ===")
    acf_K = autocorr_fft(K_series)
    tau_K = integrated_time(acf_K)
    print(f"tau_int: {tau_K:.2f}")
    print()

    # Parity的自相关（FFT法）
    print("=== Parity autocorrelation (FFT) ===")
    sign_series = 2 * parity_series - 1  # Convert 0/1 to -1/+1
    acf_sign = autocorr_fft(sign_series)
    tau_sign = integrated_time(acf_sign)
    print(f"tau_int: {tau_sign:.2f}")
    print()

    print("# lag  acf_direct  acf_fft")
    for i in range(0, min(100, len(acf_e)), 10):
        print(f"{i:6d}  {acf_e[i]:10.6f}  {acf_e_fft[i]:10.6f}")

    # Reblocking curve
    print("\n=== Generating reblocking curve ===")

    # 计算归一化的 tau 估计
    tau_normalized = 0.5 * block_vars / block_vars[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：block variance
    ax1.loglog(block_sizes, block_vars, 'o-')
    if plateau_idx is not None and len(plateau_idx) > 0:
        ax1.axhline(plateau_var, color='r', linestyle='--', alpha=0.5, label='Plateau estimate')
    ax1.set_xlabel('Block size B')
    ax1.set_ylabel('Var(block mean) / n_blocks')
    ax1.set_title('Block Variance vs Block Size')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 右图：归一化的 tau
    ax2.semilogx(block_sizes, tau_normalized, 'o-')
    if plateau_idx is not None and len(plateau_idx) > 0:
        ax2.axhline(tau_block, color='r', linestyle='--', alpha=0.5, label=f'tau_int={tau_block:.2f}')
    ax2.set_xlabel('Block size B')
    ax2.set_ylabel('0.5 * Var_B / Var_1')
    ax2.set_title('Normalized tau_int vs Block Size')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    output_pdf = filename.replace('.bin', '_reblocking.pdf')
    plt.savefig(output_pdf)
    print(f"Saved reblocking curve to: {output_pdf}")

    # 生成三个额外的可视化图（保存为PDF）
    import os
    output_dir = os.path.dirname(filename)
    base_name = os.path.basename(filename).replace('.bin', '')

    acf_pdf = os.path.join(output_dir, f'{base_name}_acf_comparison.pdf')
    blocking_pdf = os.path.join(output_dir, f'{base_name}_blocking_curve.pdf')
    tau_window_pdf = os.path.join(output_dir, f'{base_name}_tau_vs_window.pdf')

    plot_acf(acf_e_fft, acf_sign, acf_pdf)
    plot_blocking(block_sizes, block_vars, blocking_pdf)
    plot_tau_vs_window(acf_e_fft, tau_window_pdf)

    print(f"\nPlots saved:")
    print(f"  {acf_pdf}")
    print(f"  {blocking_pdf}")
    print(f"  {tau_window_pdf}")

