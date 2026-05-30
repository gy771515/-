import matplotlib.pyplot as plt
import numpy as np


# 工具函数：复用地图绘制逻辑（减少代码冗余）
def _plot_map_background(ax, grid):
    """内部工具函数：绘制地图背景和基础样式"""
    grid = np.array(grid)
    if grid.ndim != 2:
        raise ValueError("grid必须是二维数组！")
    rows, cols = grid.shape
    ax.imshow(grid, cmap='gray_r', origin="upper", extent=[0, cols, rows, 0])
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(0, cols + 1, 1))
    ax.set_yticks(np.arange(0, rows + 1, 1))
    ax.grid(True, linestyle='--', alpha=0.7)
    return ax


def visualize(grid, start, end, path=None, visited=None, title="Path Planning"):
    """优化版路径规划可视化：增加空值防护、复用工具函数"""
    # 空值防护
    grid = np.array(grid)
    if grid.ndim != 2:
        raise ValueError("grid必须是二维数组！")
    if not (isinstance(start, (tuple, list)) and isinstance(end, (tuple, list))):
        raise ValueError("start/end必须是(x,y)格式的元组/列表！")

    rows, cols = grid.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    ax = _plot_map_background(ax, grid)  # 复用地图绘制

    # 画起点+终点
    ax.scatter(start[0] + 0.5, start[1] + 0.5, c='green', s=120, marker='s', label='Start')
    ax.scatter(end[0] + 0.5, end[1] + 0.5, c='orange', s=150, marker='*', label='Goal')

    # 画访问节点（空值防护）
    if visited and len(visited) > 0:
        vx = [p[0] + 0.5 for p in visited]
        vy = [p[1] + 0.5 for p in visited]
        ax.scatter(vx, vy, c='lightblue', s=20, label='Visited Nodes')

    # 画路径（空值防护）
    if path and len(path) > 0:
        px = [p[0] + 0.5 for p in path]
        py = [p[1] + 0.5 for p in path]
        ax.plot(px, py, c='red', linewidth=2, label='Discrete Path')

    # 样式优化
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def visualize_trajectory(grid, start, end, expected_traj, actual_traj, title="Trajectory Tracking"):
    """
    优化版轨迹跟踪可视化：空值防护、降采样、边界适配、修复笔误
    :param grid: 地图数组
    :param start/end: 起点/终点（栅格索引）
    :param expected_traj: 期望连续轨迹
    :param actual_traj: 实际跟踪轨迹
    """
    # 核心修复1：空值防护 & 数据校验
    grid = np.array(grid)
    if grid.ndim != 2:
        raise ValueError("grid必须是二维数组！")
    if not expected_traj or len(expected_traj) < 2:
        print("警告：期望轨迹为空或点数不足，仅显示地图和起止点")
        expected_traj = [start, end]
    if not actual_traj or len(actual_traj) < 2:
        print("警告：实际轨迹为空或点数不足，仅显示地图和起止点")
        actual_traj = [start, end]

    rows, cols = grid.shape
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = _plot_map_background(ax, grid)  # 复用地图绘制

    # 核心修复2：降采样（减少点数，提升绘图性能）
    # 每10个点取一个，避免上千个点导致卡顿
    def downsample_traj(traj, step=10):
        return traj[::step] if len(traj) > step else traj

    exp_traj_down = downsample_traj(expected_traj)
    act_traj_down = downsample_traj(actual_traj)

    # 核心修复3：修复变量名笔误 + 适配浮点数坐标显示
    exp_x = [p[0] for p in exp_traj_down]
    exp_y = [p[1] for p in exp_traj_down]
    act_x = [p[0] for p in act_traj_down]  # 修复：act_traj → actual_traj
    act_y = [p[1] for p in act_traj_down]  # 修复：act_traj → actual_traj

    # 核心修复4：边界适配（扩展显示范围，避免轨迹截断）
    all_x = exp_x + act_x + [start[0] + 0.5, end[0] + 0.5]
    all_y = exp_y + act_y + [start[1] + 0.5, end[1] + 0.5]
    ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax.set_ylim(min(all_y) - 1, max(all_y) + 1)

    # 绘制轨迹（强化样式区分）
    ax.plot(exp_x, exp_y, c='darkblue', linewidth=3,
            label='Expected Trajectory', alpha=0.8, linestyle='-')
    ax.plot(act_x, act_y, c='red', linewidth=2,
            label='Actual Trajectory (PID)', alpha=0.9, linestyle='--')
    ax.scatter(act_x[::5], act_y[::5], c='red', s=15, alpha=1.0)  # 加密标记点

    # 绘制起点+终点（适配连续坐标）
    ax.scatter(start[0] + 0.5, start[1] + 0.5, c='green', s=120, marker='s', label='Start')
    ax.scatter(end[0] + 0.5, end[1] + 0.5, c='orange', s=150, marker='*', label='Goal')

    # 样式优化
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xticks(np.arange(0, cols + 1, 2))
    ax.set_yticks(np.arange(0, rows + 1, 2))
    plt.tight_layout()
    plt.show()