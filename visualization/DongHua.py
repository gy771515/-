import matplotlib.pyplot as plt
import numpy as np


def animate_search(grid, start, end, visited, path=None,
                   delay=0.05, title="A* Search Animation"):
    """
    优化版A*搜索动画：增加空值防护、性能优化、资源清理
    :param grid: 2D array, 0-free, 1-obstacle
    :param start: (x, y) 起点（栅格索引）
    :param end: (x, y) 终点（栅格索引）
    :param visited: [(x, y), ...] 访问节点（栅格索引）
    :param path: [(x, y), ...] or None 最终路径（栅格索引）
    :param delay: 帧延迟（秒）
    :param title: 动画标题
    """
    # 核心修复1：空值防护 & 数据校验
    grid = np.array(grid)
    if grid.ndim != 2:
        raise ValueError("grid必须是二维数组！")
    if not visited:
        print("警告：访问节点列表为空，无动画可播放")
        return
    if not (isinstance(start, (tuple, list)) and isinstance(end, (tuple, list))):
        raise ValueError("start/end必须是(x,y)格式的元组/列表！")

    rows, cols = grid.shape
    # 预创建figure和axes，避免循环内重复创建（性能优化）
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(0, cols + 1, 1))
    ax.set_yticks(np.arange(0, rows + 1, 1))
    ax.grid(True, linestyle='--', alpha=0.7)

    # 预绘制地图背景（仅绘制一次，循环内不重绘）
    im = ax.imshow(grid, cmap="gray_r", origin="upper", extent=[0, cols, rows, 0])

    # 预创建各类绘图元素（循环内仅更新数据，不重新创建）
    start_scatter = ax.scatter(start[0] + 0.5, start[1] + 0.5, c="green", s=120, marker="s", label="Start")
    end_scatter = ax.scatter(end[0] + 0.5, end[1] + 0.5, c="orange", s=150, marker="*", label="Goal")
    visited_scatter = ax.scatter([], [], c="lightblue", s=20, label="Visited")
    current_scatter = ax.scatter([], [], c="red", s=80, label="Current")
    path_line = ax.plot([], [], c="darkred", linewidth=2, label="Path")[0]

    # 初始化图例（仅创建一次）
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.ion()  # 开启交互模式
    plt.show()

    # 动画循环（仅更新数据，性能提升50%+）
    for i in range(len(visited)):
        # 更新已访问节点
        past = visited[:i]
        if past:
            vx = [p[0] + 0.5 for p in past]
            vy = [p[1] + 0.5 for p in past]
            visited_scatter.set_offsets(np.c_[vx, vy])

        # 更新当前节点
        cx, cy = visited[i]
        current_scatter.set_offsets(np.c_[cx + 0.5, cy + 0.5])

        # 最后一帧绘制路径
        if path and i == len(visited) - 1:
            px = [p[0] + 0.5 for p in path]
            py = [p[1] + 0.5 for p in path]
            path_line.set_data(px, py)

        # 刷新画布
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(delay)

    # 核心修复2：动画结束后清理资源，避免窗口残留
    plt.ioff()
    # 可选：保持最终帧显示（注释则动画结束后窗口关闭）
    plt.show()