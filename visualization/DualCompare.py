import matplotlib.pyplot as plt
import numpy as np
import time

def animate_dual_search(grid, start, end, visited_a, path_a, visited_d, path_d, delay=0.001):
    """
    并行动画对比 A* 和 Dijkstra 算法
    """
    rows, cols = grid.shape
    fig, (ax_a, ax_d) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 初始化 A* 视图
    ax_a.set_title("A* Algorithm (Heuristic)")
    ax_a.set_xlim(0, cols)
    ax_a.set_ylim(0, rows)
    ax_a.imshow(grid, cmap="gray_r", origin="upper", extent=[0, cols, rows, 0])
    visited_scatter_a = ax_a.scatter([], [], c="lightblue", s=10, label="Visited")
    current_scatter_a = ax_a.scatter([], [], c="red", s=30)
    path_line_a = ax_a.plot([], [], c="darkred", linewidth=2, label="Path")[0]
    stats_text_a = ax_a.text(0.5, -0.1, "", transform=ax_a.transAxes, ha="center", fontsize=10, color="blue")
    
    # 初始化 Dijkstra 视图
    ax_d.set_title("Dijkstra Algorithm (Breadth-first)")
    ax_d.set_xlim(0, cols)
    ax_d.set_ylim(0, rows)
    ax_d.imshow(grid, cmap="gray_r", origin="upper", extent=[0, cols, rows, 0])
    visited_scatter_d = ax_d.scatter([], [], c="lightgreen", s=10, label="Visited")
    current_scatter_d = ax_d.scatter([], [], c="red", s=30)
    path_line_d = ax_d.plot([], [], c="darkgreen", linewidth=2, label="Path")[0]
    stats_text_d = ax_d.text(0.5, -0.1, "", transform=ax_d.transAxes, ha="center", fontsize=10, color="green")

    # 绘制起点终点
    for ax in [ax_a, ax_d]:
        ax.scatter(start[0] + 0.5, start[1] + 0.5, c="blue", s=100, marker="s")
        ax.scatter(end[0] + 0.5, end[1] + 0.5, c="orange", s=120, marker="*")

    plt.tight_layout()
    plt.ion()
    plt.show()

    max_steps = max(len(visited_a), len(visited_d))
    
    for i in range(max_steps):
        # 更新 A*
        if i < len(visited_a):
            past_a = visited_a[:i+1]
            vx_a = [p[0] + 0.5 for p in past_a]
            vy_a = [p[1] + 0.5 for p in past_a]
            visited_scatter_a.set_offsets(np.c_[vx_a, vy_a])
            current_scatter_a.set_offsets(np.c_[visited_a[i][0] + 0.5, visited_a[i][1] + 0.5])
            stats_text_a.set_text(f"Nodes Visited: {i+1}")
            
            if path_a and i == len(visited_a) - 1:
                px_a = [p[0] + 0.5 for p in path_a]
                py_a = [p[1] + 0.5 for p in path_a]
                path_line_a.set_data(px_a, py_a)
                stats_text_a.set_text(f"A* Done! Nodes: {len(visited_a)}, Path: {len(path_a)}")

        # 更新 Dijkstra
        if i < len(visited_d):
            past_d = visited_d[:i+1]
            vx_d = [p[0] + 0.5 for p in past_d]
            vy_d = [p[1] + 0.5 for p in past_d]
            visited_scatter_d.set_offsets(np.c_[vx_d, vy_d])
            current_scatter_d.set_offsets(np.c_[visited_d[i][0] + 0.5, visited_d[i][1] + 0.5])
            stats_text_d.set_text(f"Nodes Visited: {i+1}")
            
            if path_d and i == len(visited_d) - 1:
                px_d = [p[0] + 0.5 for p in path_d]
                py_d = [p[1] + 0.5 for p in path_d]
                path_line_d.set_data(px_d, py_d)
                stats_text_d.set_text(f"Dijkstra Done! Nodes: {len(visited_d)}, Path: {len(path_d)}")

        # 频率控制：每10步重绘一次界面以提升流畅度
        if i % 10 == 0 or i == max_steps - 1:
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(delay)

    plt.ioff()
    plt.show()
