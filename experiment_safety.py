import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# 将当前目录加入 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from map.grid_map import GridMap
from planning.Astar import astar

def create_narrow_passage_map():
    """创建一个超大尺寸地图(100x100)，包含多级宽度的漏斗状通道"""
    m = GridMap(100, 100)
    
    # 1. 外部大边界
    m.add_obstacle_block(0, 0, 99, 0)
    m.add_obstacle_block(0, 99, 99, 99)
    m.add_obstacle_block(0, 0, 0, 99)
    m.add_obstacle_block(99, 0, 99, 99)

    # 2. 设计一个巨大的“漏斗隔断” (y=45 到 y=55 之间)
    # 这个隔断留出了三个不同等级的开口：
    # 开口1 (右侧极窄): x=85-87 (2格宽，捷径)
    # 开口2 (中间一般): x=45-55 (10格宽)
    # 开口3 (左侧极宽): x=5-35 (30格宽)
    
    # 构造隔断墙
    m.add_obstacle_block(1, 45, 4, 55)   # 左边角
    m.add_obstacle_block(36, 45, 44, 55) # 开口3和开口2之间
    m.add_obstacle_block(56, 45, 84, 55) # 开口2和开口1之间
    m.add_obstacle_block(88, 45, 98, 55) # 右边角
    
    # 3. 增加一些“诱导墙”，让路径在低权重时极度倾向于右侧
    # 底部诱导：引导向右
    m.add_obstacle_block(10, 10, 70, 15) 
    # 顶部诱导：引导向右
    m.add_obstacle_block(10, 85, 70, 90)
    
    return m

def run_safety_experiment():
    print("="*60)
    print("  避障安全权重对路径质量影响实验 (V5.0 - 超大场景 & 拓扑级变化)")
    print("="*60)
    
    m = create_narrow_passage_map()
    # 起终点拉远，跨越整个 100x100 地图
    start = (86, 5)
    goal = (86, 95)
    
    # 调整权重分布，观察拓扑跳变
    test_configs = [
        {'w': 0.0,   'dist': 12.0},  # 红：极速
        {'w': 20.0,  'dist': 12.0},  # 蓝：开始远离墙壁
        {'w': 100.0, 'dist': 12.0},  # 紫：尝试中间通道
        {'w': 1000.0, 'dist': 12.0}  # 绿：极安，必须走左侧最宽道
    ]
    results = []
    successful_paths = [] 

    print(f"[*] 实验配置: 地图 100x100, 起点{start}, 终点{goal}")
    print(f"{'Weight':<10} | {'SafeDist':<10} | {'Path Len':<12} | {'Min Dist(m)':<12}")
    print("-" * 65)

    for cfg in test_configs:
        w = cfg['w']
        s_dist = cfg['dist']
        m._dist_dirty = True
        m.update_distance_map()
        
        t0 = time.perf_counter()
        # 传入动态的 safe_dist
        path, _ = astar(m, start, goal, safety_weight=w, safe_dist=s_dist)
        t1 = time.perf_counter()
        
        if path:
            # 计算指标
            path_arr = np.array(path)
            length = 0
            for i in range(len(path)-1):
                length += np.linalg.norm(path_arr[i+1] - path_arr[i])
            
            dist_map = m.update_distance_map()
            min_dist = float('inf')
            for p in path:
                gx, gy = int(round(p[0])), int(round(p[1]))
                d = dist_map[gy, gx]
                if d < min_dist: min_dist = d
            
            results.append({
                'weight': w,
                'length': length,
                'min_dist': min_dist,
                'time': (t1-t0)*1000
            })
            successful_paths.append((path, w))
            print(f"{w:<10.1f} | {s_dist:<10.1f} | {length:<12.2f} | {min_dist:<12.2f}")
        else:
            print(f"{w:<10.1f} | {s_dist:<10.1f} | {'FAILED':<12} | {'-':<12}")

    # 可视化对比
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    m.plot(ax)
    
    # 样式设计（针对4个权重）：
    # 颜色语义优化：红(0)-极速, 蓝(20)-平衡, 紫(100)-审慎, 绿(1000)-极安
    color_map = {
        0.0:    ('#FF0000', 1.5, '-', 15),    # 红：最细，实线，最高层
        20.0:   ('#0072BD', 3.0, '--', 14),   # 蓝：稍粗，虚线
        100.0:  ('#7E2F8E', 5.0, '-.', 13),   # 紫：更粗，点划线
        1000.0: ('#2CA02C', 8.0, (0, (5, 2)), 12) # 绿：最粗，长虚线，最底层
    }
    
    # 绘图逻辑：
    # 1. 显式设置 zorder，确保即使绘图顺序改变，视觉叠加效果依然是“细线在上，粗线在下”
    # 2. 按照权重【升序】排列绘制，使图例顺序符合直觉（从 0 到 30）
    successful_paths.sort(key=lambda x: x[1])

    for path, w in successful_paths:
        path_arr = np.array(path)
        color, lw, ls, zo = color_map.get(w, ('black', 1.0, '-', 10))
        
        ax.plot(path_arr[:, 0] + 0.5, path_arr[:, 1] + 0.5, 
                label=f"Weight={w:.1f} (Safety)", 
                color=color, 
                linewidth=lw, 
                linestyle=ls,
                alpha=0.8,
                zorder=zo)
    
    ax.scatter(start[0]+0.5, start[1]+0.5, c='blue', s=150, marker='o', label='Start', edgecolors='white', zorder=20)
    ax.scatter(goal[0]+0.5, goal[1]+0.5, c='orange', s=200, marker='*', label='Goal', edgecolors='white', zorder=20)
    
    ax.set_title("Robot Path Planning: Safety Weight Comparison (A* Algorithm)", fontsize=14)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    plt.tight_layout()

    print("\n[*] 实验完成。")


    plt.show()

if __name__ == "__main__":
    run_safety_experiment()
