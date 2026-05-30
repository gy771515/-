import random
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import math

from map.grid_map import GridMap
from planning.Astar import astar, dijkstra
from visualization.DualCompare import animate_dual_search

def generate_random_map(width, height, density=0.25):
    """
    生成一个具有随机障碍物的地图
    """
    m = GridMap(width, height)
    num_obstacles = int(width * height * density)
    
    for _ in range(num_obstacles):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        m.add_obstacle(x, y)
    
    return m

def get_random_start_goal(grid_map, min_dist=0, max_dist=None):
    """
    随机选择一个起点和终点，支持距离限制
    """
    width, height = grid_map.width, grid_map.height
    if max_dist is None:
        max_dist = math.hypot(width, height)
    
    attempts = 0
    while attempts < 1000:
        # 随机选择起点
        start = (random.randint(0, width - 1), random.randint(0, height - 1))
        if not grid_map.is_free(start[0], start[1]):
            continue
            
        # 随机选择终点
        goal = (random.randint(0, width - 1), random.randint(0, height - 1))
        if not grid_map.is_free(goal[0], goal[1]) or goal == start:
            continue
            
        # 计算距离
        d = math.hypot(goal[0] - start[0], goal[1] - start[1])
        if min_dist <= d <= max_dist:
            return start, goal
        
        attempts += 1
    
    # 如果尝试多次未找到符合要求的，退而求其次
    print("[警告] 无法在限定尝试内找到符合距离要求的点位，已随机生成。")
    return start, goal

def main():
    print("="*60)
    print("  机器人路径规划算法并行动画对比系统 (A* vs Dijkstra)")
    print("="*60)
    
    # 默认设置
    width, height = 50, 50
    density = 0.25
    use_obstacles = True
    min_dist = 0
    max_dist = 100
    
    # 初始化地图
    m = generate_random_map(width, height, density if use_obstacles else 0)
    start, goal = get_random_start_goal(m, min_dist, max_dist)
    
    while True:
        print("\n" + "="*20 + " 当前配置 " + "="*20)
        print(f" - 地图规模: {width}x{height}")
        print(f" - 生成障碍物: {'是' if use_obstacles else '否'} (密度: {density if use_obstacles else 0})")
        print(f" - 点位距离限制: {min_dist} ~ {max_dist}")
        print(f" - 起点: {start}, 终点: {goal}, 实际距离: {math.hypot(goal[0]-start[0], goal[1]-start[1]):.2f}")
        print("-" * 50)
        print("请选择操作：")
        print("1. [并行对比] 同时运行 A* 与 Dijkstra")
        print("2. [单算法演示] 仅运行 A* 算法")
        print("3. [单算法演示] 仅运行 Dijkstra 算法")
        print("4. [配置修改] 设置障碍物与点位距离")
        print("5. [重新生成] 刷新地图和点位 (应用当前配置)")
        print("6. [退出程序]")
        choice = input("请输入选项 (1-6): ")
        
        if choice == '1':
            print("\n正在计算两种算法的路径...")
            t1 = time.perf_counter()
            path_a, visited_a = astar(m, start, goal)
            t2 = time.perf_counter()
            time_a = (t2 - t1) * 1000
            
            t3 = time.perf_counter()
            path_d, visited_d = dijkstra(m, start, goal)
            t4 = time.perf_counter()
            time_d = (t4 - t3) * 1000
            
            print(f"A* 完成: 耗时 {time_a:.2f}ms, 访问节点 {len(visited_a)}")
            print(f"Dijkstra 完成: 耗时 {time_d:.2f}ms, 访问节点 {len(visited_d)}")
            
            if path_a and path_d:
                animate_dual_search(m.grid, start, goal, visited_a, path_a, visited_d, path_d)
            else:
                print("[警告] 路径不可达，请重新生成地图或修改配置。")
                
        elif choice == '2':
            from visualization.DongHua import animate_search
            path, visited = astar(m, start, goal)
            if path: animate_search(m.grid, start, goal, visited, path, delay=0.001, title="A* Algorithm")
            else: print("[失败] 没找到路径。")
            
        elif choice == '3':
            from visualization.DongHua import animate_search
            path, visited = dijkstra(m, start, goal)
            if path: animate_search(m.grid, start, goal, visited, path, delay=0.001, title="Dijkstra Algorithm")
            else: print("[失败] 没找到路径。")
            
        elif choice == '4':
            print("\n--- 配置修改 ---")
            obs_choice = input("是否生成障碍物？(y/n, 默认y): ").lower()
            use_obstacles = False if obs_choice == 'n' else True
            
            try:
                min_dist = float(input(f"请输入起点到终点的最小距离 (当前 {min_dist}): ") or min_dist)
                max_dist = float(input(f"请输入起点到终点的最大距离 (当前 {max_dist}): ") or max_dist)
            except ValueError:
                print("输入无效，保持原值。")
            
            # 自动重新生成
            m = generate_random_map(width, height, density if use_obstacles else 0)
            start, goal = get_random_start_goal(m, min_dist, max_dist)
            print("[*] 配置已更新并重新生成地图。")

        elif choice == '5':
            m = generate_random_map(width, height, density if use_obstacles else 0)
            start, goal = get_random_start_goal(m, min_dist, max_dist)
            print(f"[*] 已重新生成。起点: {start}, 终点: {goal}")
            
        elif choice == '6':
            print("演示结束。")
            break
        else:
            print("无效输入。")

if __name__ == "__main__":
    main()
