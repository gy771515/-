import time
import numpy as np
import matplotlib.pyplot as plt
from map.grid_map import GridMap
from planning.Astar import astar, dijkstra
from visualization.KeShi import visualize, visualize_trajectory
from planning.motion_control import PIDController, RobotSimulator, path_to_trajectory, track_trajectory

def create_custom_map():
    """在此函数中自定义障碍物位置"""
    # 1. 初始化地图（尺寸 30x30）
    grid_map = GridMap(30, 30)

    # 2. 添加障碍物

    # grid_map.add_obstacle_block(10, 10, 11, 30)
    # grid_map.add_obstacle_block(15, 0, 16, 20)
    # grid_map.add_obstacle_block(5, 0, 6, 20)

    # grid_map.add_obstacle(5, 5)
    # grid_map.add_obstacle(6, 5)
    # grid_map.add_obstacle(7, 5)

    grid_map.add_obstacle_block(7, 4, 13, 10)
    # grid_map.add_obstacle(25, 25)

    return grid_map

def check_point_valid(grid_map, point, name="point"):
    """校验点是否在地图内且非障碍"""
    if not grid_map.in_bounds(*point):
        raise ValueError(f"{name} {point} 超出地图范围（地图尺寸：{grid_map.width}x{grid_map.height}）")
    if not grid_map.is_free(*point):
        raise ValueError(f"{name} {point} 位于障碍物上，无法作为起点/终点")

def run_planning(algo_func, grid_map, start, goal, algo_name):
    """运行规划算法并统计性能"""
    print(f"\n正在运行 {algo_name} 算法...")
    start_time = time.time()
    
    # 统一调用接口
    path, visited = algo_func(
        grid_map, 
        start, 
        goal, 
        neighbor_type='8'
    )
    
    end_time = time.time()
    exec_time = (end_time - start_time) * 1000  # ms
    
    if path is None:
        print(f"错误：{algo_name} 未找到可行路径！")
        return None, None, 0
        
    path_len = len(path)
    visited_count = len(visited)
    print(f"{algo_name} 完成：耗时 {exec_time:.2f}ms | 路径长度 {path_len} | 搜索节点 {visited_count}")
    
    return path, visited, exec_time

def run_full_simulation(algo_name):
    """运行完整的单算法仿真流程（规划->轨迹->控制）"""
    # 1. 初始化地图（使用自定义地图配置）
    grid_map = create_custom_map()
    grid = grid_map.to_array()
    start = (2, 2)
    goal = (20, 10)

    try:
        check_point_valid(grid_map, start, "起点")
        check_point_valid(grid_map, goal, "终点")
    except ValueError as e:
        print(f"初始化失败：{e}")
        return

    # 2. 选择算法函数
    if algo_name == 'A*':
        algo_func = astar
    elif algo_name == 'Dijkstra':
        algo_func = dijkstra
    else:
        print("未知算法")
        return

    # 3. 运行规划
    path, visited, _ = run_planning(algo_func, grid_map, start, goal, algo_name)
    if path is None:
        return

    # 4. 可视化离散路径
    visualize(
        grid=grid,
        start=start,
        end=goal,
        path=path,
        visited=visited,
        title=f"{algo_name} Path Planning (Visited: {len(visited)})"
    )

    # 5. 生成平滑轨迹
    path_center = [(p[0]+0.5, p[1]+0.5) for p in path]
    trajectory = path_to_trajectory(path_center, num_points=500)
    
    # 6. 初始化机器人与PID
    robot = RobotSimulator(
        init_pos=(start[0]+0.5, start[1]+0.5),
        init_theta=0.0,
        v=0.3,
        grid_map=grid_map
    )
    
    pid = PIDController(
        kp_heading=1.2,
        kd_heading=0.5,
        kp_lateral=0.2,
        lateral_deadzone=0.2
    )

    # 7. 轨迹跟踪
    print(f"\n开始 {algo_name} 轨迹跟踪仿真...")
    actual_traj = track_trajectory(trajectory, robot, pid, dt=0.01)

    # 8. 输出跟踪结果
    final_pos = (round(robot.x, 2), round(robot.y, 2))
    target_pos = (round(trajectory[-1][0], 2), round(trajectory[-1][1], 2))
    end_error = round(np.hypot(robot.x - trajectory[-1][0], robot.y - trajectory[-1][1]), 3)
    
    print(f"机器人最终位置：{final_pos}")
    print(f"终点距离误差：{end_error} 栅格")

    # 9. 可视化跟踪结果
    visualize_trajectory(
        grid=grid,
        start=start,
        end=goal,
        expected_traj=trajectory,
        actual_traj=actual_traj,
        title=f"{algo_name} PID Tracking Simulation"
    )

def compare_algorithms():
    """对比A*和Dijkstra算法性能与搜索空间"""
    # 1. 初始化（使用自定义地图配置）
    grid_map = create_custom_map()
    grid = grid_map.to_array()
    start = (2, 2)
    goal = (20, 10) # 使用稍远的终点以凸显差异
    
    print("\n========== 算法性能对比 ==========")
    print(f"{'算法':<10} | {'耗时(ms)':<10} | {'路径长度':<10} | {'搜索节点数':<10}")
    print("-" * 50)

    # 2. 运行A*
    path_astar, visited_astar, time_astar = run_planning(astar, grid_map, start, goal, "A*")
    print(f"{'A*':<10} | {time_astar:<10.2f} | {len(path_astar):<10} | {len(visited_astar):<10}")

    # 3. 运行Dijkstra
    path_dijk, visited_dijk, time_dijk = run_planning(dijkstra, grid_map, start, goal, "Dijkstra")
    print(f"{'Dijkstra':<10} | {time_dijk:<10.2f} | {len(path_dijk):<10} | {len(visited_dijk):<10}")
    print("-" * 50)
    print("注：A*利用启发式函数向目标搜索，搜索节点更少；Dijkstra向四周均匀扩散，保证最短路径但搜索量大。")

    # 4. 绘制对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 辅助绘图函数
    def plot_algo(ax, path, visited, title):
        rows, cols = grid.shape
        ax.imshow(grid, cmap='gray_r', origin="upper", extent=[0, cols, rows, 0])
        # 绘制访问过的节点（搜索空间）
        if visited:
            vx = [v[0]+0.5 for v in visited]
            vy = [v[1]+0.5 for v in visited]
            ax.scatter(vx, vy, c='cyan', s=10, alpha=0.5, label='Visited Nodes')
        # 绘制路径
        if path:
            px = [p[0]+0.5 for p in path]
            py = [p[1]+0.5 for p in path]
            ax.plot(px, py, 'r-', linewidth=2, label='Path')
        # 起终点
        ax.plot(start[0]+0.5, start[1]+0.5, 'go', markersize=10, label='Start')
        ax.plot(goal[0]+0.5, goal[1]+0.5, 'b*', markersize=12, label='Goal')
        
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim(0, rows) # 确保Y轴方向正确
        ax.invert_yaxis()

    plot_algo(ax1, path_astar, visited_astar, f"A* Algorithm\nNodes: {len(visited_astar)}, Path: {len(path_astar)}")
    plot_algo(ax2, path_dijk, visited_dijk, f"Dijkstra Algorithm\nNodes: {len(visited_dijk)}, Path: {len(path_dijk)}")
    
    plt.tight_layout()
    plt.show()

def main():
    while True:
        print("\n========== 机器人路径规划算法演示 ==========")
        print("1. 运行 A* 算法 (含PID仿真)")
        print("2. 运行 Dijkstra 算法 (含PID仿真)")
        print("3. 对比 A* 与 Dijkstra (搜索空间可视化)")
        print("0. 退出")
        
        choice = input("请输入选项 (0-3): ").strip()
        
        if choice == '1':
            run_full_simulation('A*')
        elif choice == '2':
            run_full_simulation('Dijkstra')
        elif choice == '3':
            compare_algorithms()
        elif choice == '0':
            print("退出程序。")
            break
        else:
            print("无效选项，请重新输入。")

if __name__ == "__main__":
    main()
