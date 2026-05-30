
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from map.grid_map import GridMap
from planning.Astar import astar
from planning.motion_control import PIDController, RobotSimulator, path_to_trajectory, track_trajectory

def run_single_robustness_test(test_id, density=0.1, force_path=False):
    """
    运行单次鲁棒性测试
    :param force_path: 是否强制打通路径（排除地图连通性干扰，专注于测试算法性能）
    :return: (is_success, path_len, exec_time, error_msg)
    """
    # 1. 初始化随机地图
    grid_map = GridMap(30, 30)
    grid_map.generate_random_obstacles(density=density, seed=test_id + 1000)
    
    # 2. 随机生成起终点
    def get_random_free_point():
        for _ in range(100):
            x = random.randint(0, 29)
            y = random.randint(0, 29)
            if grid_map.is_free(x, y):
                return (x, y)
        return None

    start = get_random_free_point()
    goal = get_random_free_point()
    
    if not start or not goal:
        return False, 0, 0, "无法生成有效的起终点"
        
    if np.hypot(start[0]-goal[0], start[1]-goal[1]) < 10:
         return False, 0, 0, "起终点距离过近"

    # 新增：如果开启强制打通，则在规划前先确保路径连通
    if force_path:
        grid_map.ensure_path_exists(start, goal)

    # 3. 路径规划 (A*)
    t0 = time.time()
    path, visited = astar(grid_map, start, goal, neighbor_type='8')
    t1 = time.time()
    
    if not path:
        return False, 0, (t1-t0)*1000, "路径规划失败（不可达）"
        
    # 4. 轨迹生成与跟踪
    try:
        path_center = [(p[0]+0.5, p[1]+0.5) for p in path]
        trajectory = path_to_trajectory(path_center, grid_map=grid_map) # 传入 grid_map 以便碰撞修正
        
        robot = RobotSimulator(
            init_pos=(start[0]+0.5, start[1]+0.5),
            init_theta=0.0,
            v=0.5,
            grid_map=grid_map
        )
        pid = PIDController()
        
        # 跟踪过程
        actual_traj = track_trajectory(trajectory, robot, pid, dt=0.1, max_steps=1500)
        
        # 检查是否因为碰撞停止
        if robot.is_collided:
            return False, len(path), (t1-t0)*1000, "跟踪失败（发生碰撞）"

        # 检查最终距离
        final_pos = np.array([robot.x, robot.y])
        target_pos = trajectory[-1]
        dist_error = np.linalg.norm(final_pos - target_pos)
        
        if dist_error > 1.5:
            return False, len(path), (t1-t0)*1000, f"跟踪失败（精度不足: {dist_error:.2f}）"
            
    except Exception as e:
        return False, len(path), (t1-t0)*1000, f"异常: {str(e)}"

    return True, len(path), (t1-t0)*1000, "Success"

def main():
    print("\n" + "="*50)
    print("  机器人路径规划算法鲁棒性批量测试工具")
    print("="*50)
    
    try:
        user_input = input("[*] 请输入测试密度 (例如 0.3，默认 0.2): ").strip()
        density = float(user_input) if user_input else 0.2
        
        force_input = input("[*] 是否强制打通路径？(y/n，默认 n): ").strip().lower()
        force_path = True if force_input == 'y' else False
        
        num_tests = int(input("[*] 请输入测试次数 (默认 50): ").strip() or 50)
    except ValueError:
        print("输入无效，使用默认设置。")
        density, force_path, num_tests = 0.2, False, 50

    print(f"\n开始测试: 密度={density}, 强制连通={force_path}, 次数={num_tests}")
    
    success_count = 0
    fail_reasons = {} # 统计失败原因
    total_time = 0
    valid_tests = 0
    
    for i in range(num_tests):
        success, length, t_exec, msg = run_single_robustness_test(i, density, force_path)
        
        if "距离过近" in msg or "无效的起终点" in msg:
            continue
            
        valid_tests += 1
        if success:
            success_count += 1
            total_time += t_exec
        else:
            fail_reasons[msg] = fail_reasons.get(msg, 0) + 1
        
        # 实时打印进度
        status = "PASS" if success else "FAIL"
        print(f"Test {i+1:02d}: [{status}] {msg}")
        
    # 统计结果报告
    if valid_tests > 0:
        success_rate = (success_count / valid_tests) * 100
        print("\n" + "-"*50)
        print("【鲁棒性测试实验报告】")
        print(f" - 测试环境密度: {density}")
        print(f" - 有效样本总数: {valid_tests}")
        print(f" - 总体成功率: {success_rate:.1f}%")
        print(f" - 平均规划耗时: {total_time/success_count:.2f} ms" if success_count > 0 else " - 平均规划耗时: N/A")
        
        print("\n[失败原因分布统计]:")
        for reason, count in fail_reasons.items():
            print(f"   - {reason}: {count} 次 ({(count/valid_tests)*100:.1f}%)")
        print("-"*50 + "\n")
    else:
        print("\n[错误] 未能完成有效测试。")

if __name__ == "__main__":
    main()
