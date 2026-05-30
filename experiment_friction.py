
import numpy as np
import matplotlib.pyplot as plt
import math
from planning.motion_control import PIDController, RobotSimulator, path_to_trajectory, track_trajectory, get_trajectory_info

def create_test_trajectory():
    """创建一个标准的S型测试轨迹"""
    # 定义关键点：起点 -> 转弯1 -> 转弯2 -> 终点
    path = [
        (2, 2),
        (10, 2),
        (15, 10),
        (20, 10),
        (25, 5),
        (28, 5)
    ]
    # 生成平滑轨迹
    trajectory = path_to_trajectory(path, num_points=1000)
    return trajectory

def run_single_experiment(friction, trajectory):
    """运行单次PID仿真，返回性能指标"""
    # 1. 初始化机器人
    start_pos = trajectory[0]
    dx = trajectory[1][0] - start_pos[0]
    dy = trajectory[1][1] - start_pos[1]
    init_theta = math.atan2(dy, dx)
    
    # 初始位置无偏差，主要看打滑带来的动态误差
    robot = RobotSimulator(
        init_pos=start_pos,
        init_theta=init_theta,
        v=1.0,
        grid_map=None,
        friction=friction # 关键参数
    )
    
    # 2. 初始化控制器 (使用一组固定参数)
    pid = PIDController(
        kp_heading=2.0,
        kd_heading=0.5,
        kp_lateral=0.5,
        lateral_deadzone=0.0
    )
    
    # 3. 运行跟踪
    dt = 0.05
    max_steps = 1000
    errors = []
    
    for i in range(max_steps):
        # 寻找最近点
        dists = np.sqrt(np.sum((trajectory - np.array([robot.x, robot.y]))**2, axis=1))
        nearest_idx = np.argmin(dists)
        
        # 计算误差
        theta_path, e = get_trajectory_info(trajectory, (robot.x, robot.y), nearest_idx)
        errors.append(abs(e))
        
        # PID更新
        omega, v_correction = pid.update(theta_path, robot.theta, e, dt)
        
        # 机器人移动
        v_cmd = max(0.1, robot.v + v_correction)
        robot.move(omega, v_cmd, dt)
        
        # 终止条件
        if nearest_idx >= len(trajectory) - 5:
            break
            
    # 4. 计算指标
    max_error = np.max(errors) if errors else 0.0
    avg_error = np.mean(errors) if errors else 0.0
    
    return max_error, avg_error

def main():
    print("开始摩擦系数鲁棒性实验...")
    trajectory = create_test_trajectory()
    
    # 摩擦系数范围 (0.1: 冰面 ~ 1.0: 正常路面)
    frictions = np.linspace(0.1, 1.0, 10)
    
    max_errors = []
    avg_errors = []
    
    for f in frictions:
        max_e, avg_e = run_single_experiment(f, trajectory)
        max_errors.append(max_e)
        avg_errors.append(avg_e)
        print(f"摩擦系数: {f:.1f} -> MaxErr: {max_e:.3f}, AvgErr: {avg_e:.3f}")

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 图1：最大误差
    ax1.plot(frictions, max_errors, 'r-o', linewidth=2)
    ax1.set_title('Max Tracking Error vs Friction')
    ax1.set_xlabel('Friction Coefficient')
    ax1.set_ylabel('Max Lateral Error (m)')
    ax1.grid(True)
    ax1.invert_xaxis() # 反转X轴，符合直觉（从左到右摩擦变小，误差变大）
    
    # 图2：平均误差
    ax2.plot(frictions, avg_errors, 'b-s', linewidth=2)
    ax2.set_title('Avg Tracking Error vs Friction')
    ax2.set_xlabel('Friction Coefficient')
    ax2.set_ylabel('Avg Lateral Error (m)')
    ax2.grid(True)
    ax2.invert_xaxis()
    
    plt.tight_layout()
    plt.show()
    print("实验完成。")

if __name__ == "__main__":
    main()
