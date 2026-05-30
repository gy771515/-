
import numpy as np
import matplotlib.pyplot as plt
import math
from planning.motion_control import PIDController, RobotSimulator, path_to_trajectory, track_trajectory, get_trajectory_info

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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

def run_experiment_with_logging(kp_h, kp_l, trajectory, offset_dist=0.7):
    """运行单次PID仿真，记录详细状态用于绘图"""
    # 1. 初始化机器人
    start_pos = trajectory[0]
    # 初始朝向：指向轨迹的第二个点
    dx = trajectory[1][0] - start_pos[0]
    dy = trajectory[1][1] - start_pos[1]
    init_theta = math.atan2(dy, dx)
    
    # 施加初始横向偏差
    init_x = start_pos[0] - offset_dist * math.sin(init_theta)
    init_y = start_pos[1] + offset_dist * math.cos(init_theta)

    robot = RobotSimulator(
        init_pos=(init_x, init_y),
        init_theta=init_theta,
        v=1.0,  # 恒定速度
        grid_map=None
    )
    
    # 2. 初始化控制器
    pid = PIDController(
        kp_heading=kp_h,
        kd_heading=0.5,
        kp_lateral=kp_l,
        lateral_deadzone=0.0
    )
    
    # 3. 运行跟踪
    dt = 0.05
    max_steps = 1000
    
    # 记录数据
    time_log = []
    lat_error_log = []
    traj_x = []
    traj_y = []
    
    for i in range(max_steps):
        current_time = i * dt
        
        # 寻找最近点
        dists = np.sqrt(np.sum((trajectory - np.array([robot.x, robot.y]))**2, axis=1))
        nearest_idx = np.argmin(dists)
        
        # 计算误差
        theta_path, e = get_trajectory_info(trajectory, (robot.x, robot.y), nearest_idx)
        
        # 记录
        time_log.append(current_time)
        lat_error_log.append(e) # 保留符号，观察震荡
        traj_x.append(robot.x)
        traj_y.append(robot.y)
        
        # PID更新
        omega, v_correction = pid.update(theta_path, robot.theta, e, dt)
        
        # 机器人移动
        v_cmd = max(0.1, robot.v + v_correction)
        robot.move(omega, v_cmd, dt)
        
        # 终止条件
        if nearest_idx >= len(trajectory) - 5:
            break
            
    return time_log, lat_error_log, traj_x, traj_y

def main():
    print("开始生成初始偏差收敛性对比图...")
    trajectory = create_test_trajectory()
    
    # 定义三组对比参数
    # A组 (低增益): 反应慢
    params_A = {'kp_h': 1.5, 'kp_l': 0.5, 'label': 'A组 (低增益)'}
    # B组 (高增益): 震荡
    params_B = {'kp_h': 6.0, 'kp_l': 4.0, 'label': 'B组 (高增益)'}
    # C组 (优化): 最佳
    params_C = {'kp_h': 3.5, 'kp_l': 1.5, 'label': 'C组 (优化参数)'}
    
    results = {}
    
    # 更改执行和绘图顺序，让B组最后画，防止被C组遮盖
    for p in [params_A, params_C, params_B]:
        t, e, x, y = run_experiment_with_logging(p['kp_h'], p['kp_l'], trajectory)
        results[p['label']] = {'time': t, 'error': e, 'x': x, 'y': y}
        print(f"完成 {p['label']} 仿真")

    # === 绘图 ===
    fig = plt.figure(figsize=(12, 10))
    
    # 子图1: 轨迹对比 (XY平面)
    ax1 = plt.subplot(2, 1, 1)
    
    # 绘制参考轨迹
    ref_x = trajectory[:, 0]
    ref_y = trajectory[:, 1]
    ax1.plot(ref_x, ref_y, 'k--', linewidth=2, label='参考轨迹 (Reference)')
    
    # 绘制三组实际轨迹
    styles = {'A组 (低增益)': 'b-.', 'C组 (优化参数)': 'g-', 'B组 (高增益)': 'r:'}
    
    for label, data in results.items():
        # 只画前半段，看收敛过程更清晰
        limit_idx = min(len(data['x']), 600) 
        ax1.plot(data['x'][:limit_idx], data['y'][:limit_idx], styles[label], linewidth=2, label=label)
        
    ax1.set_title('不同参数下的轨迹收敛过程对比', fontsize=14)
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.axis('equal')
    
    # 子图2: 横向误差随时间变化
    ax2 = plt.subplot(2, 1, 2)
    
    for label, data in results.items():
        # 同样只画前15秒，重点看收敛
        limit_idx = min(len(data['time']), 300) 
        t_data = data['time'][:limit_idx]
        e_data = data['error'][:limit_idx]
        ax2.plot(t_data, e_data, styles[label], linewidth=2, label=label)
        
    # 绘制稳态阈值线
    ax2.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=-0.1, color='gray', linestyle='--', alpha=0.5, label='稳态阈值 (±0.1m)')
    
    ax2.set_title('横向误差收敛曲线 (Lateral Error Convergence)', fontsize=14)
    ax2.set_xlabel('时间 Time [s]')
    ax2.set_ylabel('横向误差 Lateral Error [m]')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    print("图表已生成。")

if __name__ == "__main__":
    main()
