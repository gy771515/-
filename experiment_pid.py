
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

def run_single_experiment(kp_h, kp_l, trajectory):
    """运行单次PID仿真，返回性能指标"""
    # 1. 初始化机器人
    start_pos = trajectory[0]
    # 初始朝向：指向轨迹的第二个点
    dx = trajectory[1][0] - start_pos[0]
    dy = trajectory[1][1] - start_pos[1]
    init_theta = math.atan2(dy, dx)
############################################################
    offset_dist = 0.5
############################################################
    init_x = start_pos[0] - offset_dist * math.sin(init_theta)
    init_y = start_pos[1] + offset_dist * math.cos(init_theta)

    robot = RobotSimulator(
        init_pos=(init_x, init_y),
        init_theta=init_theta,
        v=1.0,  # 恒定速度
        grid_map=None # PID测试不需要地图避障，只看轨迹跟踪性能
    )
    
    # 2. 初始化控制器
    pid = PIDController(
        kp_heading=kp_h,
        kd_heading=0.5, # 固定KD，主要分析KP的影响
        kp_lateral=kp_l,
        lateral_deadzone=0.0
    )
    
    # 3. 运行跟踪
    dt = 0.05
    max_steps = 1000
    errors = []
    omegas = [] # 记录角速度，用于衡量控制平滑度
    time_above_threshold = 0 
    threshold = 0.1 
    
    for i in range(max_steps):
        # 寻找最近点
        dists = np.sqrt(np.sum((trajectory - np.array([robot.x, robot.y]))**2, axis=1))
        nearest_idx = np.argmin(dists)
        
        # 计算误差
        theta_path, e = get_trajectory_info(trajectory, (robot.x, robot.y), nearest_idx)
        abs_e = abs(e)
        errors.append(abs_e)
        
        if abs_e > threshold:
            time_above_threshold += 1
        
        # PID更新
        omega, v_correction = pid.update(theta_path, robot.theta, e, dt)
        omegas.append(omega)
        
        # 机器人移动 (应用速度修正，但保持最小速度)
        v_cmd = max(0.1, robot.v + v_correction)
        robot.move(omega, v_cmd, dt)
        
        # 终止条件
        if nearest_idx >= len(trajectory) - 5:
            break
            
    # 4. 计算指标
    max_error = np.max(errors) if errors else 0.0
    avg_error = np.mean(errors) if errors else 0.0
    
    # --- 核心改进：多维惩罚项 ---
    omegas = np.array(omegas)
    if len(omegas) > 1:
        # 1. 控制抖动惩罚 (Total Variation)
        diff_omega = np.abs(np.diff(omegas))
        jitter_penalty = np.mean(diff_omega ** 2) * 10.0 # 使用平方，严厉惩罚高频震荡
        
        # 2. 控制能量惩罚 (Control Effort)
        # 过大的 omega 代表电机一直在剧烈转向，这也是一种代价
        effort_penalty = np.mean(omegas ** 2) * 2.0
        
        # 3. 震荡判定 (Oscillation Check)
        # 统计 omega 穿过零点的次数。正常平滑跟踪次数应很少。
        zero_crossings = np.where(np.diff(np.sign(omegas)))[0]
        oscillation_penalty = len(zero_crossings) * 0.05
    else:
        jitter_penalty = 0.0
        effort_penalty = 0.0
        oscillation_penalty = 0.0
        
    # 4. 增益正则化 (Gain Regularization)
    # 在工程上，我们倾向于使用尽可能小的增益来达到目标
    gain_reg = (kp_h**2 + kp_l**2) * 0.02
        
    # 综合代价函数 (Cost Function)
    # J = 权重 * 误差 + 权重 * 响应速度 + 权重 * 控制质量
    cost = (
        1.0 * max_error +         # 保证安全不撞墙
        0.5 * avg_error +         # 保证路径贴合度
        0.02 * time_above_threshold + # 保证收敛速度
        jitter_penalty +          # 严惩 S 形走位
        effort_penalty +          # 惩罚大力出奇迹
        oscillation_penalty +     # 惩罚反复横跳
        gain_reg                  # 倾向于更小的 Kp 
    )
    
    return max_error, avg_error, cost

def main():
    print("开始 PID 参数优化扫描实验（工程模式）...")
    trajectory = create_test_trajectory()
    
    # 参数范围：覆盖全区间，但增加密度
    kp_h_values = np.linspace(0.5, 4.0, 20) 
    kp_l_values = np.linspace(0.1, 2.0, 20) 
    
    # 结果网格
    X, Y = np.meshgrid(kp_h_values, kp_l_values)
    Z_max_err = np.zeros_like(X)
    Z_avg_err = np.zeros_like(X)
    Z_cost = np.zeros_like(X) 
    
    total_experiments = len(kp_h_values) * len(kp_l_values)
    count = 0
    
    for i in range(len(kp_l_values)):
        for j in range(len(kp_h_values)):
            kp_l = kp_l_values[i]
            kp_h = kp_h_values[j]
            
            max_e, avg_e, cost = run_single_experiment(kp_h, kp_l, trajectory)
            Z_max_err[i, j] = max_e
            Z_avg_err[i, j] = avg_e
            Z_cost[i, j] = cost
            
            count += 1
            if count % 20 == 0:
                print(f"进度: {count}/{total_experiments} | 当前搜索点: Kp_h={kp_h:.2f}, Kp_l={kp_l:.2f}")

    # 寻找最小值
    # 1. 寻找综合代价最小的参数组合 (智能推荐)
    min_cost_idx = np.unravel_index(np.argmin(Z_cost), Z_cost.shape)
    best_kp_h_opt = X[min_cost_idx]
    best_kp_l_opt = Y[min_cost_idx]
    min_cost_val = Z_cost[min_cost_idx]

    # 2. 寻找平均误差最小（虽然可能偏大）
    min_avg_idx = np.unravel_index(np.argmin(Z_avg_err), Z_avg_err.shape)
    best_kp_h_avg = X[min_avg_idx]
    best_kp_l_avg = Y[min_avg_idx]

    print("\n" + "="*60)
    print("【PID 参数工程优化报告】")
    print("-" * 60)
    print(f"★ 智能综合推荐 (工程稳定性优先):")
    print(f"   代价函数: J = Error + Jitter + Effort + Oscillation + GainReg")
    print(f"   综合代价 (Cost) = {min_cost_val:.4f}")
    print(f"   推荐参数: Kp_Heading={best_kp_h_opt:.2f}, Kp_Lateral={best_kp_l_opt:.2f}")
    print("-" * 60)
    print(f"   对比：若只看精度（不计代价）:")
    print(f"   Params: Kp_Heading={best_kp_h_avg:.2f}, Kp_Lateral={best_kp_l_avg:.2f}")
    print("="*60 + "\n")

    # 绘图
    fig = plt.figure(figsize=(18, 6))
    
    # 图1：最大误差热力图
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_max_err, cmap='viridis')
    ax1.set_title('Max Error (Safety)')
    ax1.set_xlabel('Kp Heading')
    ax1.set_ylabel('Kp Lateral')
    
    # 图2：平均误差热力图
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_avg_err, cmap='plasma')
    ax2.set_title('Avg Error (Precision)')
    ax2.set_xlabel('Kp Heading')
    ax2.set_ylabel('Kp Lateral')
    
    # 图3：综合代价热力图
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(X, Y, Z_cost, cmap='coolwarm')
    ax3.set_title('Total Cost (Overall Performance)')
    ax3.set_xlabel('Kp Heading')
    ax3.set_ylabel('Kp Lateral')
    
    plt.tight_layout()
    plt.show()
    
    # 图1：最大误差热力图
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_max_err, cmap='viridis')
    ax1.set_title('Max Lateral Error vs PID Gains')
    ax1.set_xlabel('Kp Heading')
    ax1.set_ylabel('Kp Lateral')
    ax1.set_zlabel('Max Error (m)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # 图2：平均误差热力图
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_avg_err, cmap='plasma')
    ax2.set_title('Average Lateral Error vs PID Gains')
    ax2.set_xlabel('Kp Heading')
    ax2.set_ylabel('Kp Lateral')
    ax2.set_zlabel('Avg Error (m)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show()
    print("实验完成，图表已生成。")

if __name__ == "__main__":
    main()
