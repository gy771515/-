#motion_control.py
import math
import numpy as np
from scipy.interpolate import splprep, splev

from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d

def path_to_trajectory(path, grid_map=None, num_points=500, degree=3):
    """
    重构版：先生成插值路径，应用高斯平滑，再进行轻量化的碰撞修正。
    这种方式能有效避免 B 样条与原始路径硬切换导致的轨迹“乱套”/抖动问题。
    """
    if len(path) < 2:
        return np.array(path)

    # 1. 线性插值生成基础轨迹（保证路径点数充足）
    path_arr = np.array(path)
    t = np.linspace(0, 1, len(path))
    t_new = np.linspace(0, 1, num_points)
    x_interp = np.interp(t_new, t, path_arr[:, 0])
    y_interp = np.interp(t_new, t, path_arr[:, 1])
    base_traj = np.array([x_interp, y_interp]).T

    # 2. 应用高斯平滑（sigma决定平滑程度，目前设为3.0-5.0左右比较平衡）
    # 高斯平滑比 B 样条更稳定，不容易在转角处产生巨大的曲率波动
    sigma = 3.0
    x_smooth = gaussian_filter1d(x_interp, sigma=sigma)
    y_smooth = gaussian_filter1d(y_interp, sigma=sigma)
    smooth_traj = np.array([x_smooth, y_smooth]).T

    # 3. 碰撞修正逻辑
    if grid_map is not None:
        final_traj = []
        for i in range(num_points):
            p_s = smooth_traj[i]
            p_b = base_traj[i] # 原始线性路径上的点（已知安全）
            
            px, py = p_s
            gx, gy = int(math.floor(px)), int(math.floor(py))
            
            # 如果平滑点落入障碍物，将其向原始安全路径方向“拉回”
            # 我们不直接替换，而是寻找中间位置，保持轨迹连贯性
            if not grid_map.in_bounds(gx, gy) or grid_map.grid[gy, gx] == 1:
                # 寻找平滑点与原始点之间的中点，直到不碰撞为止
                # 这种“弹性拉回”比直接替换要平滑得多
                step = 0
                temp_p = p_s
                while (not grid_map.is_free(int(math.floor(temp_p[0])), int(math.floor(temp_p[1])))) and step < 5:
                    temp_p = 0.5 * (temp_p + p_b) # 向原始点靠近 50%
                    step += 1
                final_traj.append(temp_p)
            else:
                final_traj.append(p_s)
        return np.array(final_traj)
        
    return smooth_traj

def path_to_trajectory_linear(path, num_points=500):
    """
    将离散路径点通过线性插值生成轨迹。
    """
    if len(path) < 2:
        return np.array(path)
    
    path = np.array(path)
    x = path[:, 0]
    y = path[:, 1]
    
    t = np.linspace(0, 1, len(path))
    t_new = np.linspace(0, 1, num_points)
    
    x_new = np.interp(t_new, t, x)
    y_new = np.interp(t_new, t, y)
    
    return np.array([x_new, y_new]).T


def get_trajectory_info(trajectory, current_pos, nearest_idx):
    """
    修正版：返回轨迹切线方向(theta_path)和带符号横向误差(e)
    :param trajectory: 期望轨迹点列表 [(x1,y1), (x2,y2), ...]
    :param current_pos: 当前位置 (x, y)
    :param nearest_idx: 轨迹上最近点的索引
    :return: theta_path (轨迹切线方向), e (带符号横向误差)
    """
    # 处理边界：如果是最后一个点，取前一个点组成线段
    if nearest_idx >= len(trajectory) - 1:
        nearest_idx = len(trajectory) - 2
    p1 = trajectory[nearest_idx]
    p2 = trajectory[nearest_idx + 1]
    x, y = current_pos
    x1, y1 = p1
    x2, y2 = p2

    # 1. 计算当前位置到线段p1p2的投影点
    dx = x2 - x1
    dy = y2 - y1
    dx1 = x - x1
    dy1 = y - y1
    t = (dx1 * dx + dy1 * dy) / (dx*dx + dy*dy + 1e-6)  # 投影长度比例
    t = max(0.0, min(1.0, t))  # 限制在线段内
    px = x1 + t * dx
    py = y1 + t * dy

    # 2. 带符号横向误差：叉乘判断在轨迹左侧/右侧
    cross = dx * (y - y1) - dy * (x - x1)
    e = math.hypot(x - px, y - py) * (1 if cross > 0 else -1)

    # 3. 轨迹切线方向（线段p1p2的方向）
    theta_path = math.atan2(dy, dx)

    return theta_path, e

class PIDController:

    def __init__(self, kp_heading=1.8, kd_heading=0.5, kp_lateral=0.4, lateral_deadzone=0.1):
        self.kp_h = kp_heading  # 航向环比例系数
        self.kd_h = kd_heading  # 航向环微分系数
        self.kp_l = kp_lateral  # 横向环比例系数
        self.deadzone = lateral_deadzone  # 横向误差死区
        self.last_heading_error = 0.0  # 上一时刻航向误差（微分用）

    def update(self, target_heading, current_heading, lateral_error, dt=0.01, high_gain=False):
        """
        PID参数更新
        :param target_heading: 目标航向角（弧度）
        :param current_heading: 当前航向角（弧度）
        :param lateral_error: 横向误差（栅格单位）
        :param dt: 时间步长
        :param high_gain: 是否高增益模式（转弯/接近终点时）
        :return: 角速度修正值，线速度修正值
        """
        # 航向误差计算（归一化到[-π, π]）
        heading_error = target_heading - current_heading
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        lateral_error = 0.0 if abs(lateral_error) < self.deadzone else lateral_error
        
        # 1. 外环：横向误差计算出航向补偿角
        # e > 0 时（机器人位于轨迹左侧），需要右转（减小航向角）向轨迹靠拢
        heading_offset = -self.kp_l * lateral_error
        # 限制最大偏航补偿角
        heading_offset = np.clip(heading_offset, -math.pi/3, math.pi/3)

        # 2. 内环：计算带有补偿的最终航向误差
        adjusted_target_heading = target_heading + heading_offset
        heading_error = adjusted_target_heading - current_heading
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        # 航向环PID计算（P+D）
        kp = self.kp_h * 1.5 if high_gain else self.kp_h
        kd = self.kd_h * 1.5 if high_gain else self.kd_h
        omega_pid = kp * heading_error + kd * (heading_error - self.last_heading_error) / dt

        v_correction = -0.3 * abs(lateral_error)

        # 更新上一时刻误差
        self.last_heading_error = heading_error

        return omega_pid, v_correction



class RobotSimulator:
    """移动机器人仿真类，包含运动控制和避障检测"""
    def __init__(self, init_pos, init_theta=0.0, v=0.5, grid_map=None, tau_v=0.0, tau_omega=0.0, max_v=None, max_omega=None, friction=1.0):
        self.x, self.y = init_pos
        self.theta = init_theta
        self.v = v
        self.grid_map = grid_map
        self.tau_v = tau_v
        self.tau_omega = tau_omega
        self.max_v = max_v
        self.max_omega = max_omega
        self.friction = friction  # 摩擦系数 (0.0 ~ 1.0)
        self.v_actual = v
        self.omega_actual = 0.0
        self.collision_count = 0
        self.last_collision = False
        self.is_collided = False # 新增属性，用于兼容 robustness 测试脚本
        
        # 新增：实际运动速度矢量 (vx, vy)，用于模拟惯性漂移
        self.vx_global = v * math.cos(init_theta)
        self.vy_global = v * math.sin(init_theta)

    def move(self, omega, v, dt, count_collision=True):
        self.last_collision = False

        if self.max_v is not None:
            v = float(np.clip(v, -self.max_v, self.max_v))
        if self.max_omega is not None:
            omega = float(np.clip(omega, -self.max_omega, self.max_omega))

        if dt <= 0:
            return False

        step_len = 0.05
        dist_est = max(abs(v), math.hypot(self.vx_global, self.vy_global)) * dt
        n_steps = max(1, int(math.ceil(dist_est / step_len))) if step_len > 0 else 1
        n_steps = int(np.clip(n_steps, 1, 80))
        dt_sub = dt / n_steps

        x_safe, y_safe = self.x, self.y
        theta_safe = self.theta
        vx_safe, vy_safe = self.vx_global, self.vy_global
        v_actual_safe, omega_actual_safe = self.v_actual, self.omega_actual

        collided = False
        for _ in range(n_steps):
            effective_omega = omega * (0.2 + 0.8 * self.friction)
            self.theta += effective_omega * dt_sub
            self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

            target_vx = v * math.cos(self.theta)
            target_vy = v * math.sin(self.theta)

            alpha = self.friction * 0.2
            self.vx_global += (target_vx - self.vx_global) * alpha
            self.vy_global += (target_vy - self.vy_global) * alpha

            self.x += self.vx_global * dt_sub
            self.y += self.vy_global * dt_sub
            self.v_actual = math.hypot(self.vx_global, self.vy_global)
            self.omega_actual = effective_omega

            if self.grid_map is not None and self._is_occupied(self.x, self.y):
                collided = True
                break
            x_safe, y_safe = self.x, self.y
            theta_safe = self.theta
            vx_safe, vy_safe = self.vx_global, self.vy_global
            v_actual_safe, omega_actual_safe = self.v_actual, self.omega_actual

        if collided:
            self.x, self.y = x_safe, y_safe
            self.theta = theta_safe
            self.vx_global, self.vy_global = 0.0, 0.0
            self.v_actual, self.omega_actual = 0.0, 0.0
            self.is_collided = True # 标记已碰撞
            if count_collision:
                self.last_collision = True
                self.collision_count += 1
            return True

        return False

    def _is_occupied(self, x, y):
        if self.grid_map is None:
            return False
        width = int(getattr(self.grid_map, "width", 30))
        height = int(getattr(self.grid_map, "height", 30))
        gx = int(math.floor(x))
        gy = int(math.floor(y))
        if gx < 0 or gx >= width or gy < 0 or gy >= height:
            return True
        return int(self.grid_map.grid[gy, gx]) == 1

    def is_collision(self, look_ahead=1.0):
        """
        检测机器人前方是否有障碍物
        返回：has_obstacle (bool), obstacle_dir (str), obs_dist (float)
        """
        # 1. 边界保护：先判断look_ahead是否合法
        if look_ahead <= 0:
            return False, "", 0.0

        # 2. 计算前方检测点（增加边界裁剪，防止越界）
        check_x = self.x + look_ahead * math.cos(self.theta)
        check_y = self.y + look_ahead * math.sin(self.theta)
        width = int(getattr(self.grid_map, "width", 30)) if self.grid_map is not None else 30
        height = int(getattr(self.grid_map, "height", 30)) if self.grid_map is not None else 30
        check_x = np.clip(check_x, 0, max(0, width - 1))
        check_y = np.clip(check_y, 0, max(0, height - 1))

        # 3. 转成栅格坐标（强制裁剪）
        gx = int(math.floor(check_x))
        gy = int(math.floor(check_y))
        gx = int(np.clip(gx, 0, max(0, width - 1)))
        gy = int(np.clip(gy, 0, max(0, height - 1)))

        # 4. 边界保护（简化判断）
        if gx < 0 or gx >= width or gy < 0 or gy >= height:
            return True, "out", look_ahead

        # 5. 判断是否是障碍（增加grid_map非空判断）
        if self.grid_map is None:
            return False, "", 0.0
        if self.grid_map.grid[gy, gx] == 1:
            # 计算障碍方向（完善所有方向判断，避免无匹配）
            dx = gx - int(round(self.x))
            dy = gy - int(round(self.y))
            dir_str = "unknown"  # 默认值，避免空字符串
            if abs(dx) > abs(dy):
                dir_str = "right" if dx > 0 else "left"
            else:
                dir_str = "up" if dy > 0 else "down"
            # 增加距离保护，避免除以0
            dist = math.hypot(check_x - self.x, check_y - self.y) if (
                        check_x != self.x or check_y != self.y) else look_ahead
            return True, dir_str, dist

        # 无障碍物
        return False, "", 0.0

def track_trajectory(trajectory, robot, pid_controller, dt=0.1, max_steps=1000):
    """
    使用PID控制器跟踪给定轨迹。
    :param trajectory: 期望轨迹点列表
    :param robot: RobotSimulator对象
    :param pid_controller: PIDController对象
    :param dt: 时间步长
    :param max_steps: 最大仿真步数
    :return: 实际轨迹点列表
    """
    actual_path = []
    for i in range(max_steps):
        # 寻找轨迹上最近的点
        dist = np.sqrt(np.sum((trajectory - np.array([robot.x, robot.y]))**2, axis=1))
        nearest_idx = np.argmin(dist)

        # 计算轨迹切线和横向误差
        theta_path, e = get_trajectory_info(trajectory, (robot.x, robot.y), nearest_idx)

        # PID计算
        omega, _ = pid_controller.update(theta_path, robot.theta, e, dt)

        # 更新机器人状态
        robot.move(omega, robot.v, dt)
        actual_path.append((robot.x, robot.y, robot.theta))

        # 判断是否到达终点
        if nearest_idx >= len(trajectory) - 2:
            break
            
    return actual_path
