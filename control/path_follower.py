# control/path_follower.py
import math
from BiYeSheJi.map.grid_map import GridMap  # 引入地图用于避障检测
from BiYeSheJi.control.robot import Robot  # 引入Robot类解决未定义问题


def _check_collision(robot, target, grid_map: GridMap) -> bool:
    """检查机器人从当前位置到目标点的直线路径是否碰撞障碍物"""
    x0, y0 = robot.position()
    x1, y1 = target
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(int(dx)), abs(int(dy))) or 1  # 按栅格步数采样

    for i in range(steps + 1):
        # 线性插值路径点（转换为栅格索引）
        x = int(x0 + dx * i / steps)
        y = int(y0 + dy * i / steps)
        if not grid_map.is_free(x, y):
            return True  # 碰撞
    return False


def step_towards(robot: Robot, target, grid_map: GridMap, step_size=0.05) -> bool:
    """向目标点移动一步，若路径无障碍则更新位置，返回是否到达目标"""
    dx = target[0] - robot.x()
    dy = target[1] - robot.y()
    dist = math.hypot(dx, dy)

    if dist < step_size:
        # 到达目标前最后一步，检查目标点是否安全
        if not grid_map.is_free(int(target[0]), int(target[1])):
            raise ValueError("目标点为障碍物，无法到达")
        robot.set_pose(*target)
        return True

    # 计算下一步位置
    next_x = robot.x() + step_size * dx / dist
    next_y = robot.y() + step_size * dy / dist

    # 检查下一步路径是否碰撞
    if _check_collision(robot, (next_x, next_y), grid_map):
        raise RuntimeError("路径存在障碍物，无法移动")

    robot.set_pose(next_x, next_y)
    return False


def follow_path(robot: Robot, path, grid_map: GridMap):
    """跟踪路径，返回轨迹（含避障检查）"""
    traj = [robot.position()]
    idx = 0

    while idx < len(path):
        try:
            reached = step_towards(robot, path[idx], grid_map)
            traj.append(robot.position())
            if reached:
                idx += 1
        except (ValueError, RuntimeError) as e:
            print(f"路径跟踪失败：{e}")
            break  # 遇障终止

    return traj