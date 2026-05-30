#Astar.py
import math
import heapq


def line_intersects_obstacle(grid_map, x1, y1, x2, y2):
    """
    检查从浮点数(x1,y1)到(x2,y2)的线段是否穿过任何障碍栅格。
    使用更精确的栅格遍历方法。
    """
    # 确保坐标在地图范围内
    x1 = max(0, min(grid_map.width - 1, x1))
    y1 = max(0, min(grid_map.height - 1, y1))
    x2 = max(0, min(grid_map.width - 1, x2))
    y2 = max(0, min(grid_map.height - 1, y2))

    # 使用Bresenham算法遍历线段上的所有栅格
    x, y = round(x1), round(y1)
    x_end, y_end = round(x2), round(y2)
    dx = abs(x_end - x)
    dy = abs(y_end - y)
    step_x = 1 if x_end > x else -1
    step_y = 1 if y_end > y else -1
    err = dx - dy

    while True:
        # 检查当前栅格是否为障碍
        if not grid_map.is_free(x, y):
            return True
        # 到达终点则退出
        if x == x_end and y == y_end:
            break
        err2 = 2 * err
        if err2 > -dy:
            err -= dy
            x += step_x
        if err2 < dx:
            err += dx
            y += step_y
    return False

def heuristic(a, b, heuristic_type='chebyshev'):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    if heuristic_type == 'manhattan':
        return dx + dy
    elif heuristic_type == 'euclidean':
        return math.hypot(dx, dy)
    else:  # chebyshev
        return max(dx, dy)

def astar(grid_map, start, goal, neighbor_type='8', heuristic_type='chebyshev', safety_weight=0.0, safe_dist=5.0):

    if safety_weight > 0:
        grid_map.update_distance_map()

    # 邻域定义
    if neighbor_type == '8':
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]
    else:
        neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    # 起点/终点保留浮点数，仅判断栅格时取整（兼容上位机浮点数逻辑）
    start_grid = (round(start[0]), round(start[1]))
    goal_grid = (round(goal[0]), round(goal[1]))

    searched_nodes = 0  # 初始化搜索节点计数

    # 优先级队列：(总代价, x, y)
    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal, heuristic_type), start_grid[0], start_grid[1]))

    # 代价记录：g_score(已走代价) + f_score(总代价=g+h)
    g_score = {start_grid: 0}
    f_score = {start_grid: heuristic(start, goal, heuristic_type)}
    came_from = {start_grid: None}
    order_visited = []  # 新增：记录节点弹出的顺序，用于动画演示

    while open_heap:
        searched_nodes += 1  # 每弹出一个节点，计数+1
        current_f, x, y = heapq.heappop(open_heap)
        current = (x, y)
        order_visited.append(current)  # 记录弹出顺序

        # 到达终点，回溯路径
        if current == goal_grid:
            path = []
            while current is not None:
                # 转回浮点数，与上位机轨迹逻辑兼容
                path.append((float(current[0]), float(current[1])))
                current = came_from[current]
            path.reverse()
            # 返回路径和访问过的节点（按弹出顺序）
            return path, order_visited

        # 遍历邻域
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            next_node = (nx, ny)

            # 1) 目标栅格必须是自由空间
            if not grid_map.is_free(nx, ny):
                continue

            # 2) 禁止“斜着从墙角挤过去”：对角步长时要求两侧相邻栅格都为自由
            if dx != 0 and dy != 0:
                if (not grid_map.is_free(x + dx, y)) or (not grid_map.is_free(x, y + dy)):
                    continue

            # 3) 线段级别的再次检查：从当前栅格到目标栅格是否穿过障碍
            if line_intersects_obstacle(grid_map, x, y, nx, ny):
                continue
            # 计算邻域代价
            step_cost = math.hypot(dx, dy) if neighbor_type == '8' else 1.0
            
            # 安全代价计算
            safety_penalty = 0.0
            if safety_weight > 0:
                # 我们希望路径离障碍物至少 safe_dist 个栅格
                safety_penalty = grid_map.get_safety_cost(nx, ny, safe_dist=safe_dist)
            
            # 最终代价 = 距离代价 + 安全权重 * 安全惩罚
            # 注意：安全惩罚需要根据 step_cost 的数量级进行缩放，否则可能导致 A* 绕远路太多
            # 这里我们让 safety_penalty * 5.0 (假设最大惩罚权重) 与 step_cost (1.0) 比例协调
            tentative_g = g_score[current] + step_cost + safety_weight * safety_penalty * 2.0

            # 更新代价
            if next_node not in g_score or tentative_g < g_score[next_node]:
                came_from[next_node] = current
                g_score[next_node] = tentative_g
                f_score[next_node] = tentative_g + heuristic((nx, ny), goal, heuristic_type)
                heapq.heappush(open_heap, (f_score[next_node], nx, ny))

    # 无可行路径
    return None, []

def dijkstra(grid_map, start, goal, neighbor_type='8', safety_weight=0.0):
    """Dijkstra算法（返回3个值：路径、总代价、搜索节点数）"""
    # 确保距离图是最新的
    if safety_weight > 0:
        grid_map.update_distance_map()

    # 邻域定义
    if neighbor_type == '8':
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]
    else:
        neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    start_grid = (round(start[0]), round(start[1]))
    goal_grid = (round(goal[0]), round(goal[1]))

    # 初始化：新增搜索节点数统计
    searched_nodes = 0  # 统计搜索过的节点数
    open_heap = []
    heapq.heappush(open_heap, (0.0, start_grid[0], start_grid[1]))
    cost_so_far = {start_grid: 0.0}
    came_from = {start_grid: None}
    order_visited = []  # 新增：记录节点弹出的顺序

    while open_heap:
        searched_nodes += 1  # 每弹出一个节点，计数+1
        current_cost, x, y = heapq.heappop(open_heap)
        current = (x, y)
        order_visited.append(current)  # 记录弹出顺序

        if current == goal_grid:
            path = []
            while current is not None:
                path.append((float(current[0]), float(current[1])))
                current = came_from[current]
            path.reverse()
            # 返回路径和访问过的节点（按弹出顺序）
            return path, order_visited

        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            next_node = (nx, ny)

            # 1) 目标栅格必须是自由空间
            if not grid_map.is_free(nx, ny):
                continue

            # 2) 禁止“斜着从墙角挤过去”
            if dx != 0 and dy != 0:
                if (not grid_map.is_free(x + dx, y)) or (not grid_map.is_free(x, y + dy)):
                    continue

            step_cost = math.hypot(dx, dy) if neighbor_type == '8' else 1.0
            
            # 安全代价计算
            safety_penalty = 0.0
            if safety_weight > 0:
                safety_penalty = grid_map.get_safety_cost(nx, ny, safe_dist=5.0)
            
            new_cost = current_cost + step_cost + safety_weight * safety_penalty * 2.0

            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                came_from[next_node] = current
                heapq.heappush(open_heap, (new_cost, nx, ny))

    return None, []

def reconstruct_path(came_from, current):
    """重构路径：保留原有逻辑，增加空值防护"""
    if not came_from and current:  # 起点=终点的情况
        return [current]
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
