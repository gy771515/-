# map/grid_map.py
import numpy as np
import heapq
from scipy.ndimage import distance_transform_edt


class GridMap:

    def __init__(self, width, height):
        self.width = width  # 栅格列数（x范围：0~width-1）
        self.height = height  # 栅格行数（y范围：0~height-1）
        self.grid = np.zeros((height, width), dtype=int)  # 0:自由，1:障碍
        self.distance_map = None # 距离障碍物的欧氏距离图
        self._dist_dirty = True # 标记距离图是否需要更新

    def add_obstacle(self, x: int, y: int):
        """添加单点障碍（x,y为栅格索引，整数）"""
        if self.in_bounds(x, y):
            self.grid[y, x] = 1
            self._dist_dirty = True

    def add_obstacle_block(self, x1: int, y1: int, x2: int, y2: int):
        """添加矩形障碍（含边界，x1<=x<=x2，y1<=y<=y2）"""
        # 修正numpy切片左闭右开问题：+1包含终点
        self.grid[y1:y2+1, x1:x2+1] = 1
        self._dist_dirty = True

    def add_obstacle_rect(self, x1: int, y1: int, x2: int, y2: int):
        x_min = max(0, min(x1, x2))
        x_max = min(self.width - 1, max(x1, x2))
        y_min = max(0, min(y1, y2))
        y_max = min(self.height - 1, max(y1, y2))
        self.add_obstacle_block(x_min, y_min, x_max, y_max)

    def remove_obstacle(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y):
            return False
        if self.grid[y, x] == 0:
            return False
        self.grid[y, x] = 0
        self._dist_dirty = True
        return True

    def clear_obstacles(self):
        self.grid.fill(0)
        self._dist_dirty = True

    def update_distance_map(self):
        """更新每个点到最近障碍物的欧氏距离"""
        if self._dist_dirty or self.distance_map is None:
            # distance_transform_edt 计算的是到非零像素（障碍物）的距离
            # 我们的 grid 中 1 表示障碍，0 表示通路，所以反转一下
            binary_grid = (self.grid == 0) # 通路为True，障碍为False
            # 计算到False像素（障碍）的距离
            self.distance_map = distance_transform_edt(binary_grid)
            self._dist_dirty = False
        return self.distance_map

    def get_safety_cost(self, x, y, safe_dist=5.0):
        """
        获取指定点的安全代价。
        距离障碍物越近，代价越高。
        :param x, y: 栅格坐标
        :param safe_dist: 安全距离（米/栅格单位），超过此距离代价为0
        :return: 代价值 (0.0 ~ 1.0)
        """
        if self.distance_map is None or self._dist_dirty:
            self.update_distance_map()
        
        dist = self.distance_map[y, x]
        if dist >= safe_dist:
            return 0.0
        
        # 指数级增长代价，让靠近障碍物时代价增长更快
        cost = (safe_dist - dist) / safe_dist
        return cost ** 4 # 使用四次方，提高对极近障碍物的惩罚

    def generate_random_obstacles(self, density: float, seed: int = None, exclude_points: list = None):
        """
        随机生成障碍物
        :param density: 障碍物密度 (0.0 ~ 1.0)
        :param seed: 随机种子 (可选，用于复现实验)
        :param exclude_points: 需要排除的点列表 [(x1, y1), (x2, y2), ...]，如起点和终点
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 计算障碍物总数
        total_cells = self.width * self.height
        num_obstacles = int(total_cells * density)
        
        # 获取所有可用的位置索引
        all_indices = np.arange(total_cells)
        
        # 排除指定的点
        if exclude_points:
            exclude_indices = []
            for p in exclude_points:
                if self.in_bounds(p[0], p[1]):
                    exclude_indices.append(p[1] * self.width + p[0])
            
            # 从所有索引中移除需要排除的索引
            available_indices = np.setdiff1d(all_indices, exclude_indices)
        else:
            available_indices = all_indices

        # 确保生成的障碍物数量不超过可用位置数量
        num_obstacles = min(num_obstacles, len(available_indices))
        
        # 随机选择位置（不重复）
        indices = np.random.choice(available_indices, num_obstacles, replace=False)
        
        # 将一维索引转换为二维坐标并设置障碍
        for idx in indices:
            y, x = divmod(idx, self.width)
            self.grid[y, x] = 1

    def ensure_path_exists(self, start, goal):
        """
        确保存在一条从起点到终点的路径，若不存在则强制打通。
        使用Dijkstra算法，将障碍物视为高代价区域，优先选择现有空地。
        """
        sx, sy = int(start[0]), int(start[1])
        gx, gy = int(goal[0]), int(goal[1])
        
        # 边界检查
        if not (self.in_bounds(sx, sy) and self.in_bounds(gx, gy)):
            return

        # 优先队列 (cost, x, y)
        pq = [(0, sx, sy)]
        # 记录来源用于回溯: (x, y) -> (parent_x, parent_y)
        came_from = {(sx, sy): None}
        # 记录最小代价
        cost_so_far = {(sx, sy): 0}
        
        found = False
        final_node = None

        while pq:
            current_cost, cx, cy = heapq.heappop(pq)
            
            if (cx, cy) == (gx, gy):
                found = True
                final_node = (cx, cy)
                break
            
            # 4邻域搜索
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                
                if not self.in_bounds(nx, ny):
                    continue
                
                # 障碍物代价
                weight = 100 if self.grid[ny, nx] == 1 else 1
                new_cost = cost_so_far.get((cx, cy), float('inf')) + weight
                
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    heapq.heappush(pq, (new_cost, nx, ny))
                    came_from[(nx, ny)] = (cx, cy)
        
        if found:
            # 回溯路径并清除沿途障碍
            curr = final_node
            while curr is not None:
                cx, cy = curr
                if self.grid[cy, cx] == 1:
                    self.grid[cy, cx] = 0 # 强制清除障碍
                curr = came_from.get(curr)

    def is_free(self, x: int, y: int) -> bool:
        """检查栅格(x,y)是否为自由空间（x,y必须为整数）"""
        return self.in_bounds(x, y) and self.grid[y, x] == 0

    def in_bounds(self, x: int, y: int) -> bool:
        """检查栅格(x,y)是否在地图范围内（x,y必须为整数）"""
        return 0 <= x < self.width and 0 <= y < self.height

    def to_array(self):
        return self.grid.copy()

    def plot(self, ax):
        rows, cols = self.grid.shape
        ax.imshow(self.grid, cmap='gray_r', origin="upper", extent=[0, cols, rows, 0])
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_xticks(np.arange(0, cols + 1, 1))
        ax.set_yticks(np.arange(0, rows + 1, 1))
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, linestyle='--', alpha=0.7)


def create_test_map() -> GridMap:
    """创建测试地图（修复缩进）"""
    m = GridMap(30, 30)

    # 墙（x:10~12，y:5~25，包含边界）
    m.add_obstacle_block(10, 5, 12, 25)

    # 零散障碍（x:5~7，y:5）
    m.add_obstacle(5, 5)
    m.add_obstacle(6, 5)
    m.add_obstacle(7, 5)

    return m
