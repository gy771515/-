
import matplotlib.pyplot as plt
import numpy as np
from map.grid_map import GridMap

def test_random_map():
    # 测试不同密度的随机地图生成
    densities = [0.1, 0.2, 0.3]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, density in enumerate(densities):
        # 1. 初始化地图
        grid_map = GridMap(30, 30)
        
        # 2. 生成随机障碍物 (固定种子以便对比)
        grid_map.generate_random_obstacles(density=density, seed=42)
        
        # 3. 绘制
        ax = axes[i]
        grid_map.plot(ax)
        ax.set_title(f"Random Map (Density={density})")
        
        # 统计障碍物数量
        num_obstacles = np.sum(grid_map.grid)
        total_cells = grid_map.width * grid_map.height
        actual_density = num_obstacles / total_cells
        print(f"密度设置: {density}, 实际障碍数: {num_obstacles}/{total_cells} ({actual_density:.2%})")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_random_map()
