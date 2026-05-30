# control/robot.py
class Robot:
    def __init__(self, x, y, theta=0.0):
        """初始化机器人位置和方向角"""
        self._x = float(x)  # 私有属性，通过方法访问/修改
        self._y = float(y)
        self._theta = float(theta)  # 方向角（弧度）

    def position(self):
        """返回位置坐标 (x, y)"""
        return (self._x, self._y)

    def get_theta(self):
        """返回方向角"""
        return self._theta

    def set_pose(self, x, y, theta=None):
        """更新位置和方向角（theta可选），封装属性修改"""
        self._x = float(x)
        self._y = float(y)
        if theta is not None:
            self._theta = float(theta)

    def x(self):  # 提供x的只读访问
        return self._x

    def y(self):  # 提供y的只读访问
        return self._y