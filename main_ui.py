import sys
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from datetime import datetime
import time
import math
import random

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QLabel, QSpinBox, QPushButton, QTextEdit, QSlider,
    QGridLayout, QSizePolicy, QMessageBox, QDoubleSpinBox, QToolButton,
    QCheckBox, QFrame
)
from PyQt5.QtCore import QTimer, Qt, pyqtSlot
from PyQt5.QtGui import QFont

from map.grid_map import GridMap
from planning.Astar import astar, dijkstra
from planning.motion_control import path_to_trajectory, PIDController, RobotSimulator, get_trajectory_info
from control.communication import RobotProtocol, MockLowerComputer


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.parent = parent
        self.drawing = False
        self.mode = 'obstacle'  # 'obstacle', 'set_start', 'set_goal'
        self.start_x, self.start_y = -1, -1
        self.last_end_x, self.last_end_y = None, None
        self.preview_rect = None

        self.mpl_connect('button_press_event', self.on_mouse_press)
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.mpl_connect('button_release_event', self.on_mouse_release)

    def on_mouse_press(self, event):
        if event.inaxes != self.axes:
            self.drawing = False
            self._clear_preview()
            return

        if self.mode in ['set_start', 'set_goal']:
            x = int(round(event.xdata)) if event.xdata else 0
            y = int(round(event.ydata)) if event.ydata else 0
            if self.mode == 'set_start':
                self.parent.set_start_node(x, y)
            else:
                self.parent.set_goal_node(x, y)
            self.mode = 'obstacle'
            return

        if event.button == 1:  # Left-click to start drawing
            self.drawing = True
            self.start_x = int(round(event.xdata)) if event.xdata else 0
            self.start_y = int(round(event.ydata)) if event.ydata else 0
        elif event.button == 3:  # Right-click to remove obstacle
            x = int(round(event.xdata)) if event.xdata else 0
            y = int(round(event.ydata)) if event.ydata else 0
            self._remove_single_obstacle(x, y)

    def on_mouse_move(self, event):
        if not self.drawing or event.inaxes != self.axes:
            return

        end_x = int(round(event.xdata)) if event.xdata else 0
        end_y = int(round(event.ydata)) if event.ydata else 0
        if end_x == self.last_end_x and end_y == self.last_end_y:
            return
        self.last_end_x, self.last_end_y = end_x, end_y
        x0 = min(self.start_x, end_x)
        y0 = min(self.start_y, end_y)
        w = abs(end_x - self.start_x) + 1
        h = abs(end_y - self.start_y) + 1
        if self.preview_rect is None:
            self.preview_rect = Rectangle((x0, y0), w, h, color='gray', alpha=0.5)
            self.axes.add_patch(self.preview_rect)
        else:
            self.preview_rect.set_xy((x0, y0))
            self.preview_rect.set_width(w)
            self.preview_rect.set_height(h)
        self.draw_idle()

    def on_mouse_release(self, event):
        if not self.drawing:
            return

        self.drawing = False
        end_x = int(round(event.xdata)) if event.xdata else 0
        end_y = int(round(event.ydata)) if event.ydata else 0
        self._clear_preview()
        self.parent.add_obstacle_rect(self.start_x, self.start_y, end_x, end_y)

    def _clear_preview(self):
        if self.preview_rect:
            self.preview_rect.remove()
            self.preview_rect = None
        self.last_end_x, self.last_end_y = None, None
        self.draw_idle()

    def _remove_single_obstacle(self, x, y):
        self.parent.remove_obstacle_at(x, y)


class CollapsibleSection(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.toggle_button = QToolButton(self)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow)
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.toggled.connect(self.on_toggled)

        self.content_area = QWidget(self)
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_area)

    def on_toggled(self, checked):
        self.content_area.setVisible(checked)
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)


class RobotPlanningUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("机器人路径规划与仿真平台")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize components
        self.grid_map = GridMap(50, 50)
        self.pid_controller = PIDController(kp_heading=1.2, kd_heading=0.5, kp_lateral=0.2, lateral_deadzone=0.2)

        self.start_node = (5, 5)
        self.goal_node = (45, 45)
        self.path = None
        self.trajectory = None
        self.actual_traj = []
        self.dt = 0.05
        self.speed_ms = 50
        self.sim_steps_per_frame = 1
        self.base_v = 0.3
        self.max_v = 1.0
        self.max_omega = 2.0
        self.robot = self.create_robot()
        self.is_simulation_running = False
        
        # 上位机通信相关
        self.is_connected = False
        self.mock_hw = MockLowerComputer(init_pos=(self.start_node[0], self.start_node[1], 0.0))
        self.protocol = RobotProtocol()
        self.last_frame_sent = b''
        self.last_frame_received = b''

        self._last_recovery_log_ts = 0.0
        self._pending_recovery_result_log = False
        self._manual_recover_count = 0

        self.show_path = True
        self.show_traj = True
        self.show_robot_traj = True
        self.safety_weight = 0.0  # 新增：路径安全权重

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        # 顶层布局：垂直布局，分为 [地图+侧边栏] 和 [底部按钮栏]
        root_layout = QVBoxLayout(main_widget)
        
        # 中间区域：水平布局，包含 [地图] 和 [右侧侧边栏]
        mid_layout = QHBoxLayout()
        root_layout.addLayout(mid_layout, 1) # 占比 1

        # 左侧：地图显示
        self.canvas = MplCanvas(self, width=8, height=8, dpi=100)
        mid_layout.addWidget(self.canvas, 2)

        # 右侧：侧边栏（参数设置、监控、日志）
        control_panel = QVBoxLayout()
        mid_layout.addLayout(control_panel, 1)

        # 1. 上位机通信控制 (右侧)
        comm_section = CollapsibleSection("上位机通信控制")
        comm_layout = QGridLayout()
        self.btn_connect = QPushButton("建立连接 (Mock)")
        self.btn_connect.clicked.connect(self.toggle_connection)
        self.lbl_comm_status = QLabel("状态: 未连接")
        self.lbl_comm_status.setStyleSheet("color: red;")
        
        self.txt_data_monitor = QTextEdit()
        self.txt_data_monitor.setReadOnly(True)
        self.txt_data_monitor.setFixedHeight(60)
        self.txt_data_monitor.setFont(QFont("Courier New", 9))
        self.txt_data_monitor.setPlaceholderText("通信报文监控 (HEX)...")
        
        comm_layout.addWidget(self.btn_connect, 0, 0)
        comm_layout.addWidget(self.lbl_comm_status, 0, 1)
        comm_layout.addWidget(QLabel("协议报文 (HEX):"), 1, 0, 1, 2)
        comm_layout.addWidget(self.txt_data_monitor, 2, 0, 1, 2)
        comm_section.content_layout.addLayout(comm_layout)
        control_panel.addWidget(comm_section)

        # 2. PID 参数 (右侧)
        pid_section = CollapsibleSection("PID 参数")
        pid_layout = QGridLayout()
        pid_section.content_layout.addLayout(pid_layout)
        self.kp_heading_spin = QDoubleSpinBox()
        self.kd_heading_spin = QDoubleSpinBox()
        self.kp_lateral_spin = QDoubleSpinBox()
        self.deadzone_spin = QDoubleSpinBox()
        for spin in (self.kp_heading_spin, self.kd_heading_spin, self.kp_lateral_spin, self.deadzone_spin):
            spin.setDecimals(3)
            spin.setSingleStep(0.05)
        self.kp_heading_spin.setRange(0.0, 10.0)
        self.kd_heading_spin.setRange(0.0, 10.0)
        self.kp_lateral_spin.setRange(0.0, 5.0)
        self.deadzone_spin.setRange(0.0, 2.0)
        self.kp_heading_spin.setValue(self.pid_controller.kp_h)
        self.kd_heading_spin.setValue(self.pid_controller.kd_h)
        self.kp_lateral_spin.setValue(self.pid_controller.kp_l)
        self.deadzone_spin.setValue(self.pid_controller.deadzone)
        pid_layout.addWidget(QLabel("kp_heading"), 0, 0)
        pid_layout.addWidget(self.kp_heading_spin, 0, 1)
        pid_layout.addWidget(QLabel("kd_heading"), 1, 0)
        pid_layout.addWidget(self.kd_heading_spin, 1, 1)
        pid_layout.addWidget(QLabel("kp_lateral"), 2, 0)
        pid_layout.addWidget(self.kp_lateral_spin, 2, 1)
        pid_layout.addWidget(QLabel("deadzone"), 3, 0)
        pid_layout.addWidget(self.deadzone_spin, 3, 1)
        self.apply_pid_btn = QPushButton("应用 PID 参数")
        self.apply_pid_btn.clicked.connect(self.apply_pid_params)
        pid_layout.addWidget(self.apply_pid_btn, 4, 0, 1, 2)
        control_panel.addWidget(pid_section)

        # 3. 仿真速度 (右侧)
        speed_section = CollapsibleSection("仿真速度")
        speed_layout = QVBoxLayout()
        speed_section.content_layout.addLayout(speed_layout)
        self.speed_label = QLabel(f"{self.speed_ms} ms/步")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(20)
        self.speed_slider.setMaximum(300)
        self.speed_slider.setValue(self.speed_ms)
        self.speed_slider.valueChanged.connect(self.on_speed_change)
        speed_layout.addWidget(self.speed_label)
        speed_layout.addWidget(self.speed_slider)

        self.steps_label = QLabel(f"加速倍率: {self.sim_steps_per_frame}x")
        self.steps_slider = QSlider(Qt.Horizontal)
        self.steps_slider.setMinimum(1)
        self.steps_slider.setMaximum(50)
        self.steps_slider.setValue(self.sim_steps_per_frame)
        self.steps_slider.valueChanged.connect(self.on_steps_per_frame_change)
        speed_layout.addWidget(self.steps_label)
        speed_layout.addWidget(self.steps_slider)
        control_panel.addWidget(speed_section)

        # 4. 运动约束 (右侧)
        limits_section = CollapsibleSection("运动约束")
        limits_layout = QGridLayout()
        limits_section.content_layout.addLayout(limits_layout)
        self.base_v_slider = QSlider(Qt.Horizontal)
        self.max_v_slider = QSlider(Qt.Horizontal)
        self.max_omega_slider = QSlider(Qt.Horizontal)
        self.friction_slider = QSlider(Qt.Horizontal)
        self.base_v_slider.setMinimum(10)
        self.base_v_slider.setMaximum(200)
        self.max_v_slider.setMinimum(10)
        self.max_v_slider.setMaximum(300)
        self.max_omega_slider.setMinimum(10)
        self.max_omega_slider.setMaximum(400)
        self.friction_slider.setMinimum(10)
        self.friction_slider.setMaximum(100)
        self.base_v_slider.setValue(int(self.base_v * 100))
        self.max_v_slider.setValue(int(self.max_v * 100))
        self.max_omega_slider.setValue(int(self.max_omega * 100))
        self.friction_slider.setValue(100)
        self.base_v_label = QLabel(f"目标线速度: {self.base_v:.2f} m/s")
        self.max_v_label = QLabel(f"最大线速度: {self.max_v:.2f} m/s")
        self.max_omega_label = QLabel(f"最大角速度: {self.max_omega:.2f} rad/s")
        self.friction_label = QLabel("地面摩擦系数: 1.00")
        self.base_v_slider.valueChanged.connect(self.on_base_v_change)
        self.max_v_slider.valueChanged.connect(self.on_max_v_change)
        self.max_omega_slider.valueChanged.connect(self.on_max_omega_change)
        self.friction_slider.valueChanged.connect(self.on_friction_change)
        limits_layout.addWidget(self.base_v_label, 0, 0)
        limits_layout.addWidget(self.base_v_slider, 1, 0)
        limits_layout.addWidget(self.max_v_label, 2, 0)
        limits_layout.addWidget(self.max_v_slider, 3, 0)
        limits_layout.addWidget(self.max_omega_label, 4, 0)
        limits_layout.addWidget(self.max_omega_slider, 5, 0)
        limits_layout.addWidget(self.friction_label, 6, 0)
        limits_layout.addWidget(self.friction_slider, 7, 0)
        control_panel.addWidget(limits_section)

        # 5. 规划选项 (右侧)
        planning_opt_section = CollapsibleSection("规划选项")
        planning_opt_layout = QVBoxLayout()
        planning_opt_section.content_layout.addLayout(planning_opt_layout)
        
        self.safety_weight_label = QLabel(f"避障安全权重: {self.safety_weight:.1f}")
        self.safety_weight_slider = QSlider(Qt.Horizontal)
        self.safety_weight_slider.setMinimum(0)
        self.safety_weight_slider.setMaximum(100) # 0.0 ~ 10.0
        self.safety_weight_slider.setValue(0)
        self.safety_weight_slider.valueChanged.connect(self.on_safety_weight_change)
        
        planning_opt_layout.addWidget(self.safety_weight_label)
        planning_opt_layout.addWidget(self.safety_weight_slider)
        planning_opt_layout.addWidget(QLabel("(权重越大，路径越倾向于远离障碍物)"))
        control_panel.addWidget(planning_opt_section)

        # 6. 日志输出 (右侧)
        log_group = QGroupBox("日志输出")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(120) 
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        control_panel.addWidget(log_group)
        
        # 6. 显示选项 (右侧)
        view_group = QGroupBox("显示选项")
        view_layout = QVBoxLayout() # 改为垂直排列更省空间
        self.check_path = QCheckBox("显示规划路径(绿)")
        self.check_path.setChecked(True)
        self.check_path.toggled.connect(self.on_view_option_change)
        self.check_traj = QCheckBox("显示平滑轨迹(红)")
        self.check_traj.setChecked(True)
        self.check_traj.toggled.connect(self.on_view_option_change)
        self.check_robot_traj = QCheckBox("显示实际轨迹(青)")
        self.check_robot_traj.setChecked(True)
        self.check_robot_traj.toggled.connect(self.on_view_option_change)
        view_layout.addWidget(self.check_path)
        view_layout.addWidget(self.check_traj)
        view_layout.addWidget(self.check_robot_traj)
        view_group.setLayout(view_layout)
        control_panel.addWidget(view_group)
        
        control_panel.addStretch() # 添加弹簧，将内容往上推

        # --- 底部按钮栏 ---
        bottom_widget = QFrame()
        bottom_widget.setFrameShape(QFrame.StyledPanel)
        bottom_layout = QHBoxLayout(bottom_widget)
        root_layout.addWidget(bottom_widget)

        # 底部 - 路径规划部分
        plan_box = QGroupBox("路径规划操作")
        plan_h_layout = QHBoxLayout(plan_box)
        self.set_start_btn = QPushButton("设置起点")
        self.set_goal_btn = QPushButton("设置终点")
        self.algo_astar_btn = QPushButton("A* 算法")
        self.algo_dijkstra_btn = QPushButton("Dijkstra 算法")
        self.set_start_btn.clicked.connect(self.on_set_start_clicked)
        self.set_goal_btn.clicked.connect(self.on_set_goal_clicked)
        self.algo_astar_btn.clicked.connect(lambda: self.run_planning('astar'))
        self.algo_dijkstra_btn.clicked.connect(lambda: self.run_planning('dijkstra'))
        plan_h_layout.addWidget(self.set_start_btn)
        plan_h_layout.addWidget(self.set_goal_btn)
        plan_h_layout.addWidget(self.algo_astar_btn)
        plan_h_layout.addWidget(self.algo_dijkstra_btn)
        bottom_layout.addWidget(plan_box)

        # 底部 - 仿真执行部分
        exec_box = QGroupBox("仿真执行")
        exec_h_layout = QHBoxLayout(exec_box)
        self.start_sim_btn = QPushButton("开始仿真")
        self.pause_sim_btn = QPushButton("暂停仿真")
        self.reset_sim_btn = QPushButton("重置仿真")
        self.clear_obstacles_btn = QPushButton("清空障碍物")
        self.random_obs_btn = QPushButton("随机生成障碍物")
        self.follow_wall_checkbox = QCheckBox("自动回轨迹")
        
        self.start_sim_btn.clicked.connect(self.start_simulation)
        self.pause_sim_btn.clicked.connect(self.pause_simulation)
        self.reset_sim_btn.clicked.connect(self.reset_simulation)
        self.clear_obstacles_btn.clicked.connect(self.clear_obstacles)
        self.random_obs_btn.clicked.connect(self.generate_random_obstacles)
        
        # 设置样式，让开始仿真更醒目
        self.start_sim_btn.setStyleSheet("background-color: #e1f5fe; font-weight: bold;")
        
        exec_h_layout.addWidget(self.start_sim_btn)
        exec_h_layout.addWidget(self.pause_sim_btn)
        exec_h_layout.addWidget(self.reset_sim_btn)
        exec_h_layout.addWidget(self.clear_obstacles_btn)
        exec_h_layout.addWidget(self.random_obs_btn)
        exec_h_layout.addWidget(self.follow_wall_checkbox)
        bottom_layout.addWidget(exec_box)

        self.update_plot()

    def on_view_option_change(self):
        self.show_path = self.check_path.isChecked()
        self.show_traj = self.check_traj.isChecked()
        self.show_robot_traj = self.check_robot_traj.isChecked()
        self.update_plot()

    def update_plot(self):
        self.canvas.axes.clear()
        self.grid_map.plot(self.canvas.axes)
        
        # 绘制起点和终点
        self.canvas.axes.plot(self.start_node[0], self.start_node[1], 'go', markersize=8, label='Start')
        self.canvas.axes.plot(self.goal_node[0], self.goal_node[1], 'r*', markersize=12, label='Goal')

        if self.show_path and self.path is not None and len(self.path) > 0:
            self.canvas.axes.plot([p[0] for p in self.path], [p[1] for p in self.path], 'g-', linewidth=2)
        if self.show_traj and self.trajectory is not None and len(self.trajectory) > 0:
            traj = np.asarray(self.trajectory)
            if traj.ndim == 2 and traj.shape[1] >= 2:
                self.canvas.axes.plot(traj[:, 0], traj[:, 1], 'm--', linewidth=1.5)
            else:
                self.canvas.axes.plot([p[0] for p in traj], [p[1] for p in traj], 'm--', linewidth=1.5)
        if self.show_robot_traj and self.actual_traj is not None and len(self.actual_traj) > 0:
            act = np.asarray(self.actual_traj)
            if act.ndim == 2 and act.shape[1] >= 2:
                self.canvas.axes.plot(act[:, 0], act[:, 1], 'c-', linewidth=1.5)
        self.canvas.axes.plot(self.robot.x, self.robot.y, 'bo', markersize=6)
        self.canvas.axes.arrow(
            self.robot.x,
            self.robot.y,
            0.8 * np.cos(self.robot.theta),
            0.8 * np.sin(self.robot.theta),
            head_width=0.3,
            head_length=0.3,
            fc='b',
            ec='b'
        )
        self.canvas.draw_idle()

    @pyqtSlot(int, int, int, int)
    def add_obstacle_rect(self, x1, y1, x2, y2):
        self.grid_map.add_obstacle_rect(x1, y1, x2, y2)
        self.log(f"添加障碍物矩形: ({x1},{y1}) to ({x2},{y2})")
        self.update_plot()

    @pyqtSlot(int, int)
    def remove_obstacle_at(self, x, y):
        if self.grid_map.remove_obstacle(x, y):
            self.log(f"移除障碍物: ({x},{y})")
            self.update_plot()

    def on_set_start_clicked(self):
        self.canvas.mode = 'set_start'
        self.log("请在地图上点击选择起点...")

    def on_set_goal_clicked(self):
        self.canvas.mode = 'set_goal'
        self.log("请在地图上点击选择终点...")

    def set_start_node(self, x, y):
        if not self.grid_map.is_free(x, y):
            self.log(f"设置起点失败: ({x},{y}) 是障碍物")
            return
        self.start_node = (x, y)

        if not self.is_simulation_running:
            self.robot.x = x
            self.robot.y = y
        self.log(f"起点已设置为: ({x},{y})")
        self.update_plot()

    def set_goal_node(self, x, y):
        if not self.grid_map.is_free(x, y):
            self.log(f"设置终点失败: ({x},{y}) 是障碍物")
            return
        self.goal_node = (x, y)
        self.log(f"终点已设置为: ({x},{y})")
        self.update_plot()

    def run_planning(self, algorithm):
        start_node = (int(self.robot.x), int(self.robot.y))
        goal_node = self.goal_node

        try:
            self.log(f"开始规划路径，算法: {algorithm.upper()}, 安全权重: {self.safety_weight:.1f}")
            if algorithm == 'astar':
                path, visited = astar(self.grid_map, start_node, goal_node, safety_weight=self.safety_weight)
            else:
                path, visited = dijkstra(self.grid_map, start_node, goal_node, safety_weight=self.safety_weight)

            if path:
                self.path = path
                self.log(f"路径规划成功，长度: {len(path)}")
                # 使用B样条平滑轨迹，点数根据路径长度自适应，增加碰撞检查逻辑
                num_pts = max(500, len(path) * 20)
                self.trajectory = path_to_trajectory(path, grid_map=self.grid_map, num_points=num_pts)
                self.log(f"轨迹生成完毕，平滑点数: {len(self.trajectory)}")
                self.actual_traj = []
                self.update_plot()
            else:
                self.log("路径规划失败")
                QMessageBox.warning(self, "规划失败", "无法找到从起点到终点的路径。")
        except Exception as e:
            self.log(f"路径规划异常：{e}")
            QMessageBox.critical(self, "运行错误", f"路径规划异常：{e}")

    def start_simulation(self):
        if self.trajectory is None:
            QMessageBox.warning(self, "仿真错误", "请先进行路径规划。")
            return
        if not self.is_connected:
            QMessageBox.warning(self, "上位机未就绪", "请先建立上位机与下位机的通信连接！")
            return
        if not self.is_simulation_running:
            self.is_simulation_running = True
            self.log("仿真开始")
            QTimer.singleShot(self.speed_ms, self.update_simulation)

    def pause_simulation(self):
        self.is_simulation_running = False
        self.log("仿真暂停")

    def reset_simulation(self):
        self.is_simulation_running = False
        self.robot = self.create_robot()

        if hasattr(self, 'mock_hw'):
            self.mock_hw.x = self.robot.x
            self.mock_hw.y = self.robot.y
            self.mock_hw.theta = self.robot.theta
            self.mock_hw.v_actual = 0.0
            self.mock_hw.w_actual = 0.0
            self.mock_hw.vx_global = 0.0
            self.mock_hw.vy_global = 0.0
            
        self.path = None
        self.trajectory = None
        self.actual_traj = []
        self.log("仿真重置（位姿已归位）")
        self.update_plot()

    def clear_obstacles(self):
        self.grid_map.clear_obstacles()
        self.path = None
        self.trajectory = None
        self.actual_traj = []
        self.log("障碍物已清空")
        self.update_plot()

    def generate_random_obstacles(self):
        """随机生成障碍物"""
        try:
            from PyQt5.QtWidgets import QInputDialog
            density, ok = QInputDialog.getDouble(
                self, "随机生成障碍物", 
                "请输入障碍物密度 (0.1 ~ 0.5):", 
                0.2, 0.05, 0.8, 2
            )
            if ok:
                self.grid_map.clear_obstacles()
                
                # 排除起点、终点以及它们周围的区域（防止一出来就撞墙或终点被围死）
                start_x, start_y = self.start_node
                goal_x, goal_y = self.goal_node
                
                exclude_points = []
                # 排除起点及周围 3x3 区域
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        exclude_points.append((start_x + dx, start_y + dy))
                
                # 排除终点及周围 3x3 区域
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        exclude_points.append((goal_x + dx, goal_y + dy))

                self.grid_map.generate_random_obstacles(density=density, exclude_points=exclude_points)
                
                # 核心修复：确保至少有一条路径存在（防止高密度下堵死）
                self.grid_map.ensure_path_exists(self.start_node, self.goal_node)
                
                # 清除旧的路径，因为可能穿过新障碍物
                self.path = None
                self.trajectory = None
                self.actual_traj = []
                self.log(f"已生成随机障碍物，密度: {density} (已保护起终点)")
                self.update_plot()
        except Exception as e:
            self.log(f"生成随机障碍物失败: {e}")

    def _find_nearest_obstacle_cell(self, radius=3):
        if self.grid_map is None:
            return None
        gx0 = int(math.floor(self.robot.x))
        gy0 = int(math.floor(self.robot.y))
        best = None
        best_d2 = None
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                gx = gx0 + dx
                gy = gy0 + dy
                if gx < 0 or gx >= self.grid_map.width or gy < 0 or gy >= self.grid_map.height:
                    continue
                if int(self.grid_map.grid[gy, gx]) != 1:
                    continue
                d2 = dx * dx + dy * dy
                if best_d2 is None or d2 < best_d2:
                    best = (gx, gy)
                    best_d2 = d2
        return best

    def _wrap_angle(self, a):
        return math.atan2(math.sin(a), math.cos(a))

    def _project_to_trajectory(self, traj, nearest_idx, x, y):
        if len(traj) < 2:
            return float(x), float(y), float(self.robot.theta)

        nearest_idx = int(np.clip(nearest_idx, 0, len(traj) - 2))
        k = 25
        start = max(0, nearest_idx - k)
        end = min(len(traj) - 2, nearest_idx + k)

        best = None
        best_d2 = None
        for i in range(start, end + 1):
            p1 = traj[i]
            p2 = traj[i + 1]
            x1, y1 = float(p1[0]), float(p1[1])
            x2, y2 = float(p2[0]), float(p2[1])
            dx = x2 - x1
            dy = y2 - y1
            denom = dx * dx + dy * dy + 1e-9
            t = ((x - x1) * dx + (y - y1) * dy) / denom
            t = max(0.0, min(1.0, t))
            px = x1 + t * dx
            py = y1 + t * dy
            d2 = (x - px) * (x - px) + (y - py) * (y - py)
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = (px, py, math.atan2(dy, dx))

        if best is None:
            p1 = traj[nearest_idx]
            p2 = traj[nearest_idx + 1]
            theta_path = math.atan2(float(p2[1] - p1[1]), float(p2[0] - p1[0]))
            return float(p1[0]), float(p1[1]), float(theta_path)

        return float(best[0]), float(best[1]), float(best[2])

    def _nearest_free_point(self, x, y, max_radius=2):
        if self.grid_map is None:
            return x, y
        gx0 = int(math.floor(x))
        gy0 = int(math.floor(y))
        if self.grid_map.in_bounds(gx0, gy0) and self.grid_map.is_free(gx0, gy0):
            return float(x), float(y)

        best = None
        best_d2 = None
        for r in range(1, max_radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    gx = gx0 + dx
                    gy = gy0 + dy
                    if not self.grid_map.in_bounds(gx, gy):
                        continue
                    if not self.grid_map.is_free(gx, gy):
                        continue
                    cx = gx + 0.5
                    cy = gy + 0.5
                    d2 = (cx - x) * (cx - x) + (cy - y) * (cy - y)
                    if best_d2 is None or d2 < best_d2:
                        best = (cx, cy)
                        best_d2 = d2

        return best
##########################待优化2#############################
## 前端1m检测为1减速 碰撞自回线   # 人工势场 // DWA动态窗口？
    def manual_recover_to_trajectory(self, traj, nearest_idx):
        px, py, theta_path = self._project_to_trajectory(traj, nearest_idx, self.robot.x, self.robot.y)
        p = self._nearest_free_point(px, py, max_radius=4)
        if p is None:
            return False
        self.robot.x, self.robot.y = float(p[0]), float(p[1])

        if hasattr(self, 'mock_hw'):
            self.mock_hw.x = self.robot.x
            self.mock_hw.y = self.robot.y
            # 清除物理模拟惯性动量
            self.mock_hw.vx_global = 0.0
            self.mock_hw.vy_global = 0.0
            self.mock_hw.v_actual = 0.0
            self.mock_hw.w_actual = 0.0

        best_idx = int(np.argmin(np.hypot(traj[:, 0] - self.robot.x, traj[:, 1] - self.robot.y)))
        if best_idx >= len(traj) - 1:
            best_idx = len(traj) - 2
        dx = float(traj[best_idx + 1, 0] - traj[best_idx, 0])
        dy = float(traj[best_idx + 1, 1] - traj[best_idx, 1])
        self.robot.theta = math.atan2(dy, dx) if (dx * dx + dy * dy) > 1e-9 else float(theta_path)
        
        # 同步朝向到下位机
        if hasattr(self, 'mock_hw'):
            self.mock_hw.theta = self.robot.theta

        if hasattr(self.robot, "vx_global") and hasattr(self.robot, "vy_global"):
            self.robot.vx_global = 0.0
            self.robot.vy_global = 0.0
        if hasattr(self.robot, "v_actual"):
            self.robot.v_actual = 0.0
        if hasattr(self.robot, "omega_actual"):
            self.robot.omega_actual = 0.0
        self._manual_recover_count += 1
        return True

#######################################
    def toggle_connection(self):
        self.is_connected = not self.is_connected
        if self.is_connected:
            self.btn_connect.setText("断开连接")
            self.lbl_comm_status.setText("状态: 已建立 (Mock)")
            self.lbl_comm_status.setStyleSheet("color: green;")
            self.log("成功连接下位机仿真器 (Mock Hardware Interface Enabled)")
            # 同步初始位置和摩擦力
            self.mock_hw.x = self.robot.x
            self.mock_hw.y = self.robot.y
            self.mock_hw.theta = self.robot.theta
            self.mock_hw.friction = self.robot.friction
        else:
            self.btn_connect.setText("建立连接 (Mock)")
            self.lbl_comm_status.setText("状态: 未连接")
            self.lbl_comm_status.setStyleSheet("color: red;")
            self.log("断开与下位机的连接")

    def update_simulation(self):
        if not self.is_simulation_running or self.trajectory is None:
            return
        
        # 强制检查连接状态 (Safety Check)
        if not self.is_connected:
            self.is_simulation_running = False
            self.log("通信连接已断开，上位机控制停止。")
            return

        traj = np.asarray(self.trajectory)
        if traj.ndim != 2 or traj.shape[0] < 2:
            return

        update_gui_now = False 
        
        for _ in range(self.sim_steps_per_frame):
            # 1. 寻找最近点
            dist = np.sqrt(np.sum((traj - np.array([self.robot.x, self.robot.y])) ** 2, axis=1))
            nearest_idx = int(np.argmin(dist))
            theta_path, e = get_trajectory_info(traj, (self.robot.x, self.robot.y), nearest_idx)

            # 2. PID 计算控制量
            omega, v_corr = self.pid_controller.update(theta_path, self.robot.theta, e, self.dt)
            # 优化线速度下发：在偏差较小时维持基础速度，防止过度减速导致卡顿
            v_target = self.robot.v + v_corr * 0.5 
            v_target = max(0.1, min(v_target, self.max_v))
            
            # --- 上位机逻辑：指令下发与反馈闭环 ---
            # 打包并下发控制指令
            cmd_frame = self.protocol.pack_move_cmd(v_target, omega)
            self.last_frame_sent = cmd_frame
            
            # 模拟下位机处理并返回状态（在加速循环中传递固定的 dt 步长）
            status_frame = self.mock_hw.process_command(cmd_frame, sim_dt=self.dt)
            self.last_frame_received = status_frame if status_frame else b''
            
            # 解析回传状态
            status = self.protocol.unpack_status_frame(self.last_frame_received)
            if status:
                # 更新上位机观测到的机器人位姿
                self.robot.x = status['x']
                self.robot.y = status['y']
                self.robot.theta = status['theta']
                
                # 为了保持仿真器兼容性，调用 move(dt=0) 来更新内部碰撞状态或处理逻辑
                # 这里我们稍微调整一下逻辑：使用反馈值更新位姿，但仍需判断是否撞墙
                collided, _, _ = self.robot.is_collision(look_ahead=0.5) 
                update_gui_now = True
            else:
                collided = False
            
            # 采样记录通信日志
            if self.is_connected and random.random() < 0.02: 
                self.log(f"上位机指令下发: [CMD_MOVE] v={v_target:.2f}, w={omega:.2f}")
                self.log(f"接收下位机回传: [CMD_STATUS] x={self.robot.x:.2f}, y={self.robot.y:.2f}")

            self.actual_traj.append((self.robot.x, self.robot.y))
            
            # 3. 碰撞处理逻辑
            if collided:
                if self.follow_wall_checkbox.isChecked():
                    now = time.time()
                    if now - self._last_recovery_log_ts > 0.5:
                        self.log(f"检测到碰撞，执行人工修正回轨迹")
                        self._last_recovery_log_ts = now
                    ok = self.manual_recover_to_trajectory(traj, nearest_idx)
                    if not ok:
                        self.is_simulation_running = False
                        self.log(f"发生严重碰撞，无法修正。累计碰撞次数: {self.robot.collision_count}")
                        self.log("仿真停止")
                        break
                    
                    # 记录碰撞次数
                    self.robot.collision_count += 1
                    continue
                else:
                    self.is_simulation_running = False
                    self.robot.collision_count += 1
                    self.log(f"检测到碰撞，累计碰撞次数: {self.robot.collision_count}")
                    self.log("仿真停止")
                    break

            # 4. 到达检测
            goal = traj[-1]
            dist_to_goal = np.hypot(self.robot.x - goal[0], self.robot.y - goal[1])
            if dist_to_goal < 1.0:
                self.is_simulation_running = False
                self.log(f"到达终点，仿真结束。累计碰撞次数: {self.robot.collision_count}")
                break
        
        # --- 循环外更新：协议报文监控窗 ---
        if self.is_connected and update_gui_now:
            hex_tx = self.last_frame_sent.hex(' ').upper()
            hex_rx = self.last_frame_received.hex(' ').upper()
            self.txt_data_monitor.setPlainText(f"TX: {hex_tx}\nRX: {hex_rx}")
        
        self.update_plot()

        if self.is_simulation_running:
            QTimer.singleShot(self.speed_ms, self.update_simulation)

    def apply_pid_params(self):
        self.pid_controller.kp_h = float(self.kp_heading_spin.value())
        self.pid_controller.kd_h = float(self.kd_heading_spin.value())
        self.pid_controller.kp_l = float(self.kp_lateral_spin.value())
        self.pid_controller.deadzone = float(self.deadzone_spin.value())
        self.log("PID 参数已更新")

    def on_speed_change(self, value):
        self.speed_ms = int(value)
        self.speed_label.setText(f"{self.speed_ms} ms/步")

    def on_steps_per_frame_change(self, value):
        self.sim_steps_per_frame = int(value)
        self.steps_label.setText(f"加速倍率: {self.sim_steps_per_frame}x")

    def on_base_v_change(self, value):
        self.base_v = float(value) / 100.0
        self.base_v_label.setText(f"目标线速度: {self.base_v:.2f} m/s")
        self.robot.v = self.base_v

    def on_max_v_change(self, value):
        self.max_v = float(value) / 100.0
        self.max_v_label.setText(f"最大线速度: {self.max_v:.2f} m/s")
        self.robot.max_v = self.max_v

    def on_max_omega_change(self, value):
        self.max_omega = float(value) / 100.0
        self.max_omega_label.setText(f"最大角速度: {self.max_omega:.2f} rad/s")
        self.robot.max_omega = self.max_omega

    def on_friction_change(self, value):
        friction = float(value) / 100.0
        self.friction_label.setText(f"地面摩擦系数: {friction:.2f}")
        self.robot.friction = friction
        if hasattr(self, 'mock_hw'):
            self.mock_hw.friction = friction  # 实时更新下位机模拟器的摩擦力

    def on_safety_weight_change(self, value):
        self.safety_weight = float(value) / 10.0
        self.safety_weight_label.setText(f"避障安全权重: {self.safety_weight:.1f}")

    def create_robot(self):
        return RobotSimulator(
            init_pos=self.start_node,
            init_theta=np.deg2rad(45),
            v=self.base_v,
            grid_map=self.grid_map,
            tau_v=0.0,
            tau_omega=0.0,
            max_v=self.max_v,
            max_omega=self.max_omega,
            friction=1.0 # 默认摩擦系数
        )

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = RobotPlanningUI()
    main_win.show()
    sys.exit(app.exec_())
