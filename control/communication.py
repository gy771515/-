import struct
import time
import random
import math

class RobotProtocol:
    """
    上位机通信协议类
    定义了上位机与下位机（如STM32/ROS）之间的数据帧格式
    帧格式：[帧头(0xAA 0x55)] [长度] [命令类型] [数据载荷] [校验和]
    """
    FRAME_HEADER = b'\xAA\x55'
    
    # 命令类型定义
    CMD_MOVE = 0x01      # 下发运动控制指令 (v, omega)
    CMD_STATUS = 0x02    # 接收下位机状态回传 (x, y, theta, v, w)
    CMD_STOP = 0x03      # 紧急停止
    CMD_HEARTBEAT = 0x04 # 心跳包
    
    @staticmethod
    def pack_move_cmd(v, omega):
        """将线速度和角速度打包成数据帧"""
        payload = struct.pack('ff', float(v), float(omega))
        length = len(payload)
        # 计算简单的和校验
        checksum = (RobotProtocol.CMD_MOVE + length + sum(payload)) & 0xFF
        frame = RobotProtocol.FRAME_HEADER + bytes([length + 2, RobotProtocol.CMD_MOVE]) + payload + bytes([checksum])
        return frame

    @staticmethod
    def unpack_status_frame(data):
        """解析下位机回传的状态帧"""
        if len(data) < 7 or data[0:2] != RobotProtocol.FRAME_HEADER:
            return None
        
        length = data[2]
        cmd = data[3]
        if cmd != RobotProtocol.CMD_STATUS:
            return None
            
        payload = data[4:4+length-2]
        checksum = data[-1]
        
        # 验证校验和
        if (cmd + length + sum(payload)) & 0xFF != checksum:
            return None
            
        try:
            # 解析回传的 5 个 float 数据：x, y, theta, v_act, w_act
            x, y, theta, v_act, w_act = struct.unpack('fffff', payload)
            return {"x": x, "y": y, "theta": theta, "v": v_act, "w": w_act}
        except:
            return None

class MockLowerComputer:
    """
    模拟下位机（如STM32）
    用于在无硬件连接时验证上位机控制系统的逻辑
    """
    def __init__(self, init_pos=(0, 0, 0)):
        self.x, self.y, self.theta = init_pos
        self.v_actual = 0.0
        self.w_actual = 0.0
        self.friction = 1.0  # 新增：摩擦系数 (0.0 ~ 1.0)
        self.last_time = time.time()
        
        # 内部状态：用于模拟惯性（即使轮子打滑，动量也会保持）
        self.vx_global = 0.0
        self.vy_global = 0.0
        
    def process_command(self, frame, sim_dt=None):
        """处理来自上位机的指令。如果提供了 sim_dt，则在仿真模式下运行"""
        if frame[0:2] != RobotProtocol.FRAME_HEADER:
            return None
            
        cmd = frame[3]
        if cmd == RobotProtocol.CMD_MOVE:
            payload = frame[4:-1]
            v_target, w_target = struct.unpack('ff', payload)

            if sim_dt is not None:
                dt = sim_dt
            else:
                now = time.time()
                dt = now - self.last_time
                dt = max(0.001, min(0.1, dt)) 
                self.last_time = now
            

            alpha_v = 0.1 + 0.7 * self.friction 
            self.v_actual += (v_target - self.v_actual) * alpha_v
            
            # 模拟转向响应
            alpha_w = 0.2 + 0.6 * self.friction
            self.w_actual += (w_target - self.w_actual) * alpha_w


            self.theta += self.w_actual * dt
            self.theta = (self.theta + 3.14159) % (2 * 3.14159) - 3.14159
            
            # 模拟全局速度矢量 (实现“漂移”感)
            ideal_vx = self.v_actual * math.cos(self.theta)
            ideal_vy = self.v_actual * math.sin(self.theta)

            alpha_drift = 0.05 + 0.45 * self.friction
            self.vx_global += (ideal_vx - self.vx_global) * alpha_drift
            self.vy_global += (ideal_vy - self.vy_global) * alpha_drift
            
            # 更新位姿
            self.x += self.vx_global * dt
            self.y += self.vy_global * dt
            
            # 打包状态回传
            return self.pack_status()
        return None

    def pack_status(self):
        """生成状态回传帧"""
        payload = struct.pack('fffff', float(self.x), float(self.y), float(self.theta), float(self.v_actual), float(self.w_actual))
        length = len(payload)
        cmd = RobotProtocol.CMD_STATUS
        checksum = (cmd + length + 2 + sum(payload)) & 0xFF
        frame = RobotProtocol.FRAME_HEADER + bytes([length + 2, cmd]) + payload + bytes([checksum])
        return frame
