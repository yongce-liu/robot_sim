# 模块结构说明

## 📦 优化后的目录结构

```
robot_sim/
├── backends/          # 模拟器后端
├── communication/     # 通信层
├── config/            # 配置管理
├── controllers/       # 控制器
├── scenes/            # 场景构建
└── sensors/           # 传感器
```

## 🎯 设计理念

### 1. **扁平化架构**
- 所有核心模块直接位于 `robot_sim/` 下
- 减少嵌套层级，提高代码可读性
- 模块职责清晰，易于维护

### 2. **语义化命名**
- `backends` (复数) - 表示多个模拟器后端
- `communication` (完整单词) - 比 `comm` 更清晰
- `controllers` (复数) - 明确是控制器集合
- `scenes` (复数) - 场景相关功能
- `sensors` (复数) - 传感器模块

### 3. **功能内聚**
- 每个模块职责单一明确
- 相关功能集中在同一模块
- 减少跨模块依赖

## 📂 模块详解

### backends/ - 模拟器后端
负责与不同物理引擎交互

```
backends/
├── __init__.py
├── base.py          # 统一后端接口
├── isaac.py         # Isaac Lab 实现
├── mujoco.py        # MuJoCo 实现
├── manager.py       # 模拟管理器
└── factory.py       # 后端工厂
```

**核心类**:
- `BackendBase`: 统一接口
- `SimulationManager`: 管理多个后端
- `BackendFactory`: 创建后端实例

### communication/ - 通信层
处理模拟器与外部模型的通信

```
communication/
├── __init__.py
├── server.py        # 模拟服务器
├── client.py        # 模型客户端
├── protocol.py      # 通信协议基类
└── messages.py      # ZMQ 实现
```

**核心类**:
- `SimulationServer`: 发送状态，接收控制
- `ModelClient`: 连接模拟器，运行模型
- `ZMQProtocol`: ZMQ 通信实现

**使用场景**:
- VLA 模型交互
- 规划器通信
- 远程控制

### config/ - 配置管理
基于 Hydra 的配置系统

```
config/
├── __init__.py
└── loader.py        # 配置加载器
```

**特性**:
- YAML 配置文件
- 命令行覆盖
- 配置组合

### controllers/ - 控制器
机器人控制算法

```
controllers/
├── __init__.py
└── controller.py    # PD, 轨迹控制器
```

**支持类型**:
- PD 控制
- 轨迹跟踪
- 力控制

### scenes/ - 场景构建
快速搭建仿真环境

```
scenes/
├── __init__.py
└── builder.py       # SceneBuilder 工具
```

**功能**:
- 添加地面、物体
- 创建楼梯、斜坡
- 加载 URDF/MJCF

**示例**:
```python
from robot_sim.scenes import SceneBuilder

scene = (
    SceneBuilder()
    .add_ground_plane()
    .add_box("obstacle", position=(2, 0, 0.5))
    .build()
)
```

### sensors/ - 传感器
各类传感器实现

```
sensors/
├── __init__.py
└── base.py          # Camera, IMU, Contact
```

**传感器类型**:
- `Camera`: RGB, Depth, Segmentation
- `IMU`: 加速度、角速度
- `ContactSensor`: 接触力
- `SensorManager`: 传感器管理

## 🔄 从旧结构迁移

### 导入路径变更

| 旧路径 | 新路径 |
|--------|--------|
| `robot_sim.backend` | `robot_sim.backends` |
| `robot_sim.comm` | `robot_sim.communication` |
| `robot_sim.control` | `robot_sim.controllers` |
| `robot_sim.utils.scene_builder` | `robot_sim.scenes` |
| `robot_sim.utils.sensors` | `robot_sim.sensors` |

### 示例代码更新

**旧代码**:
```python
from robot_sim.backend import SimulationManager
from robot_sim.utils.sensors import Camera
from robot_sim.utils.scene_builder import SceneBuilder
```

**新代码**:
```python
from robot_sim.backends import SimulationManager
from robot_sim.sensors import Camera
from robot_sim.scenes import SceneBuilder
```

## ✨ 优势总结

1. **更清晰** - 模块名称语义化，一目了然
2. **更扁平** - 减少嵌套，降低复杂度
3. **更易维护** - 职责分明，便于扩展
4. **更规范** - 遵循 Python 包结构最佳实践

## 🚀 快速开始

```python
# 导入核心组件
from robot_sim import (
    SimulationManager,
    ModelClient,
    SceneBuilder,
    Camera,
    IMU,
)

# 创建模拟器
manager = SimulationManager(config)
manager.add_backend("main", "mujoco", config)

# 添加传感器
camera = Camera(name="front_cam", width=640, height=480)

# 构建场景
scene = SceneBuilder().add_ground_plane().build()

# 运行模拟
manager.step()
```
