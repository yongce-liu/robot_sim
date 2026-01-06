# robot_sim

## Basic Setup

```bash
uv venv --python 3.10
uv pip install -e .
```

## Teleoperation Package

```bash
git submodule update --init --recursive && \
cd third_party/teleop && proj_dir=$(pwd) && \
cd $proj_dir/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK  && \
rm -rf ./build && bash build.sh  && \
mkdir -p $proj_dir/XRoboToolkit-PC-Service-Pybind/lib && \
cp ./build/libPXREARobotSDK.so $proj_dir/XRoboToolkit-PC-Service-Pybind/lib && \
mkdir -p $proj_dir/XRoboToolkit-PC-Service-Pybind/include && \
cp ./PXREARobotSDK.h $proj_dir/XRoboToolkit-PC-Service-Pybind/include/ && \
cp -rf ./nlohmann $proj_dir/XRoboToolkit-PC-Service-Pybind/include/nlohmann && \
cd $proj_dir/../.. && \
uv pip install -e ".[packs]"
```

### Now install the teleop package

```python
pybind11_DIR=$(uv run python -m pybind11 --cmakedir) pip install ".[teleop]"
```
