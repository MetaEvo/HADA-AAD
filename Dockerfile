# Dockerfile: 可用于目前所有domain的Docker 镜像
# 基于 python:3.12-slim，包含 cocoex 和所有必要的依赖

FROM python:3.12-slim

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1

# 配置 Debian 镜像源（阿里云）
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources || \
    sed -i 's|http://deb.debian.org|http://mirrors.aliyun.com|g' /etc/apt/sources.list

# 安装系统依赖
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    git \
    wget \
    ca-certificates \
    graphviz \
    libgraphviz-dev \
    libosmesa6-dev \
    libgl1 \
    libglew-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# 配置 pip 使用清华源加速
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 设置工作目录
WORKDIR /hada

# 配置 Git 安全目录
RUN git config --global --add safe.directory /hada

# 配置网络和 SSL 设置
ENV PYTHONHTTPSVERIFY=0
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# 复制 requirements 文件并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 安装 CPU 版本的 torch
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 安装 coco-experiment 和 cocopp（BBOB benchmark 的核心依赖）
RUN pip install --no-cache-dir coco-experiment cocopp

# 安装 metaevobox 的所有依赖（除了 pygame）
RUN pip install --no-cache-dir \
    scipy \
    pandas \
    matplotlib==3.10.1 \
    tensorboard==2.19.0 \
    tensorboardX==2.6.2.2 \
    gym>=0.26 \
    gymnasium \
    GPUtil==1.4.0 \
    psutil==7.0.0 \
    ray==2.44.1 \
    cmaes==0.11.1 \
    cma \
    dill==0.4.0 \
    Cython>=3.0.12 \
    xgboost==3.0.0 \
    tianshou==1.1.0 \
    openpyxl==3.1.5 \
    deap>=1.4.2 \
    evox>=1.1.2

# 安装 pypop7（跳过依赖检查）
RUN pip install --no-cache-dir pypop7>=0.0.82 --no-deps

# 安装 metaevobox（使用 --no-deps 避免重复安装依赖）
RUN pip install --no-cache-dir metaevobox --no-deps

# 复制项目文件
COPY . .

# 验证 cocoex 安装
RUN python -c "import cocoex; print('cocoex installed successfully')" || echo "WARNING: cocoex import failed"

# 验证 metabbo 模块可以导入
RUN python -c "import sys; sys.path.append('metabbo'); import pso; print('pso module loaded successfully')" || echo "WARNING: pso module import failed"

# 默认命令（保持容器运行）
CMD ["tail", "-f", "/dev/null"]