#!/bin/bash

# TopsRider 完整安装脚本
# 基于官方安装包 TopsRider_t2x_2.5.136_deb_amd64.run
# 包含 eccl、torch_gcu 等关键组件

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否为root用户
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "此脚本需要root权限运行"
        log_info "请使用: sudo $0"
        exit 1
    fi
}

# 检查安装包是否存在
check_installer() {
    local installer_path="/installer/TopsRider_t2x_2.5.136_deb_amd64.run"
    
    if [[ ! -f "$installer_path" ]]; then
        log_error "安装包不存在: $installer_path"
        log_info "请确保安装包已上传到 /installer/ 目录"
        exit 1
    fi
    
    log_success "找到安装包: $installer_path"
    return 0
}

# 显示安装包组件
show_components() {
    log_info "显示安装包组件列表..."
    /installer/TopsRider_t2x_2.5.136_deb_amd64.run -l
}

# 安装基础平台组件
install_base_platform() {
    log_info "安装基础平台组件..."
    
    # 安装 TopsPlatform (驱动和主机组件)
    log_info "安装 TopsPlatform..."
    /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C topsplatform
    
    if [[ $? -eq 0 ]]; then
        log_success "TopsPlatform 安装成功"
    else
        log_error "TopsPlatform 安装失败"
        return 1
    fi
}

# 安装核心SDK组件
install_core_sdk() {
    log_info "安装核心SDK组件..."
    
    # 安装 topsfactor
    log_info "安装 topsfactor..."
    /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C topsfactor
    
    # 安装 tops-sdk
    log_info "安装 tops-sdk..."
    /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C tops-sdk
    
    # 安装 tops-eccl (关键的分布式通信库)
    log_info "安装 tops-eccl (分布式通信库)..."
    /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C tops-eccl
    
    if [[ $? -eq 0 ]]; then
        log_success "核心SDK组件安装成功"
    else
        log_error "核心SDK组件安装失败"
        return 1
    fi
}

# 检测Python版本并安装对应的torch_gcu
install_torch_gcu() {
    log_info "检测Python版本并安装torch_gcu..."
    
    # 获取Python版本
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "检测到Python版本: $python_version"
    
    case $python_version in
        "3.6")
            log_info "安装Python 3.6版本的torch_gcu..."
            /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C torch-gcu --python=3.6
            ;;
        "3.8")
            log_info "安装Python 3.8版本的torch_gcu..."
            /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C torch-gcu --python=3.8
            ;;
        "3.10"|"3.11")
            log_warning "Python $python_version 可能不被直接支持"
            log_info "尝试安装Python 3.8版本的torch_gcu..."
            /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C torch-gcu --python=3.8
            ;;
        *)
            log_error "不支持的Python版本: $python_version"
            log_info "支持的版本: 3.6, 3.8"
            return 1
            ;;
    esac
    
    if [[ $? -eq 0 ]]; then
        log_success "torch_gcu 安装成功"
    else
        log_error "torch_gcu 安装失败"
        return 1
    fi
}

# 安装Horovod (分布式训练框架)
install_horovod() {
    log_info "安装Horovod分布式训练框架..."
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    
    case $python_version in
        "3.6")
            /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C horovod_115 --python=3.6
            ;;
        "3.8")
            /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C horovod_115 --python=3.8
            ;;
        *)
            log_warning "为Python $python_version 安装Python 3.8版本的Horovod"
            /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C horovod_115 --python=3.8
            ;;
    esac
    
    if [[ $? -eq 0 ]]; then
        log_success "Horovod 安装成功"
    else
        log_warning "Horovod 安装失败，但不影响基本功能"
    fi
}

# 安装AI开发工具包
install_ai_toolkit() {
    log_info "安装AI开发工具包..."
    
    /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C ai_development_toolkit
    /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C tops-models
    
    if [[ $? -eq 0 ]]; then
        log_success "AI开发工具包安装成功"
    else
        log_warning "AI开发工具包安装失败，但不影响核心功能"
    fi
}

# 设置环境变量
setup_environment() {
    log_info "设置环境变量..."
    
    # 创建环境变量配置文件
    cat > /etc/profile.d/topsrider.sh << 'EOF'
# TopsRider Environment Variables
export TOPS_INSTALL_PATH=/usr/local/tops
export TOPS_RUNTIME_PATH=/usr/local/tops/runtime
export TOPSRIDER_PATH=/usr/local/tops
export GCU_DEVICE_PATH=/dev/gcu

# 添加到库路径
export LD_LIBRARY_PATH=/usr/local/tops/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/tops/runtime/lib:$LD_LIBRARY_PATH

# 添加到Python路径
export PYTHONPATH=/usr/local/tops/python:$PYTHONPATH
EOF

    # 使环境变量立即生效
    source /etc/profile.d/topsrider.sh
    
    log_success "环境变量设置完成"
}

# 验证安装
verify_installation() {
    log_info "验证安装结果..."
    
    # 检查关键文件
    local key_paths=(
        "/usr/local/tops"
        "/usr/local/tops/lib"
        "/usr/local/tops/bin"
        "/usr/local/tops/runtime"
    )
    
    for path in "${key_paths[@]}"; do
        if [[ -d "$path" ]]; then
            log_success "✅ $path 存在"
        else
            log_warning "⚠️  $path 不存在"
        fi
    done
    
    # 检查Python模块
    log_info "检查Python模块..."
    
    python3 -c "import torch_gcu; print('torch_gcu version:', torch_gcu.__version__)" 2>/dev/null && \
        log_success "✅ torch_gcu 可用" || \
        log_warning "⚠️  torch_gcu 不可用"
    
    python3 -c "import eccl; print('eccl 可用')" 2>/dev/null && \
        log_success "✅ eccl 可用" || \
        log_warning "⚠️  eccl 不可用"
    
    # 检查设备
    if [[ -e "/dev/gcu0" ]]; then
        log_success "✅ GCU设备可用"
        ls -la /dev/gcu*
    else
        log_warning "⚠️  GCU设备不可用"
    fi
}

# 主安装流程
main() {
    log_info "开始TopsRider完整安装..."
    log_info "安装包版本: TopsRider_t2x_2.5.136"
    
    # 检查权限
    check_root
    
    # 检查安装包
    check_installer
    
    # 显示组件
    show_components
    
    # 确认安装
    echo
    read -p "是否继续安装? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "安装已取消"
        exit 0
    fi
    
    # 执行安装步骤
    log_info "开始安装流程..."
    
    install_base_platform || exit 1
    install_core_sdk || exit 1
    install_torch_gcu || exit 1
    install_horovod
    install_ai_toolkit
    setup_environment
    
    log_success "TopsRider安装完成！"
    
    # 验证安装
    verify_installation
    
    log_info "安装总结:"
    log_info "✅ TopsPlatform (驱动和平台)"
    log_info "✅ tops-eccl (分布式通信)"
    log_info "✅ torch_gcu (PyTorch GCU支持)"
    log_info "✅ 环境变量配置"
    
    log_warning "请重新登录或执行 'source /etc/profile.d/topsrider.sh' 使环境变量生效"
    log_info "然后运行环境检测脚本验证安装: python scripts/check_torch_gcu_environment.py"
}

# 执行主函数
main "$@"