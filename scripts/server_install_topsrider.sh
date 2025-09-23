#!/bin/bash

# TopsRider Components Installation Script for T20 Server
# 在T20服务器上安装ECCL、SDK和torch_gcu组件

set -e  # 遇到错误立即退出

# 配置变量
INSTALL_DIR="/tmp/topsrider_install"
LOG_FILE="/tmp/topsrider_install.log"
BACKUP_DIR="/tmp/topsrider_backup_$(date +%Y%m%d_%H%M%S)"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOG_FILE"
    exit 1
}

# 检查是否为root用户
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "此脚本需要root权限运行。请使用 sudo $0"
    fi
}

# 检查系统兼容性
check_system() {
    log "检查系统兼容性..."
    
    # 检查操作系统
    if ! command -v dpkg &> /dev/null; then
        error "系统不支持dpkg包管理器，无法安装DEB包"
    fi
    
    # 检查Python环境
    if ! command -v python3 &> /dev/null; then
        error "未找到Python3，请先安装Python3"
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log "检测到Python版本: $PYTHON_VERSION"
    
    # 检查pip
    if ! command -v pip3 &> /dev/null; then
        log "未找到pip3，尝试安装..."
        apt-get update && apt-get install -y python3-pip
    fi
}

# 备份现有环境
backup_environment() {
    log "备份现有环境到 $BACKUP_DIR..."
    mkdir -p "$BACKUP_DIR"
    
    # 备份环境变量文件
    if [[ -f ~/.bashrc ]]; then
        cp ~/.bashrc "$BACKUP_DIR/bashrc.backup"
    fi
    
    if [[ -f /etc/environment ]]; then
        cp /etc/environment "$BACKUP_DIR/environment.backup"
    fi
    
    # 备份已安装的相关包信息
    dpkg -l | grep -E "(eccl|tops|torch)" > "$BACKUP_DIR/installed_packages.txt" 2>/dev/null || true
    pip3 list | grep -E "(torch|gcu)" > "$BACKUP_DIR/python_packages.txt" 2>/dev/null || true
}

# 安装DEB包
install_deb_packages() {
    log "开始安装DEB包..."
    
    # 检查DEB包是否存在
    local deb_packages=(
        "$INSTALL_DIR/tops-eccl_2.5.136-1_amd64.deb"
        "$INSTALL_DIR/tops-sdk_2.5.136-1_amd64.deb"
        "$INSTALL_DIR/topsfactor_2.5.136-1_amd64.deb"
    )
    
    for package in "${deb_packages[@]}"; do
        if [[ ! -f "$package" ]]; then
            error "DEB包不存在: $package"
        fi
    done
    
    # 更新包索引
    log "更新包索引..."
    apt-get update
    
    # 安装依赖
    log "安装可能的依赖..."
    apt-get install -y build-essential python3-dev
    
    # 安装DEB包
    for package in "${deb_packages[@]}"; do
        log "安装 $(basename "$package")..."
        if dpkg -i "$package"; then
            log "成功安装 $(basename "$package")"
        else
            log "dpkg安装失败，尝试修复依赖..."
            apt-get install -f -y
            if dpkg -i "$package"; then
                log "修复依赖后成功安装 $(basename "$package")"
            else
                error "无法安装 $(basename "$package")"
            fi
        fi
    done
}

# 安装torch_gcu
install_torch_gcu() {
    log "开始安装torch_gcu..."
    
    # 检测Python版本并选择对应的whl包
    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    local torch_gcu_package=""
    
    case "$python_version" in
        "3.8")
            torch_gcu_package="$INSTALL_DIR/torch_gcu-1.10.0+2.5.136-py3.8-none-any.whl"
            ;;
        "3.9")
            torch_gcu_package="$INSTALL_DIR/torch_gcu-1.10.0+2.5.136-py3.9-none-any.whl"
            ;;
        "3.10")
            torch_gcu_package="$INSTALL_DIR/torch_gcu-1.10.0+2.5.136-py3.10-none-any.whl"
            ;;
        *)
            # 默认使用py3.8版本
            torch_gcu_package="$INSTALL_DIR/torch_gcu-1.10.0+2.5.136-py3.8-none-any.whl"
            log "警告: Python版本 $python_version 可能不完全兼容，使用py3.8版本的torch_gcu"
            ;;
    esac
    
    if [[ ! -f "$torch_gcu_package" ]]; then
        error "torch_gcu包不存在: $torch_gcu_package"
    fi
    
    # 卸载可能存在的torch相关包
    log "卸载可能冲突的torch包..."
    pip3 uninstall -y torch torchvision torchaudio torch_gcu 2>/dev/null || true
    
    # 安装torch_gcu
    log "安装 $(basename "$torch_gcu_package")..."
    if pip3 install "$torch_gcu_package"; then
        log "成功安装torch_gcu"
    else
        error "无法安装torch_gcu"
    fi
}

# 配置环境变量
configure_environment() {
    log "配置环境变量..."
    
    # 创建环境变量配置文件
    local env_file="/etc/profile.d/topsrider.sh"
    
    cat > "$env_file" << 'EOF'
# TopsRider Environment Variables
export TOPS_SDK_PATH="/usr/local/tops"
export ECCL_ROOT="/usr/local/eccl"
export LD_LIBRARY_PATH="/usr/local/eccl/lib:/usr/local/tops/lib:$LD_LIBRARY_PATH"
export PATH="/usr/local/eccl/bin:/usr/local/tops/bin:$PATH"

# ECCL Configuration
export ECCL_ASYNC_DISABLE=false
export ECCL_MAX_NCHANNELS=2
export ECCL_RUNTIME_3_0_ENABLE=true
export ECCL_DEBUG=0

# GCU Configuration
export ENFLAME_VISIBLE_DEVICES=all
export GCU_MEMORY_FRACTION=0.9
EOF
    
    chmod +x "$env_file"
    log "环境变量配置文件已创建: $env_file"
    
    # 立即加载环境变量
    source "$env_file"
    
    # 添加到当前用户的bashrc
    if ! grep -q "source $env_file" ~/.bashrc; then
        echo "source $env_file" >> ~/.bashrc
        log "已将环境变量配置添加到 ~/.bashrc"
    fi
}

# 验证安装
verify_installation() {
    log "验证安装结果..."
    
    local success=true
    
    # 检查DEB包安装
    local packages=("tops-eccl" "tops-sdk" "topsfactor")
    for package in "${packages[@]}"; do
        if dpkg -l | grep -q "$package"; then
            log "✓ $package 已安装"
        else
            log "✗ $package 未正确安装"
            success=false
        fi
    done
    
    # 检查torch_gcu安装
    if python3 -c "import torch_gcu; print('torch_gcu version:', torch_gcu.__version__)" 2>/dev/null; then
        log "✓ torch_gcu 已安装并可导入"
    else
        log "✗ torch_gcu 未正确安装或无法导入"
        success=false
    fi
    
    # 检查ECCL库文件
    if [[ -f "/usr/local/eccl/lib/libeccl.so" ]] || [[ -f "/usr/lib/libeccl.so" ]]; then
        log "✓ ECCL库文件存在"
    else
        log "✗ ECCL库文件未找到"
        success=false
    fi
    
    # 检查环境变量
    source /etc/profile.d/topsrider.sh
    if [[ -n "$ECCL_ROOT" ]]; then
        log "✓ 环境变量已配置"
    else
        log "✗ 环境变量未正确配置"
        success=false
    fi
    
    if $success; then
        log "✓ 所有组件安装验证成功！"
        return 0
    else
        log "✗ 部分组件安装验证失败"
        return 1
    fi
}

# 生成安装报告
generate_report() {
    local report_file="/tmp/topsrider_install_report.txt"
    
    cat > "$report_file" << EOF
TopsRider Components Installation Report
========================================
安装时间: $(date)
安装目录: $INSTALL_DIR
日志文件: $LOG_FILE
备份目录: $BACKUP_DIR

已安装的DEB包:
$(dpkg -l | grep -E "(eccl|tops)" || echo "无")

已安装的Python包:
$(pip3 list | grep -E "(torch|gcu)" || echo "无")

环境变量配置:
$(cat /etc/profile.d/topsrider.sh 2>/dev/null || echo "配置文件不存在")

系统信息:
操作系统: $(lsb_release -d 2>/dev/null | cut -f2 || echo "未知")
Python版本: $(python3 --version)
内核版本: $(uname -r)

安装状态: $(verify_installation >/dev/null 2>&1 && echo "成功" || echo "部分失败")
EOF
    
    log "安装报告已生成: $report_file"
    cat "$report_file"
}

# 主函数
main() {
    log "开始TopsRider组件安装..."
    
    # 检查安装目录
    if [[ ! -d "$INSTALL_DIR" ]]; then
        error "安装目录不存在: $INSTALL_DIR"
    fi
    
    check_root
    check_system
    backup_environment
    install_deb_packages
    install_torch_gcu
    configure_environment
    
    if verify_installation; then
        log "✓ TopsRider组件安装完成！"
        generate_report
        
        log ""
        log "下一步操作:"
        log "1. 重新登录或运行 'source ~/.bashrc' 加载环境变量"
        log "2. 运行测试脚本验证ECCL后端功能"
        log "3. 查看安装报告: /tmp/topsrider_install_report.txt"
        
    else
        error "安装过程中出现问题，请查看日志: $LOG_FILE"
    fi
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi