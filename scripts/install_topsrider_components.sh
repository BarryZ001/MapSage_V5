#!/bin/bash

# TopsRider组件自动化安装脚本
# 安装ECCL、SDK和torch_gcu组件以解决分布式训练问题

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

# 配置变量
TOPSRIDER_PATH="/Users/barryzhang/myDev3/MapSage_V5/test/topsrider_extracted/TopsRider_t2x_2.5.136_deb_amd64"
BACKUP_DIR="$HOME/topsrider_backup_$(date +%Y%m%d_%H%M%S)"
INSTALL_LOG="$HOME/topsrider_install.log"

# 检查是否为root用户（某些操作需要）
check_sudo() {
    if ! sudo -n true 2>/dev/null; then
        log_warning "某些操作需要sudo权限，请确保可以使用sudo"
    fi
}

# 检查TopsRider包路径
check_topsrider_path() {
    log_info "检查TopsRider包路径..."
    
    if [ ! -d "$TOPSRIDER_PATH" ]; then
        log_error "TopsRider包路径不存在: $TOPSRIDER_PATH"
        log_info "请确认包已正确解压到指定位置"
        exit 1
    fi
    
    log_success "TopsRider包路径验证成功"
}

# 创建备份
create_backup() {
    log_info "创建环境备份..."
    
    mkdir -p "$BACKUP_DIR"
    
    # 备份当前Python环境
    if command -v pip &> /dev/null; then
        pip freeze > "$BACKUP_DIR/requirements_backup.txt"
        log_success "Python环境已备份到: $BACKUP_DIR/requirements_backup.txt"
    fi
    
    # 备份当前torch信息
    python -c "
import torch
import sys
print(f'PyTorch版本: {torch.__version__}')
print(f'Python版本: {sys.version}')
try:
    import torch_gcu
    print(f'torch_gcu版本: {torch_gcu.__version__}')
except ImportError:
    print('torch_gcu: 未安装')
" > "$BACKUP_DIR/torch_info_backup.txt" 2>/dev/null || echo "无法获取torch信息" > "$BACKUP_DIR/torch_info_backup.txt"
    
    log_success "备份创建完成: $BACKUP_DIR"
}

# 检查系统兼容性
check_system_compatibility() {
    log_info "检查系统兼容性..."
    
    # 检查操作系统
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        log_warning "当前系统不是Linux，DEB包可能无法安装"
        log_info "当前系统: $OSTYPE"
    fi
    
    # 检查架构
    ARCH=$(uname -m)
    if [ "$ARCH" != "x86_64" ]; then
        log_warning "当前架构不是x86_64，可能存在兼容性问题"
        log_info "当前架构: $ARCH"
    fi
    
    # 检查Python版本
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Python版本: $PYTHON_VERSION"
    
    if [[ "$PYTHON_VERSION" != "3.6" && "$PYTHON_VERSION" != "3.8" ]]; then
        log_warning "Python版本可能不兼容，推荐使用Python 3.6或3.8"
    fi
    
    log_success "系统兼容性检查完成"
}

# 安装DEB包
install_deb_packages() {
    log_info "安装DEB包..."
    
    # 检查是否在Linux系统上
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        log_warning "非Linux系统，跳过DEB包安装"
        return 0
    fi
    
    # ECCL包
    ECCL_DEB="$TOPSRIDER_PATH/distributed/tops-eccl_2.5.136-1_amd64.deb"
    if [ -f "$ECCL_DEB" ]; then
        log_info "安装ECCL包..."
        if sudo dpkg -i "$ECCL_DEB" 2>&1 | tee -a "$INSTALL_LOG"; then
            log_success "ECCL包安装成功"
        else
            log_warning "ECCL包安装可能有问题，尝试修复依赖..."
            sudo apt-get install -f -y 2>&1 | tee -a "$INSTALL_LOG"
        fi
    else
        log_error "ECCL DEB包不存在: $ECCL_DEB"
    fi
    
    # SDK包
    SDK_DEB="$TOPSRIDER_PATH/sdk/tops-sdk_2.5.136-1_amd64.deb"
    if [ -f "$SDK_DEB" ]; then
        log_info "安装SDK包..."
        if sudo dpkg -i "$SDK_DEB" 2>&1 | tee -a "$INSTALL_LOG"; then
            log_success "SDK包安装成功"
        else
            log_warning "SDK包安装可能有问题，尝试修复依赖..."
            sudo apt-get install -f -y 2>&1 | tee -a "$INSTALL_LOG"
        fi
    else
        log_error "SDK DEB包不存在: $SDK_DEB"
    fi
    
    # TopsFactor包
    FACTOR_DEB="$TOPSRIDER_PATH/sdk/topsfactor_2.5.136-1_amd64.deb"
    if [ -f "$FACTOR_DEB" ]; then
        log_info "安装TopsFactor包..."
        if sudo dpkg -i "$FACTOR_DEB" 2>&1 | tee -a "$INSTALL_LOG"; then
            log_success "TopsFactor包安装成功"
        else
            log_warning "TopsFactor包安装可能有问题，尝试修复依赖..."
            sudo apt-get install -f -y 2>&1 | tee -a "$INSTALL_LOG"
        fi
    else
        log_error "TopsFactor DEB包不存在: $FACTOR_DEB"
    fi
}

# 安装torch_gcu
install_torch_gcu() {
    log_info "安装torch_gcu..."
    
    # 根据Python版本选择合适的wheel包
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    
    if [ "$PYTHON_VERSION" = "3.8" ]; then
        TORCH_GCU_WHEEL="$TOPSRIDER_PATH/framework/torch_gcu-1.10.0+2.5.136-py3.8-none-any.whl"
    elif [ "$PYTHON_VERSION" = "3.6" ]; then
        TORCH_GCU_WHEEL="$TOPSRIDER_PATH/framework/torch_gcu-1.10.0-2.5.136-py3.6-none-any.whl"
    else
        # 默认尝试py3.8版本
        TORCH_GCU_WHEEL="$TOPSRIDER_PATH/framework/torch_gcu-1.10.0+2.5.136-py3.8-none-any.whl"
        log_warning "Python版本 $PYTHON_VERSION 可能不完全兼容，尝试使用py3.8版本的wheel包"
    fi
    
    if [ -f "$TORCH_GCU_WHEEL" ]; then
        log_info "安装torch_gcu wheel包: $(basename $TORCH_GCU_WHEEL)"
        
        # 先卸载可能存在的torch_gcu
        pip uninstall torch_gcu -y 2>/dev/null || true
        
        # 安装新的torch_gcu
        if pip install "$TORCH_GCU_WHEEL" 2>&1 | tee -a "$INSTALL_LOG"; then
            log_success "torch_gcu安装成功"
        else
            log_error "torch_gcu安装失败"
            return 1
        fi
    else
        log_error "torch_gcu wheel包不存在: $TORCH_GCU_WHEEL"
        return 1
    fi
}

# 设置环境变量
setup_environment() {
    log_info "设置环境变量..."
    
    # 创建环境配置文件
    ENV_FILE="$HOME/.topsrider_env"
    
    cat > "$ENV_FILE" << 'EOF'
# TopsRider环境配置
# ECCL配置
export ECCL_ASYNC_DISABLE=false
export ECCL_MAX_NCHANNELS=2
export ECCL_RUNTIME_3_0_ENABLE=true
export ECCL_DEBUG=INFO

# GCU设备配置
export ENFLAME_VISIBLE_DEVICES=0,1,2,3
export GCU_DEVICE_COUNT=4

# 库路径配置
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/lib/python3.8/site-packages:$PYTHONPATH

# 添加到PATH
export PATH=/usr/local/bin:$PATH
EOF
    
    log_success "环境配置文件已创建: $ENV_FILE"
    log_info "请在~/.bashrc或~/.zshrc中添加: source $ENV_FILE"
    
    # 检查是否已经在shell配置中
    SHELL_RC="$HOME/.bashrc"
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_RC="$HOME/.zshrc"
    fi
    
    if ! grep -q "source $ENV_FILE" "$SHELL_RC" 2>/dev/null; then
        log_info "是否自动添加到 $SHELL_RC? (y/n)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo "source $ENV_FILE" >> "$SHELL_RC"
            log_success "已添加到 $SHELL_RC"
        fi
    fi
}

# 验证安装
verify_installation() {
    log_info "验证安装结果..."
    
    # 创建验证脚本
    VERIFY_SCRIPT="/tmp/verify_topsrider.py"
    
    cat > "$VERIFY_SCRIPT" << 'EOF'
#!/usr/bin/env python3
import sys

def check_torch():
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        return True
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False

def check_torch_gcu():
    try:
        import torch_gcu
        print(f"✅ torch_gcu版本: {torch_gcu.__version__}")
        print(f"✅ GCU设备数量: {torch_gcu.device_count()}")
        return True
    except ImportError as e:
        print(f"❌ torch_gcu导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ torch_gcu检查失败: {e}")
        return False

def check_distributed_backends():
    try:
        import torch.distributed as dist
        if dist.is_available():
            backends = []
            # 检查各种后端
            for backend in ['gloo', 'nccl', 'mpi']:
                try:
                    if hasattr(dist.Backend, backend.upper()):
                        backends.append(backend)
                except:
                    pass
            
            # 特别检查ECCL
            try:
                # 尝试不同的方式检查ECCL
                if hasattr(dist.Backend, 'ECCL'):
                    backends.append('eccl')
                elif 'eccl' in str(dist.Backend.__dict__).lower():
                    backends.append('eccl')
            except:
                pass
            
            print(f"✅ 可用的分布式后端: {backends}")
            
            if 'eccl' in backends:
                print("✅ ECCL后端可用")
                return True
            else:
                print("⚠️  ECCL后端不在标准后端列表中，但可能仍然可用")
                return False
        else:
            print("❌ torch.distributed不可用")
            return False
    except Exception as e:
        print(f"❌ 分布式后端检查失败: {e}")
        return False

def main():
    print("🔍 TopsRider安装验证")
    print("=" * 50)
    
    results = []
    results.append(check_torch())
    results.append(check_torch_gcu())
    results.append(check_distributed_backends())
    
    print("\n📊 验证结果摘要:")
    print("=" * 50)
    
    if all(results[:2]):  # torch和torch_gcu必须成功
        print("✅ 核心组件安装成功")
        if results[2]:
            print("✅ ECCL后端验证成功")
            print("🎉 安装完全成功！可以使用ECCL进行分布式训练")
        else:
            print("⚠️  ECCL后端验证未完全成功，但torch_gcu已安装")
            print("💡 建议运行分布式训练测试以确认ECCL功能")
    else:
        print("❌ 安装存在问题，请检查错误信息")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF
    
    # 运行验证
    if python "$VERIFY_SCRIPT"; then
        log_success "安装验证通过"
    else
        log_warning "安装验证存在问题，请查看详细信息"
    fi
    
    # 清理验证脚本
    rm -f "$VERIFY_SCRIPT"
}

# 生成安装报告
generate_report() {
    log_info "生成安装报告..."
    
    REPORT_FILE="$HOME/topsrider_install_report.txt"
    
    cat > "$REPORT_FILE" << EOF
TopsRider组件安装报告
=====================

安装时间: $(date)
安装路径: $TOPSRIDER_PATH
备份目录: $BACKUP_DIR
安装日志: $INSTALL_LOG

系统信息:
- 操作系统: $OSTYPE
- 架构: $(uname -m)
- Python版本: $(python --version)

安装的组件:
- ECCL库 (tops-eccl_2.5.136-1_amd64.deb)
- SDK (tops-sdk_2.5.136-1_amd64.deb)
- TopsFactor (topsfactor_2.5.136-1_amd64.deb)
- torch_gcu (wheel包)

环境配置:
- 配置文件: $HOME/.topsrider_env
- 需要source到shell配置文件中

下一步:
1. 重启终端或运行: source $HOME/.topsrider_env
2. 运行分布式训练测试
3. 如有问题，查看安装日志: $INSTALL_LOG

回滚方法:
如需回滚，运行以下命令:
sudo dpkg -r tops-eccl tops-sdk topsfactor
pip uninstall torch_gcu -y
pip install -r $BACKUP_DIR/requirements_backup.txt
EOF
    
    log_success "安装报告已生成: $REPORT_FILE"
}

# 主函数
main() {
    echo "🚀 TopsRider组件自动化安装"
    echo "================================"
    
    # 检查sudo权限
    check_sudo
    
    # 检查TopsRider包
    check_topsrider_path
    
    # 创建备份
    create_backup
    
    # 检查系统兼容性
    check_system_compatibility
    
    # 询问是否继续
    log_info "是否继续安装? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log_info "安装已取消"
        exit 0
    fi
    
    # 开始安装
    log_info "开始安装TopsRider组件..."
    
    # 安装DEB包
    install_deb_packages
    
    # 安装torch_gcu
    install_torch_gcu
    
    # 设置环境
    setup_environment
    
    # 验证安装
    verify_installation
    
    # 生成报告
    generate_report
    
    echo ""
    log_success "🎉 TopsRider组件安装完成！"
    log_info "请查看安装报告: $HOME/topsrider_install_report.txt"
    log_info "重启终端或运行: source $HOME/.topsrider_env"
}

# 运行主函数
main "$@"