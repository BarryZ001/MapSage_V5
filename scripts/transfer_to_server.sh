#!/bin/bash

# TopsRider组件传输脚本
# 将本地分析的TopsRider组件传输到T20服务器

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
LOCAL_TOPSRIDER_PATH="/Users/barryzhang/myDev3/MapSage_V5/test/topsrider_extracted/TopsRider_t2x_2.5.136_deb_amd64"
SERVER_HOST="root@10.20.52.143"
SERVER_INSTALL_DIR="/tmp/topsrider_install"
SERVER_SCRIPTS_DIR="/workspace/code/MapSage_V5/scripts"

# 检查本地TopsRider包
check_local_package() {
    log_info "检查本地TopsRider包..."
    
    if [ ! -d "$LOCAL_TOPSRIDER_PATH" ]; then
        log_error "本地TopsRider包不存在: $LOCAL_TOPSRIDER_PATH"
        exit 1
    fi
    
    # 检查关键文件
    local files_to_check=(
        "distributed/tops-eccl_2.5.136-1_amd64.deb"
        "sdk/tops-sdk_2.5.136-1_amd64.deb"
        "sdk/topsfactor_2.5.136-1_amd64.deb"
        "framework/torch_gcu-1.10.0+2.5.136-py3.8-none-any.whl"
    )
    
    for file in "${files_to_check[@]}"; do
        if [ ! -f "$LOCAL_TOPSRIDER_PATH/$file" ]; then
            log_error "关键文件不存在: $file"
            exit 1
        fi
    done
    
    log_success "本地TopsRider包验证成功"
}

# 测试服务器连接
test_server_connection() {
    log_info "测试服务器连接..."
    
    if ssh -o ConnectTimeout=10 "$SERVER_HOST" "echo 'Connection test successful'" >/dev/null 2>&1; then
        log_success "服务器连接成功"
    else
        log_error "无法连接到服务器: $SERVER_HOST"
        log_info "请检查："
        log_info "1. 服务器IP地址是否正确"
        log_info "2. SSH密钥是否配置正确"
        log_info "3. 网络连接是否正常"
        exit 1
    fi
}

# 在服务器上创建目录
create_server_directories() {
    log_info "在服务器上创建安装目录..."
    
    ssh "$SERVER_HOST" "
        mkdir -p $SERVER_INSTALL_DIR
        mkdir -p $SERVER_SCRIPTS_DIR
        echo '服务器目录创建成功'
    "
    
    log_success "服务器目录创建完成"
}

# 传输DEB包
transfer_deb_packages() {
    log_info "传输DEB包到服务器..."
    
    # ECCL包
    log_info "传输ECCL包..."
    scp "$LOCAL_TOPSRIDER_PATH/distributed/tops-eccl_2.5.136-1_amd64.deb" \
        "$SERVER_HOST:$SERVER_INSTALL_DIR/"
    
    # SDK包
    log_info "传输SDK包..."
    scp "$LOCAL_TOPSRIDER_PATH/sdk/tops-sdk_2.5.136-1_amd64.deb" \
        "$SERVER_HOST:$SERVER_INSTALL_DIR/"
    
    # TopsFactor包
    log_info "传输TopsFactor包..."
    scp "$LOCAL_TOPSRIDER_PATH/sdk/topsfactor_2.5.136-1_amd64.deb" \
        "$SERVER_HOST:$SERVER_INSTALL_DIR/"
    
    log_success "DEB包传输完成"
}

# 传输torch_gcu包
transfer_torch_gcu() {
    log_info "传输torch_gcu包..."
    
    # 检查服务器Python版本
    SERVER_PYTHON_VERSION=$(ssh "$SERVER_HOST" "python --version 2>&1 | grep -oP 'Python \K\d+\.\d+'" || echo "unknown")
    log_info "服务器Python版本: $SERVER_PYTHON_VERSION"
    
    # 根据Python版本选择合适的wheel包
    if [ "$SERVER_PYTHON_VERSION" = "3.8" ]; then
        TORCH_GCU_FILE="torch_gcu-1.10.0+2.5.136-py3.8-none-any.whl"
        log_info "选择Python 3.8版本的torch_gcu包"
    elif [ "$SERVER_PYTHON_VERSION" = "3.6" ]; then
        TORCH_GCU_FILE="torch_gcu-1.10.0-2.5.136-py3.6-none-any.whl"
        log_info "选择Python 3.6版本的torch_gcu包"
    else
        TORCH_GCU_FILE="torch_gcu-1.10.0+2.5.136-py3.8-none-any.whl"
        log_warning "未知Python版本，默认使用py3.8版本的torch_gcu包"
    fi
    
    # 传输选定的torch_gcu包
    if [ -f "$LOCAL_TOPSRIDER_PATH/framework/$TORCH_GCU_FILE" ]; then
        scp "$LOCAL_TOPSRIDER_PATH/framework/$TORCH_GCU_FILE" \
            "$SERVER_HOST:$SERVER_INSTALL_DIR/"
        log_success "torch_gcu包传输完成: $TORCH_GCU_FILE"
    else
        log_error "torch_gcu包不存在: $TORCH_GCU_FILE"
        exit 1
    fi
}

# 传输安装脚本
transfer_installation_scripts() {
    log_info "传输安装和验证脚本..."
    
    # 创建服务器端安装脚本
    cat > /tmp/server_install_topsrider.sh << 'EOF'
#!/bin/bash

# T20服务器TopsRider组件安装脚本

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

INSTALL_DIR="/tmp/topsrider_install"

main() {
    echo "🚀 开始在T20服务器上安装TopsRider组件"
    echo "============================================"
    
    cd "$INSTALL_DIR"
    
    # 备份当前环境
    log_info "备份当前环境..."
    pip freeze > /tmp/pip_backup_$(date +%Y%m%d_%H%M%S).txt
    
    # 安装DEB包
    log_info "安装DEB包..."
    
    if [ -f "tops-eccl_2.5.136-1_amd64.deb" ]; then
        log_info "安装ECCL..."
        dpkg -i tops-eccl_2.5.136-1_amd64.deb || apt-get install -f -y
    fi
    
    if [ -f "tops-sdk_2.5.136-1_amd64.deb" ]; then
        log_info "安装SDK..."
        dpkg -i tops-sdk_2.5.136-1_amd64.deb || apt-get install -f -y
    fi
    
    if [ -f "topsfactor_2.5.136-1_amd64.deb" ]; then
        log_info "安装TopsFactor..."
        dpkg -i topsfactor_2.5.136-1_amd64.deb || apt-get install -f -y
    fi
    
    # 安装torch_gcu
    log_info "安装torch_gcu..."
    TORCH_GCU_FILE=$(ls torch_gcu-*.whl | head -1)
    if [ -n "$TORCH_GCU_FILE" ]; then
        pip uninstall torch_gcu -y 2>/dev/null || true
        pip install "$TORCH_GCU_FILE"
        log_success "torch_gcu安装完成"
    else
        log_error "未找到torch_gcu wheel文件"
        exit 1
    fi
    
    # 配置环境变量
    log_info "配置环境变量..."
    cat > /root/.topsrider_env << 'ENVEOF'
# TopsRider环境配置
export ECCL_ASYNC_DISABLE=false
export ECCL_MAX_NCHANNELS=2
export ECCL_RUNTIME_3_0_ENABLE=true
export ECCL_DEBUG=INFO

# GCU设备配置
export ENFLAME_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GCU_DEVICE_COUNT=8

# 库路径配置
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/lib/python3.8/site-packages:$PYTHONPATH
ENVEOF
    
    # 添加到bashrc（如果还没有）
    if ! grep -q "source /root/.topsrider_env" /root/.bashrc 2>/dev/null; then
        echo "source /root/.topsrider_env" >> /root/.bashrc
    fi
    
    log_success "环境配置完成"
    
    # 重新加载库
    ldconfig
    
    log_success "🎉 TopsRider组件安装完成！"
    log_info "请运行验证脚本: python /tmp/verify_topsrider_installation.py"
}

main "$@"
EOF
    
    # 创建验证脚本
    cat > /tmp/verify_topsrider_installation.py << 'EOF'
#!/usr/bin/env python3
import sys
import os

def verify_torch_gcu():
    try:
        import torch
        import torch_gcu
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ torch_gcu版本: {torch_gcu.__version__}")
        print(f"✅ GCU设备数量: {torch_gcu.device_count()}")
        return True
    except Exception as e:
        print(f"❌ torch_gcu验证失败: {e}")
        return False

def verify_eccl_backend():
    try:
        import torch.distributed as dist
        
        if dist.is_available():
            print("✅ torch.distributed可用")
            
            # 设置测试环境变量
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            
            try:
                # 尝试初始化ECCL
                dist.init_process_group(backend='eccl', rank=0, world_size=1)
                print("✅ ECCL后端初始化成功")
                dist.destroy_process_group()
                return True
            except Exception as e:
                print(f"⚠️ ECCL后端测试: {e}")
                # 尝试其他后端作为对比
                try:
                    dist.init_process_group(backend='gloo', rank=0, world_size=1)
                    print("✅ gloo后端可用（作为备选）")
                    dist.destroy_process_group()
                except:
                    pass
                return False
        else:
            print("❌ torch.distributed不可用")
            return False
    except Exception as e:
        print(f"❌ 分布式后端检查失败: {e}")
        return False

def main():
    print("🔍 T20服务器TopsRider安装验证")
    print("=" * 50)
    
    # 加载环境变量
    try:
        with open('/root/.topsrider_env', 'r') as f:
            for line in f:
                if line.startswith('export '):
                    key_value = line.replace('export ', '').strip()
                    if '=' in key_value:
                        key, value = key_value.split('=', 1)
                        os.environ[key] = value
        print("✅ 环境变量已加载")
    except:
        print("⚠️ 无法加载环境变量文件")
    
    torch_gcu_ok = verify_torch_gcu()
    eccl_ok = verify_eccl_backend()
    
    print("\n📊 验证结果:")
    print("=" * 50)
    
    if torch_gcu_ok and eccl_ok:
        print("🎉 安装验证成功！ECCL后端可用")
        print("💡 现在可以使用ECCL进行分布式训练")
        return 0
    elif torch_gcu_ok:
        print("⚠️ torch_gcu安装成功，但ECCL后端可能需要进一步配置")
        print("💡 可以先使用gloo后端进行分布式训练")
        return 1
    else:
        print("❌ 安装存在问题，请检查错误信息")
        return 2

if __name__ == "__main__":
    sys.exit(main())
EOF
    
    # 传输脚本到服务器
    scp /tmp/server_install_topsrider.sh "$SERVER_HOST:$SERVER_INSTALL_DIR/"
    scp /tmp/verify_topsrider_installation.py "$SERVER_HOST:/tmp/"
    
    # 给脚本添加执行权限
    ssh "$SERVER_HOST" "chmod +x $SERVER_INSTALL_DIR/server_install_topsrider.sh"
    
    log_success "安装和验证脚本传输完成"
    
    # 清理临时文件
    rm -f /tmp/server_install_topsrider.sh /tmp/verify_topsrider_installation.py
}

# 显示传输摘要
show_transfer_summary() {
    log_info "传输摘要"
    echo "============================================"
    
    ssh "$SERVER_HOST" "
        echo '服务器端文件列表:'
        ls -la $SERVER_INSTALL_DIR/
        echo ''
        echo '磁盘使用情况:'
        du -sh $SERVER_INSTALL_DIR/
    "
}

# 显示下一步操作
show_next_steps() {
    echo ""
    log_success "🎉 文件传输完成！"
    echo "============================================"
    log_info "下一步操作："
    echo ""
    echo "1. SSH连接到服务器："
    echo "   ssh $SERVER_HOST"
    echo ""
    echo "2. 运行安装脚本："
    echo "   cd $SERVER_INSTALL_DIR"
    echo "   ./server_install_topsrider.sh"
    echo ""
    echo "3. 验证安装结果："
    echo "   python /tmp/verify_topsrider_installation.py"
    echo ""
    echo "4. 测试分布式训练："
    echo "   cd /workspace/code/MapSage_V5"
    echo "   # 修改训练脚本使用ECCL后端"
    echo ""
    log_warning "注意：安装过程需要root权限，可能需要重启服务或重新加载环境"
}

# 主函数
main() {
    echo "🚀 TopsRider组件传输脚本"
    echo "================================"
    
    check_local_package
    test_server_connection
    create_server_directories
    transfer_deb_packages
    transfer_torch_gcu
    transfer_installation_scripts
    show_transfer_summary
    show_next_steps
}

# 运行主函数
main "$@"