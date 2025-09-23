# T20服务器TopsRider组件安装指导

## 📋 概述

基于对本地TopsRider安装包的分析，我们已经确定了需要在T20服务器上安装的关键组件。本指导将帮助你在服务器上正确安装ECCL、SDK和torch_gcu组件。

## 🎯 安装目标

解决T20服务器上的ECCL后端不可用问题：`Invalid backend: 'eccl'`

## 📦 需要传输的关键文件

基于分析结果，需要从本地传输到服务器的文件：

### ECCL核心组件
```
distributed/tops-eccl_2.5.136-1_amd64.deb
```

### SDK组件
```
sdk/tops-sdk_2.5.136-1_amd64.deb
sdk/topsfactor_2.5.136-1_amd64.deb
```

### torch_gcu包
```
framework/torch_gcu-1.10.0+2.5.136-py3.8-none-any.whl  # Python 3.8
framework/torch_gcu-1.10.0-2.5.136-py3.6-none-any.whl  # Python 3.6
```

## 🚀 服务器安装步骤

### 第一步：传输文件到服务器

使用scp命令将必要文件传输到T20服务器：

```bash
# 创建服务器端目录
ssh root@10.20.52.143 "mkdir -p /tmp/topsrider_install"

# 传输ECCL包
scp /Users/barryzhang/myDev3/MapSage_V5/test/topsrider_extracted/TopsRider_t2x_2.5.136_deb_amd64/distributed/tops-eccl_2.5.136-1_amd64.deb root@10.20.52.143:/tmp/topsrider_install/

# 传输SDK包
scp /Users/barryzhang/myDev3/MapSage_V5/test/topsrider_extracted/TopsRider_t2x_2.5.136_deb_amd64/sdk/tops-sdk_2.5.136-1_amd64.deb root@10.20.52.143:/tmp/topsrider_install/

scp /Users/barryzhang/myDev3/MapSage_V5/test/topsrider_extracted/TopsRider_t2x_2.5.136_deb_amd64/sdk/topsfactor_2.5.136-1_amd64.deb root@10.20.52.143:/tmp/topsrider_install/

# 传输torch_gcu包（根据服务器Python版本选择）
scp /Users/barryzhang/myDev3/MapSage_V5/test/topsrider_extracted/TopsRider_t2x_2.5.136_deb_amd64/framework/torch_gcu-1.10.0+2.5.136-py3.8-none-any.whl root@10.20.52.143:/tmp/topsrider_install/
```

### 第二步：在服务器上执行安装

SSH连接到服务器并执行安装：

```bash
ssh root@10.20.52.143
```

在服务器上执行以下命令：

#### 1. 安装DEB包
```bash
cd /tmp/topsrider_install

# 安装ECCL
dpkg -i tops-eccl_2.5.136-1_amd64.deb
apt-get install -f  # 修复依赖问题

# 安装SDK
dpkg -i tops-sdk_2.5.136-1_amd64.deb
apt-get install -f

# 安装TopsFactor
dpkg -i topsfactor_2.5.136-1_amd64.deb
apt-get install -f
```

#### 2. 安装torch_gcu
```bash
# 检查Python版本
python --version

# 安装torch_gcu（根据Python版本选择）
pip uninstall torch_gcu -y  # 卸载旧版本
pip install torch_gcu-1.10.0+2.5.136-py3.8-none-any.whl
```

#### 3. 配置环境变量
```bash
# 创建环境配置文件
cat > /root/.topsrider_env << 'EOF'
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
EOF

# 添加到bashrc
echo "source /root/.topsrider_env" >> /root/.bashrc
source /root/.bashrc
```

### 第三步：验证安装

在服务器上运行验证脚本：

```python
# 创建验证脚本
cat > /tmp/verify_installation.py << 'EOF'
#!/usr/bin/env python3
import sys

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
        
        # 尝试初始化ECCL后端
        if dist.is_available():
            print("✅ torch.distributed可用")
            
            # 检查ECCL后端
            try:
                # 设置环境变量
                import os
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
                os.environ['RANK'] = '0'
                os.environ['WORLD_SIZE'] = '1'
                
                # 尝试初始化ECCL
                dist.init_process_group(backend='eccl', rank=0, world_size=1)
                print("✅ ECCL后端初始化成功")
                dist.destroy_process_group()
                return True
            except Exception as e:
                print(f"⚠️ ECCL后端测试失败: {e}")
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
    
    torch_gcu_ok = verify_torch_gcu()
    eccl_ok = verify_eccl_backend()
    
    print("\n📊 验证结果:")
    print("=" * 50)
    
    if torch_gcu_ok and eccl_ok:
        print("🎉 安装验证成功！ECCL后端可用")
        return 0
    elif torch_gcu_ok:
        print("⚠️ torch_gcu安装成功，但ECCL后端可能需要进一步配置")
        return 1
    else:
        print("❌ 安装存在问题")
        return 2

if __name__ == "__main__":
    sys.exit(main())
EOF

# 运行验证
python /tmp/verify_installation.py
```

## 🔧 故障排除

### 如果ECCL后端仍然不可用

1. **检查库文件**：
```bash
find /usr -name "*eccl*" 2>/dev/null
ldconfig -p | grep eccl
```

2. **检查环境变量**：
```bash
env | grep ECCL
```

3. **重新加载环境**：
```bash
source /root/.topsrider_env
ldconfig
```

4. **检查进程组初始化**：
```bash
# 在Python中测试
python -c "
import torch.distributed as dist
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
try:
    dist.init_process_group(backend='eccl', rank=0, world_size=1)
    print('ECCL初始化成功')
    dist.destroy_process_group()
except Exception as e:
    print(f'ECCL初始化失败: {e}')
"
```

## 📋 安装后测试

安装完成后，使用我们之前创建的分布式训练脚本进行测试：

```bash
cd /workspace/code/MapSage_V5

# 测试ECCL后端
python scripts/test_gloo_distributed.py  # 先用gloo测试基础功能

# 如果ECCL安装成功，修改脚本使用ECCL后端
# 将 backend='gloo' 改为 backend='eccl'
```

## ⚠️ 重要注意事项

1. **备份环境**：安装前备份当前的torch_gcu版本
2. **版本兼容性**：确保torch_gcu版本与PyTorch版本兼容
3. **权限问题**：某些操作需要root权限
4. **网络连接**：确保服务器可以访问apt源来解决依赖

## 🔄 回滚方案

如果安装失败，可以回滚：

```bash
# 卸载安装的包
dpkg -r tops-eccl tops-sdk topsfactor
pip uninstall torch_gcu -y

# 恢复原来的torch_gcu（如果有备份）
# pip install torch_gcu==原版本
```

---

**下一步**：我将创建自动化的scp传输脚本和服务器端安装脚本，让整个过程更加自动化。