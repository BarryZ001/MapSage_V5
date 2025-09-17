# T20服务器文件上传命令 - 分步执行

由于网络问题无法使用git pull，请按以下步骤逐个执行scp命令上传文件。

## 服务器信息
- **SSH连接**: `ssh -p 60025 root@117.156.108.234`
- **目标目录**: `/root/mapsage_project/code/MapSage_V5/`

## 步骤1: 上传文档文件

```bash
# T20服务器环境配置文档
scp -P 60025 "docs/T20服务器环境配置.md" root@117.156.108.234:/root/mapsage_project/code/MapSage_V5/docs/

# 权重文件准备指导
scp -P 60025 "docs/权重文件准备指导.md" root@117.156.108.234:/root/mapsage_project/code/MapSage_V5/docs/

# 燧原T20适配指导
scp -P 60025 "docs/燧原T20适配指导.md" root@117.156.108.234:/root/mapsage_project/code/MapSage_V5/docs/

# 阶段0执行指导
scp -P 60025 "docs/阶段0执行指导.md" root@117.156.108.234:/root/mapsage_project/code/MapSage_V5/docs/

# 阶段0验证清单
scp -P 60025 "docs/阶段0验证清单.md" root@117.156.108.234:/root/mapsage_project/code/MapSage_V5/docs/

# T20集群环境配置手册
scp -P 60025 "docs/T20集群TopsRider软件栈环境配置成功手册.md" root@117.156.108.234:/root/mapsage_project/code/MapSage_V5/docs/
```

## 步骤2: 上传脚本文件

```bash
# 燧原T20适配脚本
scp -P 60025 scripts/adapt_to_enflame_t20.py root@117.156.108.234:/root/mapsage_project/code/MapSage_V5/scripts/

# 快速适配脚本
scp -P 60025 scripts/quick_adapt_t20.sh root@117.156.108.234:/root/mapsage_project/code/MapSage_V5/scripts/

# 路径更新脚本
scp -P 60025 scripts/update_paths_for_t20.py root@117.156.108.234:/root/mapsage_project/code/MapSage_V5/scripts/

# 验证脚本(已更新路径)
scp -P 60025 scripts/validate_tta.py root@117.156.108.234:/root/mapsage_project/code/MapSage_V5/scripts/
```

## 步骤3: 设置脚本执行权限

```bash
# 连接到服务器设置权限
ssh -p 60025 root@117.156.108.234

# 在服务器上执行以下命令
cd /root/mapsage_project/code/MapSage_V5
chmod +x scripts/quick_adapt_t20.sh
chmod +x scripts/update_paths_for_t20.py
chmod +x scripts/adapt_to_enflame_t20.py
```

## 步骤4: 验证上传结果

```bash
# 检查文档文件
ls -la /root/mapsage_project/code/MapSage_V5/docs/ | grep -E '(T20|权重|燧原|阶段0)'

# 检查脚本文件
ls -la /root/mapsage_project/code/MapSage_V5/scripts/ | grep -E '(adapt_to_enflame|quick_adapt|update_paths)'
```

## 快速执行选项

如果你想一次性执行所有命令，可以运行:

```bash
# 在本地Mac上执行
./upload_to_t20.sh
```

## 上传完成后的下一步

1. **连接到T20服务器**:
   ```bash
   ssh -p 60025 root@117.156.108.234
   ```

2. **进入项目目录**:
   ```bash
   cd /root/mapsage_project/code/MapSage_V5
   ```

3. **查看执行指导**:
   ```bash
   cat docs/阶段0执行指导.md
   ```

4. **开始适配流程**:
   ```bash
   # 启动容器
   docker run -dit \
     --name t20_mapsage_env \
     --privileged \
     --ipc=host \
     --network=host \
     -v /root/mapsage_project/code:/workspace/code \
     -v /data/datasets:/workspace/data \
     -v /root/mapsage_project/weights:/workspace/weights \
     -v /root/mapsage_project/outputs:/workspace/outputs \
     ubuntu:20.04
   
   # 进入容器
   docker exec -it t20_mapsage_env bash
   
   # 在容器内执行适配
   cd /workspace/code
   ./scripts/quick_adapt_t20.sh
   ```

## 注意事项

- 确保T20服务器上的目标目录存在
- 如果遇到权限问题，可能需要先创建目录:
  ```bash
  ssh -p 60025 root@117.156.108.234 "mkdir -p /root/mapsage_project/code/MapSage_V5/{docs,scripts}"
  ```
- 上传过程中如果某个文件失败，可以单独重试该命令