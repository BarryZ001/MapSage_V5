
## 版本申明

| 版本   | 修改内容 | 修改时间      |
|------|------|-----------|
| v1.0 | 初始化  | 5/12/2022 |
| v1.1 | 新增二次开发说明  | 5/22/2023 |
| v1.2 | 更新组件文档说明  | 6/25/2023 |
| v1.3 | 更新组件说明  | 7/03/2023 |
| v1.4 | 更新一些词语  | 12/12/2023 |


## 简介

GCU 是燧原的 AI 计算加速设备，TopsCloud 是基于 GCU 的K8S以及容器化解决方案。TopsCloud 支持 k8s部署和运维，
包括kubeone，gcu-operator, container-toolkit，k8s-device-plugin，gcu-exporter，gcushare等主要组件，其中kubeone负责部署k8s集群，gcu-operator负责部署和GCU相关组件，container-toolkit与k8s-device-plugin用于支持GCU k8s调度， gcu-exporter是gcu运行数据采集组件，gcushare是GCU k8s空分组件。


## 专有名词解释

| 名词                        | 描述                                              |
|---------------------------|-------------------------------------------------|
| TopsCloud                 | 燧原基于GCU的K8S集群化解决方案                              |
| enflame-container-toolkit | 燧原基于GCU的容器化插件，用于在非特权模式下提供容器内GCU设备的支持，目前支持docker |
| k8s-device-plugin   | 燧原基于GCU的k8s插件，向k8s集群注册GCU资源                     |
| gcu-exporter   | 燧原GCU的数据采集组件，提供gcu设备运行相关指标的时序数据             |
| gcu-feature-discovery | 用于给GCU设备打上标签                                    |
| gcu-operator  | 自定义资源GcuResource用于自动化管理gcu软件                                     |



## 使用说明

注：当前topscloud完全离线部署功能尚未完成，有网络依赖条件的请参考相应文档本地构建镜像。


### 一键部署使用说明
topscloud提供整体部署解决方案，支持脚本一键部署包括k8s集群，gcu相关插件，网络相关插件在内的所有组件，执行步骤如下：

- 进入topscloud_<VERSION>/deployment
- 执行./setup.sh install 即可自动部署k8s集群以及gcu相关组件和网络相关组件


### 单独部署使用说明

topscloud支持组件单独部署

#### k8s集群使用说明
见kubeone用户手册，k8s_plugin用户手册，container-toolkit用户手册

#### gcu-operater使用说明
见gcu-operator用户手册

#### gcu-exporter使用说明
见gcu-exporter用户手册

#### node-feature-discovery使用说明
node-feature-discovery 保持了与开源版本一次，未做任何修改，使用说明见：

```
https://kubernetes-sigs.github.io/node-feature-discovery/stable/get-started/index.html
```

#### 其他组件
其他独立组件见相应的组件文档，确认依赖组件已就绪后，依照用户手册进行安装使用。



## 二次开发用户使用说明

### 一般用户（内网用户）
1）见相应的独立组件用户使用手册

### 需要二次开发的外网用户

1）见相应的独立组件用户使用手册；

2）安装包的Yaml配置文件里涉及 image的地方需要自我定义以及根据自己使用实际情况来修改，例如，以下文件里的image内容需要处于外网的二次开发用户参考用户文档自我DIY去定制：

```
./gcu-feature-discovery/deployments/gcu-feature-discovery-daemonset.yaml:        - image: artifact.enflame.cn/enflame_docker_images/enflame/gcu-gfd:latest
./k8s-operator/gcu-operator_2.0/enflame-resources/exporter/gcu-exporter-ds.yaml:          image: artifact.enflame.cn/enflame_docker_images/enflame/gcu-exporter:default
./k8s-operator/gcu-operator_2.0/enflame-resources/gcushare-scheduler-extender/deployment.yaml:          image: "artifact.enflame.cn/enflame_docker_images/enflame/gcushare-scheduler-extender:default"
./k8s-operator/gcu-operator_2.0/enflame-resources/gcushare-scheduler-extender/daemonset-config.yaml:      - image: "artifact.enflame.cn/enflame_docker_images/enflame/gcushare-config-manager:latest"
./k8s-operator/gcu-operator_2.0/enflame-resources/gcu-feature-discovery/gcu-feature-discovery-ds.yaml:        - image: artifact.enflame.cn/enflame_docker_images/enflame/gcu-gfd:default
./k8s-operator/gcu-operator_2.0/enflame-resources/driver/gcu-driver.yaml:          image: artifact.enflame.cn/enflame_docker_images/enflame/gcu-driver:default
./k8s-operator/gcu-operator_2.0/enflame-resources/node-feature-discovery/nfd-ds.yaml:          image: "artifact.enflame.cn/enflame_docker_images/enflame/node-feature-discovery:v0.11.3"
./k8s-operator/gcu-operator_2.0/enflame-resources/node-feature-discovery/nfd-ds.yaml:          image: "artifact.enflame.cn/enflame_docker_images/enflame/node-feature-discovery:v0.11.3"
./k8s-operator/gcu-operator_2.0/enflame-resources/gcushare-device-plugin/daemonset.yaml:      - image: "artifact.enflame.cn/enflame_docker_images/enflame/gcushare-device-plugin:default"
./k8s-operator/gcu-operator_2.0/enflame-resources/container-toolkit/gcu-docker-plugin.yaml:          image: artifact.enflame.cn/enflame_docker_images/enflame/gcu-runtime:default
./k8s-operator/gcu-operator_2.0/enflame-resources/k8s-device-plugin/enflame-device-plugin.yaml:          image: artifact.enflame.cn/enflame_docker_images/enflame/gcu-plugin:default
./prometheus/gcu-exporter/yaml/gcu-exporter.yaml:          image: artifact.enflame.cn/enflame_docker_images/enflame/gcu-exporter:latest
./prometheus/gcu-exporter/yaml/gcu-exporter-for-arm.yaml:          image: artifact.enflame.cn/enflame_docker_images/enflame/gcu-exporter:latest
./k8s-device-plugin/yaml/enflame-device-plugin.yaml:      - image: artifact.enflame.cn/enflame_docker_images/enflame/k8s-device-plugin:latest
./k8s-device-plugin/yaml_v1.9/enflame-device-plugin-v1.9.yaml:      - image: artifact.enflame.cn/enflame_docker_images/enflame/k8s-device-plugin-v1.9:latest
./k8s-device-plugin/yaml_v1.9/enflame-device-plugin-compat-with-cpumanager-v1.9.yaml:      - image: artifact.enflame.cn/enflame_docker_images/enflame/k8s-device-plugin-v1.9:latest
./k8s-device-plugin/yaml/enflame-vdevice-plugin.yaml:      - image: artifact.enflame.cn/enflame_docker_images/enflame/k8s-device-plugin:latest
./k8s-device-plugin/yaml/enflame-device-plugin-compat-with-cpumanager.yaml:      - image: artifact.enflame.cn/enflame_docker_images/enflame/k8s-device-plugin:latest

```

> 注：除了以上文件，也需要二次开发用户根据自己的实际使用情况梳理且修改相应的镜像，避免遗漏。

## FAQ

### 整体方案部署后，如何验证部署成功
进入topscloud_<VERSION>/deployment目录，执行./setup.sh install，如提示installed successfully，并且没有任何报错，表示安装成功

### 如何卸载topscloud，卸载完的状态是怎样的
进入topscloud_<VERSION>/deployment目录，执行./setup.sh uninstall, 会卸载整个k8s集群，但是会保留docker组件
