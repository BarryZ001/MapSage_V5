
## 版本申明

| 版本 | 修改内容               | 修改时间   |
| ---- | --------------------- | ---------- |
| v1.0 | 初始化                 | 6/25/2023  |



## 简介

Node Feature Discovery是一款部署在k8s集群上的用于检测硬件功能和系统配置的 Kubernetes 插件。


## 部署示例

### 部署要求

- 安装docker
- k8s集群版本高于1.8
- 集群中安装了GCU驱动
- 集群中安装了Enflame Container Toolkit
- 集群中安装了Enflame K8s Device Plugin


### 制作NFD组件镜像

在topscloud的release包中，打开NFD的目录：

```
node-feature-discovery_<VERSION>
├── bin
│   ├── nfd-master
│   ├── nfd-topology-updater
│   └── nfd-worker
├── docker
│   └── Dockerfile.ubuntu
├── build-image.sh
└── README.md


```

执行`build-image.sh`脚本一键构建GFD组件镜像：

```
node-feature-discovery_<VERSION> # ./build-image.sh
1. Clear old image if exist
Untagged: artifact.enflame.cn/enflame_docker_images/enflame/node-feature-discovery:v0.11.3
Deleted: sha256:a3ede8c067a2fb2ecfe8aab016e9d0f66a6e168926fbc62fa1d085b777178173
Deleted: sha256:12d3407af379fc022d3958babf676f0e86b13e8cd80ca8afc35294affceae3ca
Deleted: sha256:f757643fb311599cb891599084aae1c48bde3f7a3e176c452c69247013df8ca1
Deleted: sha256:7c0d2f44881b15bc63601eb2e450da8684c5a9f8eb1a6698516b1911974711b1
2. Build image start...
image name:artifact.enflame.cn/enflame_docker_images/enflame/node-feature-discovery, image version:v0.11.3
Sending build context to Docker daemon  239.9MB
Step 1/4 : FROM ubuntu:18.04
 ---> 35b3f4f76a24
Step 2/4 : WORKDIR .
 ---> Running in 2749e3269b4c
Removing intermediate container 2749e3269b4c
 ---> 8e4cf8612d30
Step 3/4 : ENV GRPC_GO_LOG_SEVERITY_LEVEL="INFO"
 ---> Running in 21a1b383e724
Removing intermediate container 21a1b383e724
 ---> 32e8e3254fd0
Step 4/4 : COPY ./bin/* /usr/bin/
 ---> 4e89e248dfc1
Successfully built 4e89e248dfc1
Successfully tagged artifact.enflame.cn/enflame_docker_images/enflame/node-feature-discovery:v0.11.3
build image success
3. save image to ./images

```

### 部署使用

当前node-feature-discovery通过 gcu-operator_2.0 进行部署，在构建好node-feature-Discovery的镜像后，可以通过gcu-operator进行部署。相应过程可以参考《gcu_operator_2.0用户使用手册》。


## 自定义node-feature-discovery镜像名称

build-image.sh 里默认的镜像路径与名称为: `artifact.enflame.cn/enflame_docker_images/enflame/node-feature-discovery:v0.11.3`，如下：

```
ORIGIN_NAME="node-feature-discovery"
VERSION="v0.11.3"
REPO="artifact.enflame.cn/enflame_docker_images/enflame"

```

可以根据自己的需要自定义这个镜像路径与名称。


## node-feature-discovery功能介绍

topscloud里的node-feature-discovery与开源版本100%兼容，其他相关介绍见文档：

```
https://kubernetes-sigs.github.io/node-feature-discovery/stable/get-started/index.html

```

