
## 版本申明

| 版本   | 修改内容                    | 修改时间       |
| ---- | ----------------------- | ---------- |
| v1.0 | 初始化                     | 2022/02/23 |
| v1.1 | 更新软件包目录                 | 2022/04/14 |
| v1.2 | 更新一些说明                  | 2022/06/30 |
| v1.3 | 从MS word 格式专为markdown格式 | 2022/08/10 |
| v1.4 | 更新cpumanager使用说明        | 2023/08/04 |
| v1.5 | 更新一些不适合使用的词语        | 2023/12/13 |
|      |                         |            |


> 注：阅读本文档需要有k8s、docker以及Linux基础

## 简介

Enflame提供topscloud云端部署解决方案包用于在k8s平台部署与调度Enflame GCU设备。topscloud包括k8s-device-plugin、container-toolkit、gcu-exporter、gcu-operator等这几个重要组件，例如（注：不同版本安装包里，版本会有变化）：

```bash
topscloud_2.3.0
├── container-toolkit_1.2.0
├── deployment
├── documents
├── gcu-exporter_1.3.0
├── gcu-feature-discovery_1.0.0
├── gcu-monitor-examples_1.0.0
├── gcu-operator_2.1.0
├── gcushare-device-plugin_1.0.0
├── gcushare-scheduler-extender_1.0.0
├── gcu-upgrade-manager_1.0.0
├── go-eflib_1.3.0
├── helm
├── k8s-device-plugin_k8s-v1.9+_1.2.0
├── k8s-device-plugin_k8s-v1.9-only_1.2.0
├── kubeone_1.0.0
├── node-feature-discovery_1.0.0
└── scripts
```

其中k8s-device-plugin是使得Enflame GCU 能被K8S识别且用得起来的一个插件，container-toolkit是使得Enflame GCU能被容器识别且使用的一个容器扩展工具，k8s-device-plugin 与container-toolkit组合使用以支撑Enflame GCU 在K8S系统内的资源调度，本文档主要描述k8s-device-plugin 以及container-toolkit的使用方式。

## container-toolkit

k8s-device-plugin 的使用依赖于container-toolkit, 因此这里先从container-toolkit开始介绍，在topscloud/container-toolkit安装包里涵盖以下几个文件：

```bash
container-toolkit_1.2.0/
├── daemon.json.template
├── enflame-container-toolkit_1.2.0_amd64.deb
├── enflame-container-toolkit_1.2.0_x86_64.rpm
├── install.sh
├── LICENSE.md
└── README.md
```

1） enflame-container-toolkit_1.2.0_amd64.deb 是 ubuntu 系统安装包，安装命令：

```bash
# dpkg -i enflame-container-toolkit_1.2.0_amd64.deb
```

2）enflame-container-toolkit_1.2.0_x86_64.rpm 是 tlinux ，centos，redhat系统安装包，安装命令：

```bash
# rpm -ivh enflame-container-toolkit_1.2.0_x86_64.rpm
```

3）daemon.json.template为 docker /etc/docker/daemon.json 配置模板，内容如下：

```json
{
    "default-runtime": "enflame",
    "runtimes": {
        "enflame": {
            "path": "/usr/bin/enflame-container-runtime",
            "runtimeArgs": []
        }
    },
    "registry-mirrors": ["https://mirror.ccs.tencentyun.com", "https://docker.mirrors.ustc.edu.cn"],
    "insecure-registries": ["127.0.0.1/8", "artifact.enflame.cn"],
    "max-concurrent-downloads": 10,
    "log-driver": "json-file",
    "log-level": "warn",
    "log-opts": {
        "max-size": "30m",
        "max-file": "3"
    },
    "default-shm-size": "1G",
    "default-ulimits": {
         "memlock": { "name":"memlock", "soft":  -1, "hard": -1 },
         "stack"  : { "name":"stack", "soft": 67108864, "hard": 67108864 },
         "nofile": {"name": "nofile","soft": 65536, "hard": 65536}
    },
    "data-root": "/var/lib/docker"
}
```

这里可以根据用户自己的实际情况以及具体需求，更改daemon.json.template 的内容，然后copy到 /etc/docker下，这个配置文件会将docker的default-runtime 设置成enflame container runtime。需要注意的是考虑到安全问题，安装 enflame-container-toolkit 时，当OS 的 /etc/docker 下已经有 daemon.json 存在，则不会覆盖；

4） install.sh， 一键安装脚本, 执行 ./install.sh 会将enflame-container-toolkit自动安装到系统。

5） LICENSE.md ，版权说明;

6） README.md，readme 文件

## k8s-device-plugin

在topscloud安装包里涵盖以下几个文件（注：版本会有差异），如下：

```bash
topscloud_2.3.0
├── container-toolkit_1.2.0
.......................
├── k8s-device-plugin_k8s-v1.9+_1.2.0
├── k8s-device-plugin_k8s-v1.9-only_1.2.0
.......................
```

- k8s-device-plugin_k8s-v1.9+  是 Kubernetes v1.10以及v1.10+版本插件
- k8s-device-plugin_k8s-v1.9-only 是 Kubernetes v1.9 版本专用插件

以k8s-device-plugin_k8s-v1.9+_1.2.0为例，其目录涵盖以下内容：

```bash
k8s-device-plugin_k8s-v1.9+_1.2.0/
├── docker
│   └── Dockerfile.ubuntu
├── docker-image-build.sh
├── enflame-device-plugin
├── k8s-config.sh
├── LICENSE.md
├── README.md
└── yaml
    ├── enflame-device-plugin-compat-with-cpumanager.yaml
    ├── enflame-device-plugin.yaml
    ├── enflame-vdevice-plugin.yaml
    ├── examples
    │   ├── namespace.yaml
    │   ├── pod-gcu-example.yaml
    │   └── pod-vgcu-example.yaml
    └── extensions-v1beta1-enflame-device-plugin.yaml
```

文件说明如下：

1）enflame-device-plugin，这个文件为enflame GCU k8s plugin二进制文件；

2）docker-image-build.sh，构建 k8s-device-plugin docker镜像用的脚本，执行 `docker-image-build.sh`后会生成 `k8s-device-plugin:latest`镜像，这里也可以根据自己的实际需要修改镜像名称，然后push到企业内部的docker镜像库里；

3）k8s-config.sh， 配置 k8s 系统用的，包括关闭 swap，打开 k8s plugin的支持功能，以及配置docker，可根据自己的K8S版本的具体需求根据实际情况修改；

4）插件以及示例yaml 文件如下：

```bash
└── yaml
    ├── enflame-device-plugin.yaml
    ├── enflame-vdevice-plugin.yaml
    ├── enflame-device-plugin-compat-with-cpumanager.yaml
    ├── examples
    │   ├── namespace.yaml
    │   ├── pod-gcu-example.yaml
    │   └── pod-vgcu-example.yaml
    └── extensions-v1beta1-enflame-device-plugin.yaml


```

- `enflame-device-plugin.yaml`,  k8s-device-plugin GCU 设备 yaml文件；

- `enflame-vdevice-plugin.yaml`， k8s-device-plugin vGCU 设备 yaml文件；

- `enflame-device-plugin-compat-with-cpumanager.yaml` ,  该yaml文件里添加了 `args: ["--pass-device-specs"]`标志，如果k8s集群里设置了 `--cpu-manager-ploicy=static`, 建议参采用这个 plugin yaml文件，不然会出现设备丢操作权限的问题；

- `extensions-v1beta1-enflame-device-plugin.yaml`，Kubernetes < v1.16 k8s-device-plugin GCU 设备 yaml文件， 该文件里 daemonset 配置为使用 `apiVersion: extensions/v1beta1` 的版本，仅支持 Kubernetes < v1.16；

- `examples/namespace.yaml`,  gcu pod 用例命名空间名称

- `examples/pod-gcu-example.yaml`， GCU pod 使用yaml用例

- `examples/pod-vgcu-example.yaml`， vGCU pod 使用yaml用例



5）在 第2）步生成k8s-device-plugin 镜像后，在k8s里执行

```bash
如果k8s集群未开启cpumanager功能：
# Kubectl apply -f yaml/enflame-device-plugin.yaml

如果k8s集群已开启cpumanager功能：
# Kubectl apply -f yaml/enflame-device-plugin-compat-with-cpumanager.yaml
```

即可完成enflame gcu k8s plugin 部署。

6）yaml/examples/pod-gcu-example.yaml，为enflame gcu 使用用例，其内容如下：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-gcu-example
  namespace: enflame
spec:
.............................
    resources:
      limits:
        enflame.com/gcu: 8
#        rdma/hca: 1
    volumeMounts:
   ........................
```

在 enflame.com/gcu 后填上需要的卡数，例如：`enflame.com/gcu:8`，然后执行：

```bash
# kubectl apply -f yaml/examples/pod-gcu-example.yaml
```

即可完成enflame GCU加速卡的申请，在执行`kubectl exec -it pod-gcu-example bash -n enflame`后 ， `ls /dev ` 可以看到名称为/dev/gcu\*的加速卡已经挂载完毕。



## 部署示例

### 准备一个已经部署好的K8S集群

示例OS：ubuntu 16.04 , Kubernetes v1.19.8，docker-ce v19.03.14，执行"kubectl get po -A"如下：

```bash
~# kubectl get po -A
NAMESPACE         NAME                                         READY   STATUS    RESTARTS   AGE
calico-system     calico-kube-controllers-69dfd59986-w4z9p     1/1     Running   0          34s
calico-system     calico-node-7k6nv                            1/1     Running   0          34s
calico-system     calico-typha-54cf6fd848-tg6qt                1/1     Running   0          35s
kube-system       coredns-86dfcb4f6f-xbsnx                     1/1     Running   0          40s
kube-system       coredns-86dfcb4f6f-xf4hf                     0/1     Running   0          40s
kube-system       etcd-sse-lab-inspur-002                      0/1     Running   0          46s
kube-system       kube-apiserver-sse-lab-inspur-002            1/1     Running   0          46s
kube-system       kube-controller-manager-sse-lab-inspur-002   1/1     Running   0          46s
kube-system       kube-proxy-ghhnn                             1/1     Running   0          40s
kube-system       kube-scheduler-sse-lab-inspur-002            0/1     Running   0          46s
tigera-operator   tigera-operator-7cdb76dd8b-4r4jn             1/1     Running   0          40s
```

### 确保GCU驱动已经安装好

执行 "cat /sys/module/enflame/version " ,如果已安装好驱动则会显示一个驱动版本号，例如：`1.0.20230110`， 如果没装好则需要安装。如果驱动正常，执行"ls /dev/gcu\*" 可以查找出enflame gcu 设备号 从 gcu0 -- gcu7 (注：后续GCU的命名会全部改成GCU)，如下：

```bash
# cat /sys/module/enflame/version
1.0.20230110
# ls /dev/gcu*
/dev/gcu0  /dev/gcu1  /dev/gcu2  /dev/gcu3  /dev/gcu4  /dev/gcu5  /dev/gcu6  /dev/gcu7
```

### 安装container-toolkit

在topscloud_xxx内执行" cd container-toolkit_x.x.x/; ./install.sh"，这个步骤会把enflame-container-toolkit 安装进系统，例如：

```bash
topscloud_2.3.0/container-toolkit_1.2.0# ./install.sh
(Reading database ... 352604 files and directories currently installed.)
Preparing to unpack enflame-container-toolkit_1.2.0_amd64.deb ...
Unpacking enflame-container-toolkit (1.2.0) over (1.1.0) ...
Setting up enflame-container-toolkit (1.2.0) ...
[INFO] enflame-container-toolkit had been installed
[INFO] log dir: /var/log/enflame
[INFO] config dir: /etc/enflame-container-runtime
[INFO] ldconfig...
[INFO] Docker service is restarting...
[INFO] Docker service had been restarted
Processing triggers for libc-bin (2.23-0ubuntu11.3) ...
[INFO] systemctl restart docker

```

> 注：

- 这一步需要注意daemon.json 的内容要根据自己的实际情况按需修改，再安装进/etc/docker下，默认如果/etc/docker/daemon.json 已存在则不会覆盖；

- 要确保deamon.json 里 \"default-runtime\": 为\"enflame\"， 不然docker识别不了enflame GCU；

- 对应的logs目录为/var/log/enflame/enflame-container-runtime，可以从这里获取container-toolkit 的运行日志信息；

### 安装 k8s-device-plugin

1） 生成k8s-device-plugin:latest镜像

以kubernetes v1.19.8版本为例，进入topscloud\_\<VERSION\>/k8s-device-plugin_k8s-v1.9+\_<VERSION\>目录，执行："./docker-image-build.sh" 这一步则会生成k8s-device-plugin:latest镜像，如下：

```bash
# ./docker-image-build.sh
Sending build context to Docker daemon  8.101MB
Step 1/4 : FROM ubuntu:18.04
 ---> b6f507652425
Step 2/4 : ENV ENFLAME_VISIBLE_DEVICES=all
 ---> Running in 3b3ecf3770f5
Removing intermediate container 3b3ecf3770f5
 ---> 866afb1726b6
Step 3/4 : COPY ./enflame-device-plugin /usr/bin/enflame-device-plugin
 ---> 4317b855662d
Step 4/4 : ENTRYPOINT ["/usr/bin/enflame-device-plugin"]
 ---> Running in a54e3049caf2
Removing intermediate container a54e3049caf2
 ---> 89c48331c748
Successfully built 89c48331c748
Successfully tagged artifact.enflame.cn/enflame_docker_images/enflame/k8s-device-plugin:latest
```

通过修改 `docker-image-build.sh` 里的tag内容，可以自我定制镜像名称,如下：

```bash
# cat docker-image-build.sh
#!/bin/bash

docker build --tag artifact.enflame.cn/enflame_docker_images/enflame/k8s-device-plugin:latest --file docker/Dockerfile.ubuntu .
```

2）Apply插件配置文件

执行`kubectl apply -f yaml/enflame-device-plugin.yaml`，会输出`daemonset.apps/enflame-device-plugin-daemonset created` 信息，如下：

```bash
# kubectl apply -f yaml/enflame-device-plugin.yaml
daemonset.apps/enflame-device-plugin-daemonset created
```

执行`kubectl get po -A` 查看一下，新增了一个 `enflame-device-plugin-daemonset-bdrv9` ,如下：

```bash
# kubectl get po -A
NAMESPACE          NAME                                         READY   STATUS    RESTARTS   AGE
calico-apiserver   calico-apiserver-7f66bbc9cf-p7gvs            1/1     Running   1          17m
calico-apiserver   calico-apiserver-7f66bbc9cf-xmfhw            1/1     Running   1          17m
calico-system      calico-kube-controllers-69dfd59986-w4z9p     1/1     Running   2          20m
calico-system      calico-node-7k6nv                            1/1     Running   4          20m
calico-system      calico-typha-54cf6fd848-tg6qt                1/1     Running   4          20m
kube-system        coredns-86dfcb4f6f-xbsnx                     1/1     Running   2          20m
kube-system        coredns-86dfcb4f6f-xf4hf                     1/1     Running   2          20m
kube-system        enflame-device-plugin-daemonset-bdrv9        1/1     Running   0          61s
kube-system        etcd-sse-lab-inspur-002                      1/1     Running   3          20m
kube-system        kube-apiserver-sse-lab-inspur-002            1/1     Running   4          20m
kube-system        kube-controller-manager-sse-lab-inspur-002   1/1     Running   3          20m
kube-system        kube-proxy-ghhnn                             1/1     Running   3          20m
kube-system        kube-scheduler-sse-lab-inspur-002            1/1     Running   3          20m
tigera-operator    tigera-operator-7cdb76dd8b-4r4jn             1/1     Running   5          20m
```

3） 申请gcu使用

执行：`kubectl apply -f yaml/examples/namespace.yaml`，创建一个新的用户`namespace: enflame`， 如下：

```bash
# kubectl apply -f yaml/examples/namespace.yaml
namespace/enflame created
```

根据实际情况编辑 `yaml/examples/pod-gcu-example.yaml`，例如，申请1个 GCU资源，则设置 `enflame.com/gcu:1` ，如下：

```yaml
apiVersion: v1
kind: Pod
.......................
    resources:
      limits:
        enflame.com/gcu: 1
#        rdma/hca: 1
    volumeMounts:
  .........................
```

执行：`kubectl apply -f yaml/examples/pod-gcu-example.yaml `， 正常情况下，会生成一个`pod-gcu-example`用例，如下：

```bash
# kubectl apply -f yaml/examples/pod-gcu-example.yaml
pod/pod-gcu-example created

# kubectl get po -A
NAMESPACE          NAME                                         READY   STATUS    RESTARTS   AGE
calico-apiserver   calico-apiserver-7f66bbc9cf-p7gvs            1/1     Running   1          21m
calico-apiserver   calico-apiserver-7f66bbc9cf-xmfhw            1/1     Running   1          21m
calico-system      calico-kube-controllers-69dfd59986-w4z9p     1/1     Running   2          24m
calico-system      calico-node-7k6nv                            1/1     Running   4          24m
calico-system      calico-typha-54cf6fd848-tg6qt                1/1     Running   4          24m
enflame            pod-gcu-example                              1/1     Running   0          51s
kube-system        coredns-86dfcb4f6f-xbsnx                     1/1     Running   2          24m
kube-system        coredns-86dfcb4f6f-xf4hf                     1/1     Running   2          24m
kube-system        enflame-device-plugin-daemonset-bdrv9        1/1     Running   0          5m16s
kube-system        etcd-sse-lab-inspur-002                      1/1     Running   3          24m
kube-system        kube-apiserver-sse-lab-inspur-002            1/1     Running   4          24m
kube-system        kube-controller-manager-sse-lab-inspur-002   1/1     Running   3          24m
kube-system        kube-proxy-ghhnn                             1/1     Running   3          24m
kube-system        kube-scheduler-sse-lab-inspur-002            1/1     Running   3          24m
tigera-operator    tigera-operator-7cdb76dd8b-4r4jn             1/1     Running   5          24m
```

执行`kubectl exec -it pod-gcu-example bash -n enflame` 进入这个pod，`ls /dev/` 可以看到已经申请成功一个 GCU在/dev目录下，如下：

```bash
# kubectl exec -it pod-gcu-example bash -n enflame
kubectl exec [POD] [COMMAND] is DEPRECATED and will be removed in a future version. Use kubectl exec [POD] -- [COMMAND] instead.
:/home# ls /dev/
core  gcu0  fd  full  mqueue  null  ptmx  pts  random  shm  stderr  stdin  stdout  termination-log  tty  urandom  zero
:/home# ls /dev/gcu*
/dev/gcu0
```

至此，就可以在这个pod里根据实际需要在GCU上跑业务。


### 使用NUMA亲和性调度

k8s-device-plugin从v1.4.0版本开始，提供了NUMA亲和性调度功能，但如果想使该功能生效，还需要手动修改k8s相关配置来启动CPUManager和TopologyManager功能。

1）停止kubelet服务
```bash
# systemctl stop kubelet.service
```

2）删除CPUmanager文件
```bash
# rm /var/lib/kubelet/cpu_manager_state
```

3）编辑kubelet配置文件
请注意，具体的配置文件路径可能会因Kubernetes版本和安装方式而有所不同。本文档是使用kubeone部署的k8s v1.20版本集群，配置文件路径：/var/lib/kubelet/config.yaml。

如果你是使用其它工具搭建的k8s集群，那么以下信息供参考：
- 使用kubeadm部署的集群，配置文件一般位于/etc/systemd/system/kubelet.service.d/10-kubeadm.conf。
- 尝试查看kubelet进程信息来查找配置文件路径。

```bash
# ps -ef|grep kubelet
root      21180      1  5 Nov21 ?        02:23:37 /usr/bin/kubelet --bootstrap-kubeconfig=/etc/kubernetes/bootstrap-kubelet.conf --kubeconfig=/etc/kubernetes/kubelet.conf --config=/var/lib/kubelet/config.yaml --network-plugin=cni --pod-infra-container-image=sea.hub:5000/pause:3.2
```
其中，--config就是配置文件路径。
以上信息仅供参考，如果仍无法确认配置文件路径，请查阅相关文献或联系您的集群管理员。

打开并编辑kubelet配置文件：
```yaml
# vim /var/lib/kubelet/config.yaml
...
configMapAndSecretChangeDetectionStrategy: Watch
containerLogMaxFiles: 5
containerLogMaxSize: 10Mi
contentType: application/vnd.kubernetes.protobuf
cpuCFSQuota: true
cpuCFSQuotaPeriod: 100ms
cpuManagerPolicy: static       # 默认为none，需修改为static
systemReserved:                # 新增systemReserved，为系统应用预留一定资源
  cpu: "4"
  memory: 8G
kubeReserved:                  # 新增kubeReserved，为k8s组件应用预留一定资源
  cpu: "2"
  memory: 2G
topologyManagerPolicy: best-effort   # 新增字段
cpuManagerReconcilePeriod: 10s
enableControllerAttachDetach: true
enableDebuggingHandlers: true
enforceNodeAllocatable:
- pods
eventBurst: 10
...
```

4）启动kubelet服务
```bash
# systemctl start kubelet.service
```

5）查看CPUmanager static开启成功
```json
# cat /var/lib/kubelet/cpu_manager_state
{"policyName":"static","defaultCpuSet":"0-151","checksum":2944298236}
```
开启之前policyName是none，开启后为指定的static。

关于更多NUMA亲和性调度的详细信息，请参考Kubernetes官方文档。


### 使用pcie Switch亲和性调度

在多卡分布式训练中，我们希望最大化提升多张GCU卡片之间的通信效率，以节省分布式训练的时间和成本。虽然使用NUMA亲和性调度策略可以使用同一NUMA节点下的卡片进行分布式训练以提升训练效率，但相比于使用同一pcie Switch下的多张卡进行训练，其通信效率仍可以进一步提升。

k8s-device-plugin从1.5.0版本开始，提供了基于pcie Switch亲和性的多卡调度策略。要使该策略生效，部署k8s-device-plugin时，使用yaml/enflame-device-plugin-pcie-switch-affinity.yaml进行部署即可。

```bash
# kubectl create -f yaml/enflame-device-plugin-pcie-switch-affinity.yaml 
clusterrole.rbac.authorization.k8s.io/k8s-device-plugin created
serviceaccount/k8s-device-plugin created
clusterrolebinding.rbac.authorization.k8s.io/k8s-device-plugin created
daemonset.apps/enflame-device-plugin-daemonset created
```

以某8卡机器为例，假设8张GCU卡的拓扑关系如下：

```bash
gcu 0-0000:11:00:0 --> bridge-0000:10:00:0 --> bridge-0000:0f:00:0 --> bridge-0000:0e:00:0 --> bridge-0000:0d:00:0 --> bridge-0000:0c:02:0 --> cpu
gcu 1-0000:12:00:0 --> bridge-0000:10:10:0 --> bridge-0000:0f:00:0 --> bridge-0000:0e:00:0 --> bridge-0000:0d:00:0 --> bridge-0000:0c:02:0 --> cpu
gcu 2-0000:15:00:0 --> bridge-0000:14:00:0 --> bridge-0000:13:00:0 --> bridge-0000:0e:04:0 --> bridge-0000:0d:00:0 --> bridge-0000:0c:02:0 --> cpu
gcu 3-0000:19:00:0 --> bridge-0000:17:10:0 --> bridge-0000:16:00:0 --> bridge-0000:0e:08:0 --> bridge-0000:0d:00:0 --> bridge-0000:0c:02:0 --> cpu
gcu 4-0000:9c:00:0 --> bridge-0000:9b:00:0 --> bridge-0000:9a:00:0 --> bridge-0000:99:00:0 --> bridge-0000:98:00:0 --> bridge-0000:97:02:0 --> cpu
gcu 5-0000:9d:00:0 --> bridge-0000:9b:10:0 --> bridge-0000:9a:00:0 --> bridge-0000:99:00:0 --> bridge-0000:98:00:0 --> bridge-0000:97:02:0 --> cpu
gcu 6-0000:a0:00:0 --> bridge-0000:9f:00:0 --> bridge-0000:9e:00:0 --> bridge-0000:99:04:0 --> bridge-0000:98:00:0 --> bridge-0000:97:02:0 --> cpu
gcu 7-0000:a4:00:0 --> bridge-0000:a3:10:0 --> bridge-0000:a2:00:0 --> bridge-0000:99:08:0 --> bridge-0000:98:00:0 --> bridge-0000:97:02:0 --> cpu
```
可以看到GCU0和GCU1挂在同一个Switch下，GCU2和GCU3属于另外两个不同的Switch；GCU4和GCU5挂在同一个Switch下，GCU6和GCU7属于另外两个不同的Switch。

1. 部署pod1申请2个GCU，pod1应该分配到GCU0和GCU1

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-gcu-example
  namespace: kube-system
spec:
  ...
    resources:
      limits:
        enflame.com/gcu: 2
  ...
```
pod部署成功后，查看GCU分配情况：

```bash
# kubectl exec -it pod-gcu-example -n kube-system bash
kubectl exec [POD] [COMMAND] is DEPRECATED and will be removed in a future version. Use kubectl exec [POD] -- [COMMAND] instead.

# ls /dev
core  fd  full  gcu0  gcu1  gcuctl  mqueue  null  ptmx  pts  random  shm  stderr  stdin  stdout  termination-log  tty  urandom  zero
```
可以看到，pod成功分配到了位于同一Switch下的GCU0和GCU1。

2. 部署pod2申请2个GCU，pod应该分配到GCU0和GCU1

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-gcu-example-2
  namespace: kube-system
spec:
  ...
    resources:
      limits:
        enflame.com/gcu: 2
  ...
```
pod部署成功后，查看GCU分配情况：

```bash
# kubectl exec -it pod-gcu-example-2 -n kube-system bash
kubectl exec [POD] [COMMAND] is DEPRECATED and will be removed in a future version. Use kubectl exec [POD] -- [COMMAND] instead.

# ls /dev
core  fd  full  gcu4  gcu5  gcuctl  mqueue  null  ptmx  pts  random  shm  stderr  stdin  stdout  termination-log  tty  urandom  zero
```
可以看到，由于GCU0和GCU1已经被占用，pod成功分配到了另一对位于同一Switch下的GCU4和GCU5。

**注意**:

1. 如果你对某个申请了GCU的pod发起了删除操作，那么在部署新的pod申请GCU之前，请确保之前删除的pod已经在k8s集群中被完全删除，否则该pod占有的GCU资源不会被释放到可用的GCU资源池中。

2. 当前实现方案在处理多pod或者多容器并发部署时，可能会出现pod保存的分配记录和pod容器实际的分配记录不一致的问题。这是因为k8s-device-plugin无法知道kubelet发送过来的容器请求属于哪个pod或者哪个容器（需要自定义调度器插件才能解决），这也是k8s插件机制当前的一个局限性。该问题对于单pod多容器场景不会造成实质性的影响。而对于多pod并发部署场景（如通过deployment控制器部署多个pod副本），可能会出现分配结果混乱，甚至分配资源失败的情况。因此，k8s-device-plugin暂时不支持多pod并发部署场景。


## 常见问题

### k8s-device-plugin 版本差异

- k8s-device-plugin_k8s-v1.9+_{VERSION} 是 Kubernetes v1.10以及v1.10+版本插件
- k8s-device-plugin_k8s-v1.9-only_{VERSION} 是 Kubernetes v1.9 版本专用插件

### 对k8s版本有什么要求？

 Kubernetes > 1.10 才能正式支持device plugin 功能，因此建议选用 Kubernetes > 1.10 版本：

DevicePlugins   false   Alpha   1.8     1.9
DevicePlugins   true    Beta    1.10

### k8s-device-plugin有什么依赖条件？

k8s-device-plugin 的依赖条件如下：

* enflame驱动需要安装；
* container-toolkit 需要安装；
* docker 需要将enflame 配置为默认的runtime；
* Kubernetes 版本 >= 1.10；
* 确保 Kubelet以 `--feature-gates=DevicePlugins=true`  的配置启动；

### 出现k8s找不到GCU 的情况

1） 检查驱动是否已经安装好，如果没装好，安装驱动；

2） 检查container-toolkit 是否已经安装好，如果没装好，安装container-toolkit；

3） 确保将 enflame runtime 设置为默认的 docker runtime ，编辑 `/etc/docker/daemon.json`， 添加如下内容

```json
{
    "default-runtime": "enflame",
    "runtimes": {
        "enflame": {
            "path": "/usr/bin/enflame-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

然后重启docker

```bash
# systemctl restart docker
```

### 如何修复 enflame.sock: bind: address already in use 错误？

如果出现类似以下错误：

> 2020/06/12 13:11:17 Could not start device plugin for 'enflame.com/gcu': listen unix /var/lib/kubelet/device-plugins/enflame.sock: bind: address already in use

 那么删除 enflame.sock, 然后重启 kubelet 服务.

```bash
# rm /var/lib/kubelet/device-plugins/enflame.sock
# systemctl restart kubelet
```

### 如何关闭swap？

执行 `swapoff  -a` 命令，例如：

```bash
# swapoff -a
```

### DevicePlugins  feature-gates 如何配置？

在 `/etc/systemd/system/kubelet.service` 里添加 `--feature-gates=DevicePlugins=true`, 然后重启kubelet，例如：

```shell
# systemctl daemon-reload
# systemctl restart kubelet
```

### 要正常使用containerd，如何配置DevicePlugin？

需要在containerd配置中，将容器运行时指向enflame runtime，在`/etc/containerd/config.toml` 中找到`[plugins."io.containerd.grpc.v1.cri".containerd]`并替换为以下配置后，重启containerd服务，device plugin就可以正常工作， 配置如下：

```toml
[plugins."io.containerd.grpc.v1.cri".containerd]
      default_runtime_name = "enflame"
      disable_snapshot_annotations = true
      discard_unpacked_layers = false
      no_pivot = false
      snapshotter = "overlayfs"

      [plugins."io.containerd.grpc.v1.cri".containerd.default_runtime]
        base_runtime_spec = ""
        container_annotations = []
        pod_annotations = []
        privileged_without_host_devices = false
        runtime_engine = ""
        runtime_root = ""
        runtime_type = ""

        [plugins."io.containerd.grpc.v1.cri".containerd.default_runtime.options]

      [plugins."io.containerd.grpc.v1.cri".containerd.runtimes]

        [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc]
          base_runtime_spec = ""
          container_annotations = []
          pod_annotations = []
          privileged_without_host_devices = false
          runtime_engine = ""
          runtime_root = ""
          runtime_type = "io.containerd.runc.v1"

          [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options]
            SystemdCgroup = true
        [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.enflame]
            privileged_without_host_devices = false
            runtime_engine = ""
            runtime_root = ""
            runtime_type = "io.containerd.runc.v2"
            [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.enflame.options]
              BinaryName = "/usr/bin/enflame-container-runtime"
              SystemdCgroup = true
```

### 驱动升级后，会影响container-toolkit和k8s-device-plugin吗？

驱动升级会影响container-toolkit和k8s-device-plugin，可以选择以下两种方法进行升级：

1）直接删除k8s插件的pod，等待控制器创建新的pod。
```bash
# kubectl delete pod enflame-device-plugin-daemonset-bdrv9 -n kube-system
```

2）直接卸载已经安装的k8s-device-plugin，然后重新安装。
```bash
# kubectl delete -f yaml/enflame-device-plugin.yaml
# kubectl create -f yaml/enflame-device-plugin.yaml
```



### yaml文件里的镜像找不到

本组件所提供的yaml文件，用户可以根据自己的实际使用修改。例如以下镜像文件，这些镜像文件需要根据用户自己的实际使用情况修改，定义成自己在用的名称或者镜像库路径：

```bash
文件yaml/enflame-device-plugin.yaml里的
"image: artifact.enflame.cn/enflame_docker_images/enflame/k8s-device-plugin:latest"

文件yaml/examples/pod-gcu-example.yaml里的
"image: ubuntu:18.04"
```

### cpu-manager-policy=static 导致EFML Initialize failed

当K8S集群里开启 `cpu-manager-policy=static`, pod内运行 efsmi 出现以下错误：

```bash
HAL initialization failed
Failed to initialize efml lib
Initialize failed...
```

这需要 采用 cpumanager兼容的yaml配置文件`enflame-device-plugin-compat-with-cpumanager.yaml`，执行:

```bash
# kubectl create -f yaml/enflame-device-plugin-compat-with-cpumanager.yaml
```

### 执行systemctl daemon-reload 设备操作权限丢失

这是runc 1.1.3引入的一个bug，参考URL ：`https://github.com/opencontainers/runc/issues/3671`

这需要升级 runc 到 1.1.7及以上版本。
