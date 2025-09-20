
## 版本申明

| 版本 | 修改内容 | 修改时间  |
| ---- | -------- | --------- |
| v1.0 | 初始化   | 5/09/2022 |
| v1.1 | 添加内容 | 8/11/2022 |
|      |          |           |



## 简介

Enflame kubeone 是基于sealer进行定制二次开发的k8s 集群部署工具，其遵循Apache-2.0 协议。当前1.0.x 版本仅支持enflame公司内网使用，外网用户请依据sealer 用户文档`http://sealer.cool/zh/getting-started/introduction.html`   自行参照使用。

enflame kubeone继承了sealer的所有优点， 可以像docker那样把整个集群制作成镜像，把所有的enflame软件栈以及依赖整体打包到集群镜像中，实现燧原训练或推理集群的快速构建、一键交付以及可靠运行。

不同于sealer ，enflame kubeone更专注于燧原软件栈打包、专有云、数据中心、离线部署这样的场景，kubeone实现整个燧原训练或推理集群的镜像化打包和交付，从而提供一种“开箱即用”的应用封装技术。



## 前置准备

- 所有节点操作系统均为ubuntu 系统，当前暂不支持centos/redhat 系列；
- 检查执行kubeone所在节点5000端口是否被占用，如果被占用应关掉相应进程
- 检查/etc/systemd/system/docker.service是否存在，如果存在建议备份后删除
- 具备root用户权限；
- 所有节点部署同一套ssh公钥，私钥部署在kubeone所在机器上/root/.ssh/id_rsa下;
- 如果是部署单机伪集群，本机需要同时部署公钥和私钥；
- 保证所有节点对外暴露`IP`互通（单机伪集群忽略）；



## 配置与命令

### 安装包

当前发布的安装包内容如下：

```
kubeone_x.x.x/
├── kubeone
├── kubeone.json.multi-nodes
├── kubeone.json.one-node
├── LICENSE
└── README.md
```

其中：

- kubeone， k8s集群部署工具；
- kubeone.json.multi-nodes， kubeone多节点配置模板；
-  kubeone.json.one-node， kubeone单节点配置模板；
- LICENSE， 协议文件；
- README.md ， 简单的README文档；



### kubeone.json

Enflame kubeone 默认的配置文件为`kubeone.json`，如果本地没有配置文件，则按照默认配置部署，其默认部署单机k8s集群，当前kubernetes默认版本为`artifact.enflame.cn/enflame_docker_images/enflame/kubernetes-v.1.20.0:1.0.1`，`kubeone.json·`格式如下：

```
{
   "masters": ["IP1", "IP2", "IP3"],
   "nodes": ["IP4", "IP5"],
   "image": "kubernetes镜像仓库路径",
   "image_version: "kubernetes版本号",
   "cluster_name": "集群名称"
}
```

kubeone.json包括以下几个元素：

- masters（主节点）：采用一个数组进行描述，里面包含所有承担master role的机器的IP地址；
- nodes（工作节点）：同样用一个数组去描述，里面包含所有承担工作节点role的机器的IP地址；
- image（kubernetes镜像）：默认为enflame内网定制的kubernetes镜像路径:`artifact.enflame.cn/enflame_docker_images/enflame/kubernetes-v1.20.0`；
- image_version(镜像版本）：用一个字符串去描述，对应我们要安装的镜像版本号，目前默认为1.0.1;
- cluster_name(集群名称)：给待部署集群定一个名字，比如： "enflame";

其中，同一个IP既可以是master，也可以是node，但作为生产环境建议每个IP只有一个角色，对于集群建议是最少1个或3个节点作为master节点。



### 命令集

#### kubeone start

直接执行该命令，不需要任何参数，默认读取位于当前工作目录下的`kubeone.json`配置文件，并按照配置文件中的描述自动部署集群。

#### kubeone stop

直接执行该命令，不需要任何参数，默认读取位于当前工作目录下的`kubeone.json`配置文件，并按照配置文件中的描述自动清理k8s环境。

#### kubeone pull

k8s镜像拉取命令，命令示例:

```
# kubeone pull artifact.enflame.cn/enflame_docker_images/enflame/kubernetes-v1.20.0:1.0.1
```

当所需镜像不在本地时，`kubeone start`会自动将镜像pull到本地


#### kubeone save

将集群镜像从本地导出，命令示例:

```
# kubeone save kubernetes-1.20.0:1.0.1 -o k8s.tar
```

#### kubeone load

从tar包中导入集群镜像，命令示例:

```
# kubeone load -i k8s.tar
```

#### kubeone --help

更多命令参考 `kubeone --help` 或 `kubeone -h`， 如下：

```
#./kubeone --help
Usage:
  kubeone [command]

Available Commands:
  help        Help about any command
  images      list all cluster images
  load        load image from a tar file
  login       login image repository
  pull        pull cloud image to local
  push        push cloud image to registry
  rmi         remove local images by name
  save        save image to a tar file
  start       start a cluster with config
  stop        stop a cluster with config

Flags:
  -d, --debug           turn on debug mode
  -h, --help            help for kubeone

Use "kubeone [command] --help" for more information about a command.

```



## 部署示例

### 单机部署

1） 不基于kubeone.json

在安装包目录直接执行`kubeone start`，会在本机快速安装默认版本为v1.20.0的k8s单机伪集群，例如：

```
# cd kubeone_1.0.1
# kubeone start
```



2） 基于kubeone.json

将kubeone安装包里的 `kubeone.json.one-node ` 改名为  kubeone.json:

```
# cp kubeone.json.one-node kubeone.json
```

根据实际情况编辑里头的内容，比如可以根据需要自我定制master节点IP、node节点IP、镜像、镜像版本、集群名称，例如：

```
{
   "masters": ["10.12.0.1"],
   "nodes": ["10.12.0.1"],
   "image": "artifact.enflame.cn/enflame_docker_images/enflame/kubernetes-v1.20.0",
   "image_version": "1.0.1",
   "cluster_name": "enflame"
}
```

然后执行：

```
# kubeone start
```



### 多机部署

1） 修改kubeone.json

将kubeone安装包里的 `kubeone.json.multi-nodes ` 改名为  `kubeone.json`：

```
# cp kubeone.json.multi-nodes kubeone.json
```

根据实际情况编辑里头的内容，比如可以根据需要自我定制master节点IP、node节点IP、镜像、镜像版本、集群名称，例如：

```
{
   "masters": ["10.12.0.1", "10.12.0.2", "10.12.0.3"],
   "nodes": ["10.12.0.4", "10.12.0.5"],
   "image": "artifact.enflame.cn/enflame_docker_images/enflame/kubernetes-v1.20.0",
   "image_version": "1.0.1",
   "cluster_name": "enflame"
}
```



2） kubeone start

kubeone.json 配置好后 ，执行 kubeone start 即可

```
# kubeone start
```




## 常见问题
### 外网环境下，kubeone无法拉取k8s镜像


需要使用EGC平台，具体请咨询CSE团队


### 如何清理系统内原有的K8S

如果需要清理原有的OS系统已安装的k8s，可以采用`kubeone stop`，这个命令可以直接清理环境。



### 如何离线部署

在燧原内网里使用`kubeone pull`，拉取已有k8s镜像，再使用`kubeone save`导出镜像成tar包，通过介质拷贝到目标机器后，再通过`kubeone load`进行导入，最后执行`kubeone start`。



### 如何安装docker-ce

ubuntu下安装docker-ce 步骤如下：

```
# apt install dkms -y
# curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
# add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(cat /etc/os-release | grep UBUNTU_CODENAME | cut -d '=' -f 2) stable"
# apt update
# apt install docker-ce -y
# systemctl enable docker
# systemctl restart docker
# docker --version
 
如果需要清理原有docker-ce再安装的话，可以参考以下步骤：
# apt purge contained.io docker-ce
# rm -rf /var/lib/containerd/
# apt install docker-ce containerd.io
```



### node(s) had taint 错误 

单机部署时，该节点既是master又是node，如果出现类似 node(s) had taint 这样的错误，例如：


```
 Type     Reason            Age   From               Message
  ----     ------            ----  ----               -------
  Warning  FailedScheduling  119s  default-scheduler  0/1 nodes are available: 1 node(s) had taint {node-role.kubernetes.io/master: }, that the pod didn't tolerate.
  Warning  FailedScheduling  119s  default-scheduler  0/1 nodes are available: 1 node(s) had taint {node-role.kubernetes.io/master: }, that the pod didn't tolerate.
 

```

处理策略

这时需要打开master节点的资源调度能力，执行一下以下命令即可修复：

```
kubectl taint nodes --all node-role.kubernetes.io/master-
```



### 出现coredns  CrashLoopBackOff  错误

例如，当k8s集群部署好后，出现类似 coredns CrashLoopBackOff 这样的错误，例如：

```
# kubectl get pod -o wide -A
NAMESPACE         NAME                                        READY   STATUS             RESTARTS   AGE   IP               NODE                NOMINATED NODE   READINESS GATES
calico-system     calico-kube-controllers-5689d4dfdf-tzh5k    1/1     Running            0          58s   100.98.178.129   develop-host   <none>           <none>
calico-system     calico-node-kzzcp                           1/1     Running            0          59s   10.8.52.16       develop-host   <none>           <none>
calico-system     calico-typha-7949c4b4db-lj72x               1/1     Running            0          59s   10.8.52.16       develop-host   <none>           <none>
kube-system       coredns-597c5579bc-7w4hx                    0/1     CrashLoopBackOff   2          66s   100.98.178.130   develop-host   <none>           <none>
kube-system       coredns-597c5579bc-pkbk9                    0/1     CrashLoopBackOff   1          66s   100.98.178.131   develop-host   <none>           <none>
kube-system       etcd-jeff-develop-host                      1/1     Running            0          84s   10.8.52.16       develop-host   <none>           <none>
kube-system       kube-apiserver-jeff-develop-host            1/1     Running            0          84s   10.8.52.16       develop-host   <none>           <none>
kube-system       kube-controller-manager-jeff-develop-host   1/1     Running            0          84s   10.8.52.16      develop-host   <none>           <none>
kube-system       kube-proxy-d2g7q                            1/1     Running            0          66s   10.8.52.16       develop-host   <none>           <none>
kube-system       kube-scheduler-jeff-develop-host            1/1     Running            0          83s   10.8.52.16       develop-host   <none>           <none>
tigera-operator   tigera-operator-86c4fc874f-dsz4n            1/1     Running            0          66s   10.8.52.16       develop-host   <none>           <none>

```

处理策略：

这时需要在`/etc/resolv.conf`里添加合适的 nameserver,例如：

```
# Dynamic resolv.conf(5) file for glibc resolver(3) generated by resolvconf(8)
#     DO NOT EDIT THIS FILE BY HAND -- YOUR CHANGES WILL BE OVERWRITTEN
nameserver 10.12.31.11
nameserver 172.16.11.11

```

然后再重启coredns，例如：

```
# kubectl delete po coredns-597c5579bc-7w4hx -n kube-system
pod "coredns-597c5579bc-7w4hx" deleted
# kubectl delete po coredns-597c5579bc-pkbk9 -n kube-system
pod "coredns-597c5579bc-pkbk9" deleted
```



### 出现`Failed to execute operation: File exists`错误

当部署k8s集群 出现类似 `Failed to execute operation: File exists` 错误时，例如：

```
vm.swappiness = 0
net.ipv4.ip_forward = 1
/usr/sbin/ufw
防火墙在系统启动时自动禁用
Failed to execute operation: File exists
2021-08-15 15:16:15 [EROR] [sshcmd.go:82] exec command failed Process exited with status 1
2021-08-15
 15:16:15 [EROR] [filesystem.go:187] exec init.sh failed exec command
failed 10.8.52.16 cd /var/lib/sealer/data/my-cluster/rootfs &&
chmod +x scripts/* && cd scripts && sh init.sh
2021-08-15 15:16:15 [EROR] [run.go:55] mount rootfs failed mountRootfs failed
```

对应的处理策略如下：

```
先执行systemctl enable kubelet 看看是否出现上述错误，
再执行 find / -name "kubelet*"  找多多余的那个删掉。
```



### 出现`Unit docker.service is masked.`错误

执行 `systemctl restart docker` 时出现`Unit docker.service is masked.` ，例如：

```
# systemctl restart docker
Failed to restart docker.service: Unit docker.service is masked.
```



处理策略

```
# systemctl unmask docker.service
# systemctl unmask docker.socket
# systemctl start docker.service
```



### 出现`The recommended driver is "systemd.`错误

当出现类似 `The recommended driver is "systemd"` 错误时，例如：

```
    [WARNING Service-Docker]: docker service is not enabled, please run 'systemctl enable docker.service'
    [WARNING IsDockerSystemdCheck]: detected "cgroupfs" as the Docker cgroup driver. The recommended driver is "systemd". Please follow the guide at https://kubernetes.io/docs/setup/cri/
error execution phase preflight: [preflight] Some fatal errors occurred:
    [ERROR Port-10250]: Port 10250 is in use

```

处理策略

```
在/etc/docker/daemon.json 里加
"exec-opts": ["native.cgroupdriver=systemd"]
```

然后再重启docker:

```
systemctl restart docker
```



### 出现`Port 10250 is in use`错误

当出现 类似`Port 10250 is in use` 这样的错误时，例如：

```
Port 10250 is in use
error execution phase preflight: [preflight] Some fatal errors occurred:
    [ERROR Port-10250]: Port 10250 is in use

```

处理策略：

```
执行 kubeadm reset 
```



### 出现 `overlayfs: maximum fs stacking depth exceeded`错误

dmesg 里看到类似 `overlayfs: maximum fs stacking depth exceeded`  这样的错误，例如

```
[ 2204.645961] overlayfs: maximum fs stacking depth exceeded
```

处理策略

```
# df
# umount overlay
```



### 出现 `write /proc/self/attr/keycreate: invalid argument` 错误

当出现类似 `write /proc/self/attr/keycreate: invalid argument`  这样的错误时，例如：

```
 docker: Error response from daemon: OCI runtime create failed: container_linux.go:349: starting container process caused "process_linux.go:449: container init caused \"write /proc/self/attr/keycreate: invalid argument\"": unknown.

```

处理策略：

这是 遇到 一个 runc 与 selinux 的bug，升级 runc 到RC93后的即可，例如：

```
# runc -v
1.0.0-rc10
commit: dc9208a3303feef5b3839f4323d9beb36df0a9dd
spec: 1.0.0-dev

# wget https://github.com/opencontainers/runc/releases/download/v1.0.1/runc.amd64
# mv /usr/bin/runc /usr/bin/runc.bak
# chmod runc.amd64 && cp runc.amd64 /usr/bin/
```









