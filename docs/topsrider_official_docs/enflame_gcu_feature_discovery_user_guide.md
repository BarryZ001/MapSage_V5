
## 版本申明

| 版本 | 修改内容               | 修改时间   |
| ---- | --------------------- | ---------- |
| v1.0 | 初始化                 | 11/30/2022  |
| v1.1 | 格式调整               | 12/01/2022  |


## 简介

GCU Feature Discovery是一款部署在k8s集群上的组件，主要用于给GCU节点打上一些与GCU设备属性相关的标签。比如：该节点的GCU驱动是哪个版本，GCU显存是多大等。这些标签多是以"enflame.com"开头的标签，打上这些标签的主要目的是在之后的任务调度中，可以根据标签很方便的将任务调度到指定节点上。


## 部署示例

### 部署要求

- 安装docker
- k8s集群版本高于1.8
- 集群中安装了GCU驱动
- 集群中安装了Enflame Container Toolkit
- 集群中安装了Enflame K8s Device Plugin
- 集群中安装了Node Feature Discovery


### GCU Feature Discovery提供的标签

GFD提供的标签以enflame.com开头，主要标签如下：

| Label Name                              | Value Type | Meaning                                             | Example              |
| --------------------------------------- | ---------- | --------------------------------------------------- | -------------------- |
| enflame.com/gfd.timestamp               | string     | Timestamp of the deploy gfd              (optional) | 2023-03-02-02-59-47  |
| enflame.com/gcu.count                   | Integer    | Number of GCUs                                      | 8                    |
| enflame.com/gcu.machine                 | String     | Machine type                                        | NF5468M5             |
| enflame.com/gcu.memory                  | Integer    | Memory of the GCU in Mb                             | 16384                |
| enflame.com/gcu.model                   | String     | Model of the GCU                                    | T10                  |
| enflame.com/gcu.family                  | String     | Family of the GCU                                   | LEO                  |
| enflame.com/gfd.latestLabeledTimestamp  | string     | Timestamp of the latest generated labels            | 2023-03-02-02-59-47  |


如果节点存在VGCU设备，那么GFD提供的主要标签如下：

| Label Name                              | Value Type | Meaning                                             | Example              |
| --------------------------------------- | ---------- | --------------------------------------------------- | -------------------- |
| enflame.com/gfd.timestamp               | string     | Timestamp of the deploy gfd              (optional) | 2023-03-02-02-59-47  |
| enflame.com/vgcu.present                | string     | Whether there is VGCU device             (optional) | true                 |
| enflame.com/vgcu.count                  | Integer    | Number of VGCUs                                     | 4                    |
| enflame.com/vgcu.machine                | String     | Machine type                                        | NF5468M5             |
| enflame.com/vgcu.memory                 | Integer    | Memory of the VGCU in Mb                            | 4096                 |
| enflame.com/vgcu.model                  | String     | Model of the VGCU                                   | T10                  |
| enflame.com/vgcu.family                 | String     | Family of the VGCU                                  | LEO                  |
| enflame.com/gfd.latestLabeledTimestamp  | string     | Timestamp of the latest generated labels            | 2023-03-02-05-57-20  |



### 部署NFD组件

- GFD组件依赖于NFD组件去执行为节点打标签的动作，所以GFD组件是依赖于NFD的。因此安装GFD之前请确保NFD组件已经安装完成。关于NFD的安装，可以参考gcu-operator的用户手册进行操作。
- GFD组件也依赖于libefml，因此安装GFD之前，请检查主机上的libefml存在且可用，检查方法：
```
gcu-feature-discovery_<VERSION> # ll /usr/lib/libefml.so
lrwxrwxrwx 1 root root 50 Mar  2 02:22 /usr/lib/libefml.so -> /usr/local/efsmi/efsmi-1.14.0/lib/libefml.so.1.0.0*

```


#### 制作GFD组件镜像

在topscloud的release包中，打开GFD的目录：

```
gcu-feature-discovery_<VERSION> # ll
total 1832
drwxrwxr-x  3 root root    4096 11月 14 16:42 ./
drwxrwxr-x 18 root root    4096 11月 14 16:42 ../
-rwxr-xr-x  1 root root    2031 11月 14 16:42 build-image.sh*
-rwxr-xr-x  1 root root    1063 11月 14 16:42 delete.sh*
-rwxr-xr-x  1 root root    3853 11月 14 16:42 deploy.sh*
-rw-r--r--  1 root root     166 11月 14 16:42 Dockerfile
-rwxr-xr-x  1 root root 1821576 11月 14 16:42 gcu-feature-discovery*
-rw-r--r--  1 root root    1284 11月 14 16:42 gcu-feature-discovery-daemonset.yaml
-rw-r--r--  1 root root    1200 11月 14 16:42 gcu-feature-discovery-job.yaml.template
-rw-r--r--  1 root root    2742 11月 14 16:42 nfd.yaml
-rw-r--r--  1 root root   10617 11月 14 16:42 README.md

```

执行build-image.sh脚本一键构建GFD组件镜像：

```
gcu-feature-discovery_<VERSION> # ./build-image.sh
1. Clear old image if exist
Untagged: artifact.enflame.cn/enflame_docker_images/enflame/gcu-gfd:latest
Deleted: sha256:f1b77ca94b34c64815b995648faf6ea7d491a3c8bae41b221f54e1cad58f951e
Deleted: sha256:6a99ab9a8ba984086be90865b8b58219280ac95efe00c50ab1ca971f5e00275a
2. Build image start...
image name:artifact.enflame.cn/enflame_docker_images/enflame/gcu-gfd, image version:latest
Sending build context to Docker daemon   69.2MB
Step 1/3 : FROM ubuntu:18.04
 ---> 5d2df19066ac
Step 2/3 : COPY gcu-feature-discovery /usr/bin/
 ---> 15c539215306
Step 3/3 : ENTRYPOINT ["/usr/bin/gcu-feature-discovery"]
 ---> Running in bcb06da24541
Removing intermediate container bcb06da24541
 ---> 2c8c4f3aef69
Successfully built 2c8c4f3aef69
Successfully tagged artifact.enflame.cn/enflame_docker_images/enflame/gcu-gfd:latest
build image success
3. save image to ./images
/home/zxx/workspace/gcu-feature-discovery_1.0.20230301
build success, you can deploy gcushare device plugin use deploy.sh

```

查看制作好的镜像：
```
gcu-feature-discovery_<VERSION> # docker images|grep gcu
artifact.enflame.cn/enflame_docker_images/enflame/gcu-gfd                       latest              8c4a64fd44d4        About a minute ago   93.1MB

```


#### 部署GFD组件

使用deploy.sh一键部署GFD组件：

```
gcu-feature-discovery_<VERSION> # ./deploy.sh
1. Try to push component image to enflame repo...
Untagged: artifact.enflame.cn/enflame_docker_images/enflame/gcu-gfd:latest
Deleted: sha256:8c4a64fd44d4eb4c2bde7e56485eb31be98c19c74aa60eb4380085e1722f388b
Deleted: sha256:2d0788bca032514fd1e65d842db451743af252ab40c33323ebb59583ece5df58
Deleted: sha256:ea58b9a47e7d6bfed92cefd9d1ac019a9ee160957b6106401e059dfd6962724f
Deleted: sha256:12b5b256683589bcf4fec31eab3b31efa9edbe6a69af82b49206f4f66ecfa405
Deleted: sha256:017f2a6edd63912db97f1c6d32d45be771a2dd3e2cf4aa709905fffbff7c4edf
511051358a99: Loading layer [==================================================>]  1.824MB/1.824MB
2b8dbba73517: Loading layer [==================================================>]  28.18MB/28.18MB
Loaded image: artifact.enflame.cn/enflame_docker_images/enflame/gcu-gfd:latest
The push refers to repository [artifact.enflame.cn/enflame_docker_images/enflame/gcu-gfd]
2b8dbba73517: Preparing
511051358a99: Preparing
69f57fbceb1b: Preparing
unauthorized: User is unauthorized to upload to enflame_docker_images/enflame/gcu-gfd/_uploads
Push images to repo failed, will load operator image to all nodes
2. Load images to cluster nodes start...
cluster node name list:
sse-lab-inspur-048
load image to cluster nodes success
3. Deploy gcu-feature-discovery start...
daemonset.apps/gcu-feature-discovery created

```


#### 检查GFD组件工作正常

查看pod运行正常：

```
gcu-feature-discovery_<VERSION> # kubectl get pod -A
NAMESPACE          NAME                                         READY   STATUS    RESTARTS   AGE
calico-apiserver   calico-apiserver-977d5f498-wqw9w             1/1     Running   3          7d21h
calico-apiserver   calico-apiserver-977d5f498-xjc9n             1/1     Running   3          7d21h
calico-system      calico-kube-controllers-69dfd59986-r4n66     1/1     Running   3          7d21h
calico-system      calico-node-55psn                            1/1     Running   3          7d21h
calico-system      calico-typha-645746747f-vmzqq                1/1     Running   3          7d21h
kube-system        coredns-86dfcb4f6f-47vws                     1/1     Running   3          7d21h
kube-system        coredns-86dfcb4f6f-nhzkl                     1/1     Running   3          7d21h
kube-system        enflame-gcu-docker-plugin-2b7d4              1/1     Running   0          3m14s
kube-system        enflame-gcu-driver-fpgkj                     1/1     Running   0          3m17s
kube-system        enflame-node-feature-discovery-2bpz2         2/2     Running   0          3m24s
kube-system        etcd-sse-lab-inspur-048                      1/1     Running   3          7d21h
kube-system        gcu-feature-discovery-msn8t                  1/1     Running   0          9s
...
```

查看节点标签更新成功：

```
gcu-feature-discovery_<VERSION> # kubectl describe node
Name:               sse-lab-inspur-048
Roles:              control-plane,master
Labels:             beta.kubernetes.io/arch=amd64
                    beta.kubernetes.io/os=linux
                    enflame.com/gcu.count=8
                    enflame.com/gcu.family=LEO
                    enflame.com/gcu.machine=NF5468M5
                    enflame.com/gcu.memory=16384
                    enflame.com/gcu.model=T10
                    enflame.com/gcu.present=true
                    enflame.com/gfd.timestamp=2023-03-02-06-11-03
                    enflame.com/gfd.latestLabeledTimestamp=2023-03-02-06-11-03
                    ......

```


#### 卸载GFD组件

你可以执行delete.sh一键卸载GFD组件

```
gcu-feature-discovery_<VERSION> # ./delete.sh
Uninstall gcu-feature-discovery in namespace:kube-system start...
daemonset.apps "gcu-feature-discovery" deleted

```
