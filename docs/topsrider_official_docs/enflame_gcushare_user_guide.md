
## 版本申明

| 版本 | 修改内容 | 修改时间  |
| ---- | -------- | --------- |
| v1.0 | 初始化   | 8/15/2022 |
| v1.1 | 格式调整 | 9/1/2022 |
|      |          |           |
|      |          |           |


## 简介

### 背景

Kubernetes基础设施使用GCU设备时，是不支持多个pod共享GCU的，这样可以实现更好的隔离，保证每个应用的GCU使用不受其它应用的影响，非常适合深度学习模型训练。但是对于想要提高集群中GCU利用率的用户来说，这样通常会造成GCU资源的浪费。比如：

- 作为集群管理员，在开发过程中，允许多个用户在同一个GCU上共享同一个模型开发环境，以增加集群的GCU使用率。
- 作为一名应用操作员，希望能够同时在同一个GCU上运行多个推理任务。

针对这些场景，就需要实现GCU设备的共享功能，以满足特定场景下的用户需求。


### GCUShare主要组件
GCUShare通过gcushare-scheduler-extender+gcushare-device-plugin这两个组件实现GCU设备的共享。

1）gcushare-scheduler-extender：共享GCU的扩展调度器，主要负责根据用户的资源请求（enflame.com/gcu-memory字段），计算可调度的节点，并分配GCU设备。

2）gcushare-device-plugin：共享GCU的k8s设备插件，主要负责向kubelet注册enflame.com/gcu-memory资源，并负责实际的GCU设备分配工作。


### GCUShare功能概要
1）GCUShare依赖于EFML（Enflame Management Library）来获取Enflame GCU的显存信息。

2）GCUShare作为扩展调度程序，将会修改k8s集群的默认调度器配置，但这并不会入侵原有的调度程序。并且GCUShare组件卸载后，将会自动恢复默认的配置信息。

3）按显存(GCU Memory)和按卡(GCU count)调度的方式不可以在集群内并存。

4）GCUShare只支持调度级别的GCU设备共享，暂不支持共享GCU显存资源的隔离。需要用户应用在代码中配置该任务可使用的GCU显存大小。

5）GCUShare依赖于enflame gcu driver和enflame gcu container toolkit，部署GCUShare组件前，必须按顺序部署好这两个依赖组件（gcu-operator2.0已支持GCUShare组件和依赖组件的一键部署）

6）GCUShare提供了inspect接口，用户可以访问该接口来查询集群所有共享GCU的使用情况，从而对整体资源的使用有一个初步的判断。

7）GCUShare不仅支持以显存GB粒度（默认模式）共享GCU，也支持以显存0.1GB粒度共享GCU。

8）通常组件日志都存储在容器中，一旦容器重启或组件卸载，很容易造成日志丢失。GCUShare组件的日志采用本地持久化存储方式，用户可通过日志自行定位问题。

9）GCUShare支持由用户指定部分节点共享GCU。


### GCUShare Chart说明

Chart 是 Helm 的应用打包格式。gcushare 的 chart 由一系列文件组成，这些文件描述了 Kubernetes 部署gcushare应用时所需要的资源，比如 Service、Deployment、Role等。chart 将这些文件放置在预定义的目录结构中，便于 Helm 部署。GCUShare chart 的目录结构以及包含的各类文件如下：

```
gcushare-device-plugin/build-and-deploy/gcushare-device-plugin-chart/
├── Chart.yaml
├── templates
│   ├── clusterrolebinding.yaml
│   ├── clusterrole.yaml
│   ├── daemonset.yaml
│   └── serviceaccount.yaml
└── values.yaml

gcushare-scheduler-extender/build-and-deploy/gcushare-scheduler-extender-chart/
├── Chart.yaml
├── templates
│   ├── clusterrolebinding.yaml
│   ├── clusterrole.yaml
│   ├── daemonset-config-manager.yaml
│   ├── deployment.yaml
│   ├── serviceaccount.yaml
│   └── service.yaml
└── values.yaml

```


## 安装GCUShare组件
安装说明：

- gcushare-scheduler-extender组件依赖于gcushare-device-plugin组件，因此需要先安装gcushare-device-plugin组件
- 环境提前安装好k8s，docker，helm3，enflame driver，enflame docker等组件

本手册所使用测试环境安装了单节点k8s集群。我们称该节点为节点1，下文同。


### 制作gcushare组件镜像

#### 制作gcushare-device-plugin镜像

切换到gcushare-device-plugin安装包目录下，直接构建镜像：

```
gcushare-device-plugin_{VERSION} # ./build-image.sh
1. Clear old image if exist
Untagged: artifact.enflame.cn/enflame_docker_images/enflame/gcushare-device-plugin:latest
Deleted: sha256:44e83d55c939c494529dc60d87abeb5ab9de0037636b005c31c890ad567bb277
Deleted: sha256:bf85d0721cb55cddee0235550330aca426a2a9ea134834bb3001dbf6e47552c8
Deleted: sha256:aed1954c4976df2a55a87957f0f0d0977a67d0ef495faedb8af4e0a80fef7a04
2. Build image start...
image name:artifact.enflame.cn/enflame_docker_images/enflame/gcushare-device-plugin, image version:latest
Sending build context to Docker daemon  202.7MB
Step 1/6 : FROM debian:bullseye-slim
 ---> e7bb3280b4c7
Step 2/6 : ENV ENFLAME_VISIBLE_DEVICES=all
 ---> Running in 9f3be53661bd
Removing intermediate container 9f3be53661bd
......
```

镜像制作成功后，将会自动载入当前节点，并把镜像包保存到images目录下：

```
gcushare-device-plugin_{VERSION} # docker images|grep gcus
artifact.enflame.cn/enflame_docker_images/enflame/gcushare-device-plugin        latest              8bda9bbd85ea        3 minutes ago       140MB

gcushare-device-plugin_{VERSION} # ll images/
total 139992
drwxrwxr-x 2 root root      4096 10月 12 16:16 ./
drwxrwxr-x 6 root root      4096 10月 12 15:42 ../
-rw------- 1 root root 143343616 10月 12 16:16 gcushare-device-plugin.tar
```


#### 制作gcushare-scheduler-extender镜像
切换到gcushare-scheduler-extender的安装包目录下，直接构建镜像：

```
gcushare-scheduler-extender_{VERSION} # ./build-image.sh
1. Clear old image if exist
Untagged: artifact.enflame.cn/enflame_docker_images/enflame/gcushare-scheduler-extender:latest
Deleted: sha256:55ec7ea36204dbe7690ef0865bbffc9336da0fb6bdaf1de57b1d7cf35254aa7f
Deleted: sha256:77a11422cd7fab640c05bede9173ed4b059f3a9d672583543362a185ebbfb269
2. Build image start...
image name:artifact.enflame.cn/enflame_docker_images/enflame/gcushare-scheduler-extender, image version:latest
Sending build context to Docker daemon  201.5MB
Step 1/3 : FROM debian:bullseye-slim
 ---> e7bb3280b4c7
Step 2/3 : COPY gcushare-scheduler-extender /usr/bin/gcushare-scheduler-extender
 ---> 2c44812c0e3b
Step 3/3 : CMD ["gcushare-scheduler-extender"]
 ---> Running in 8934300952e7
Removing intermediate container 8934300952e7
 ---> ba7540816cdc
Successfully built ba7540816cdc
Successfully tagged artifact.enflame.cn/enflame_docker_images/enflame/gcushare-scheduler-extender:latest
build image success
3. save image to ./images
/home/zxx/workspace/gcushare-scheduler-extender_{VERSION}
build success, you can deploy gcushare scheduler extender use deploy.sh
```

镜像制作成功后，将会自动载入当前节点，并把镜像包保存到images目录下：

```
gcushare-scheduler-extender_{VERSION} # docker images|grep gcus
artifact.enflame.cn/enflame_docker_images/enflame/gcushare-scheduler-extender   latest              ba7540816cdc        38 seconds ago      130MB
artifact.enflame.cn/enflame_docker_images/enflame/gcushare-device-plugin        latest              8bda9bbd85ea        4 minutes ago       140MB

gcushare-scheduler-extender_{VERSION} # ll images/
total 130400
drwxrwxr-x 2 root root      4096 10月 12 16:21 ./
drwxrwxr-x 5 root root      4096 10月 12 15:41 ../
-rw------- 1 root root 133517824 10月 12 16:21 gcushare-scheduler-extender.tar
```



### 安装gcushare组件

#### 安装gcushare-device-plugin

gcushare-device-plugin依赖于libefml，因此安装gcushare-device-plugin之前，请检查主机上的libefml存在且可用，检查方法：

```
gcushare-device-plugin_<VERSION> # ll /usr/lib/libefml.so
lrwxrwxrwx 1 root root 50 Mar  2 02:22 /usr/lib/libefml.so -> /usr/local/efsmi/efsmi-1.14.0/lib/libefml.so.1.0.0*

```

gcushare-device-plugin组件安装时，支持用户指定要在哪些节点共享GCU设备。声明使用共享GCU资源的pod只会调度到这些节点上。

在gcushare-device-plugin安装包目录下，执行./deploy.sh一键安装gcushare-device-plugin组件。

```
gcushare-device-plugin_{VERSION} # ./deploy.sh
Try to push component image to enflame repo...
Untagged: artifact.enflame.cn/enflame_docker_images/enflame/gcushare-device-plugin:latest
Deleted: sha256:e2dc41a2f2d87239109ec8d54ff4b835ad7d47b013ea93646932a25bdfaae43c
Deleted: sha256:4b7d28efe13b925e0e78ac6c467047f298a99963e2518c6226eed08e45ab86be
Deleted: sha256:292ba4578f1337ef2ac88f5f4accad6d39b4e8ea38eb1fc80fe1dff550280ce3
c9d16f4eb746: Loading layer [==================================================>]  28.17MB/28.17MB
9bf9fe7249b6: Loading layer [==================================================>]  31.08MB/31.08MB
Loaded image: artifact.enflame.cn/enflame_docker_images/enflame/gcushare-device-plugin:latest
The push refers to repository [artifact.enflame.cn/enflame_docker_images/enflame/gcushare-device-plugin]
9bf9fe7249b6: Preparing
c9d16f4eb746: Preparing
6485bed63627: Preparing
unauthorized: User is unauthorized to upload to enflame_docker_images/enflame/gcushare-device-plugin/_uploads
push images to repo failed, will load operator image to all nodes
awk: warning: escape sequence `\u' treated as plain `u'
cluster nodes: "10.12.110.166"
Cluster node name list:               # 这里会打印集群所有节点，方便用户输入节点名称
sse-lab-inspur-048
Please enter gcushare node name and separated by space(if all nodes use shared GCU, you can just press Enter):   #此处会请用户输入要使用共享GCU的节点名称，并以空格分开，直接回车表示全部节点使用共享GCU
 
node/sse-lab-inspur-048 labeled
deploy gcushare device plugin release start...
NAME: gcushare-device-plugin
LAST DEPLOYED: Tue Sep 13 17:36:50 2022
NAMESPACE: kube-system
STATUS: deployed
REVISION: 1
TEST SUITE: None
```

deploy.sh主要做了三件事：

- 将镜像推送到软件仓库artifact.enflame.cn/enflame_docker_images/enflame/，如果推送失败则自动将镜像载入到集群的每个节点上。
- 询问并请用户输入需要共享GCU设备的节点名称，并给这些节点自动打上"enflame.com/gcushare": "true"标签。只有打了该标签的节点才会部署gcushare-device-plugin组件。
- 使用helm部署gcushare-device-plugin的release。


我们可以通过查看gcushare-device-plugin的pod信息，来确认gcushare-device-plugin是够运行正常：
```
# kubectl get pod -A
NAMESPACE          NAME                                          READY   STATUS    RESTARTS   AGE
admin              dccm-operator-56d9ff4975-mkrgq                1/1     Running   0          26h
admin              enflame-gcu-docker-plugin-d2bpl               1/1     Running   0          26h
admin              enflame-gcu-driver-95wnm                      1/1     Running   0          26h
admin              node-feature-discovery-ds-m8m2f               2/2     Running   6          26h
calico-apiserver   calico-apiserver-564fbd67-bqjwg               1/1     Running   1          27h
calico-apiserver   calico-apiserver-564fbd67-d56m6               1/1     Running   1          27h
calico-system      calico-kube-controllers-69dfd59986-89sl2      1/1     Running   1          27h
calico-system      calico-node-cxxks                             1/1     Running   2          27h
calico-system      calico-typha-5c97f9b578-fpx4n                 1/1     Running   2          27h
kube-system        coredns-86dfcb4f6f-9bmmz                      1/1     Running   1          27h
kube-system        coredns-86dfcb4f6f-j8tps                      1/1     Running   1          27h
kube-system        etcd-sse-lab-inspur-048                       1/1     Running   1          27h   
kube-system        gcushare-device-plugin-n6c5w                  1/1     Running   0          26h           # gcushare-device-plugin运行正常
kube-system        kube-apiserver-sse-lab-inspur-048             1/1     Running   2          27h          
kube-system        kube-controller-manager-sse-lab-inspur-048    1/1     Running   2          27h
kube-system        kube-proxy-sccf6                              1/1     Running   1          27h
kube-system        kube-scheduler-sse-lab-inspur-048             1/1     Running   0          26h
```

再检查下节点的"enflame.com/gcu-count"字段和"enflame.com/gcu-memory"字段是否更新：

```
# kubectl get node
NAME                 STATUS   ROLES                  AGE   VERSION
sse-lab-inspur-048   Ready    control-plane,master   27h   v1.20.0
gcushare-device-plugin # kubectl describe node sse-lab-inspur-048
......
Capacity:
  cpu:                     80
  enflame.com/gcu-count:   8             # 当前节点有8张gcu卡
  enflame.com/gcu-memory:  128           # 当前节点的gcu总显存为128GB，每张卡为16GB
  ephemeral-storage:       1345603940Ki
  hugepages-1Gi:           0
  hugepages-2Mi:           0
  memory:                  394869612Ki
  pods:                    110
Allocatable:
  cpu:                     80
  enflame.com/gcu-count:   8
  enflame.com/gcu-memory:  128
  ephemeral-storage:       1240108589051
  hugepages-1Gi:           0
  hugepages-2Mi:           0
  memory:                  394767212Ki
  pods:                    110
```


#### 安装gcushare-scheduler-extender组件

在gcushare-scheduler-extender安装包目录下，执行./deploy.sh一键安装gcushare-scheduler-extender组件

```
gcushare-scheduler-extender_{VERSION} # ./deploy.sh
Try to push component image to enflame repo...
Untagged: artifact.enflame.cn/enflame_docker_images/enflame/gcushare-scheduler-extender:latest
Deleted: sha256:e3f3a64691e8fff1cff92310fe85a54ff37a78349da66df396986f39339d84f2
Deleted: sha256:edda4932d3b6f121e0d10d2d1188c06b737eed9a6a734cccafdda8f83d2a159d
3a2ac27bde54: Loading layer [==================================================>]  49.37MB/49.37MB
Loaded image: artifact.enflame.cn/enflame_docker_images/enflame/gcushare-scheduler-extender:latest
The push refers to repository [artifact.enflame.cn/enflame_docker_images/enflame/gcushare-scheduler-extender]
3a2ac27bde54: Preparing
721384ec99e5: Preparing
unauthorized: User is unauthorized to upload to enflame_docker_images/enflame/gcushare-scheduler-extender/_uploads
push images to repo failed, will load operator image to all nodes
awk: warning: escape sequence `\u' treated as plain `u'
cluster nodes: "10.12.110.166"
deploy gcushare scheduler extender release start...
NAME: gcushare-scheduler-extender
LAST DEPLOYED: Tue Sep 13 17:34:44 2022
NAMESPACE: kube-system
STATUS: deployed
REVISION: 1
TEST SUITE: None
start modify scheduler config...
check current k8s cluster version is v1.20.0
copy scheduler extender config file success, for detail, see /etc/kubernetes/scheduler-extender-config.json
stat /etc/kubernetes/kube-scheduler.back.yaml: no such file or directory
backup kube-scheduler.yaml success, for detail, see /etc/kubernetes/kube-scheduler.back.yaml
modify scheduler pod yaml template success, for detail, see /etc/kubernetes/kube-scheduler.yaml
modify kube-scheduler config finish, for detail, see /etc/kubernetes/manifests/kube-scheduler.yaml
```

deploy.sh主要做了两件事：

- 将镜像推送到软件仓库artifact.enflame.cn/enflame_docker_images/enflame/，如果推送失败则自动将镜像载入到集群的每个节点上。
- 使用helm部署gcushare-scheduler-extender的release。

同样的，我们可以通过查询pod确定gcushare-scheduler-extender运行正常：

```
gcushare-scheduler-extender/build-and-deploy # kubectl get pod -A
NAMESPACE          NAME                                          READY   STATUS    RESTARTS   AGE
admin              dccm-operator-56d9ff4975-mkrgq                1/1     Running   0          26h
admin              enflame-gcu-docker-plugin-d2bpl               1/1     Running   0          26h
admin              enflame-gcu-driver-95wnm                      1/1     Running   0          26h
admin              node-feature-discovery-ds-m8m2f               2/2     Running   6          26h
calico-apiserver   calico-apiserver-564fbd67-bqjwg               1/1     Running   1          27h
calico-apiserver   calico-apiserver-564fbd67-d56m6               1/1     Running   1          27h
calico-system      calico-kube-controllers-69dfd59986-89sl2      1/1     Running   1          27h
calico-system      calico-node-cxxks                             1/1     Running   2          27h
calico-system      calico-typha-5c97f9b578-fpx4n                 1/1     Running   2          27h
kube-system        coredns-86dfcb4f6f-9bmmz                      1/1     Running   1          27h
kube-system        coredns-86dfcb4f6f-j8tps                      1/1     Running   1          27h
kube-system        etcd-sse-lab-inspur-048                       1/1     Running   1          27h
kube-system        gcushare-device-plugin-n6c5w                  1/1     Running   0          26h           # gcushare-device-plugin运行正常
kube-system        gcushare-scheduler-extender-9b57bd745-rxd6r   1/1     Running   0          26h           # gcushare-scheduler-extender运行正常
kube-system        kube-apiserver-sse-lab-inspur-048             1/1     Running   2          27h
kube-system        kube-controller-manager-sse-lab-inspur-048    1/1     Running   2          27h
kube-system        kube-proxy-sccf6                              1/1     Running   1          27h
kube-system        kube-scheduler-sse-lab-inspur-048             1/1     Running   0          26h           # 这是默认的调度器，会随着调度器配置的修改而重启
```

我们也可以通过简单的接口访问来测试下组件是否能正常提供服务：

```
gcushare-scheduler-extender_{VERSION}# kubectl get svc -A|grep gcushare-scheduler-extender
kube-system        gcushare-scheduler-extender       ClusterIP   10.96.1.37    <none>        32766/TCP                106s

gcushare-scheduler-extender_{VERSION}# curl 10.96.1.37:32766/version
v1.0.0

```

gcushare-scheduler-extender组件使用service转发访问，目标端口为32766。访问上述URL，将会返回gcushare-scheduler-extender版本号信息，说明组件正常运行。



## 使用共享GCU

使用共享GCU需要在容器内编排"enflame.com/gcu-memory"字段，示例：

```
{
    "apiVersion": "v1",
    "kind": "Pod",
    "metadata": {
        "name": "gcushare-pod-1",
        "namespace": "kube-system"
    },
    "spec": {
        "containers": [{
            "resources": {
                "limits": {
                    "enflame.com/gcu-memory": 4       # 声明申请共享GCU，使用4个共享GCU
                }
            }
        }]
    }
}
```

注意，"enflame.com/gcu-memory": 4，表示申请4个共享GCU，实际单位取决于部署gcushare-device-plugin时的共享方式。如果按GB共享（默认共享方式），则4表示申请4GB显存；如果以0.1GB共享，则4表示申请4*0.1GB=0.4GB显存。


### 场景示例
gcushare-device-plugin提供了pod的示例json文件，目录：build-and-deploy/example/gcushare-pod.json。以下场景的测试文件均为基于该模板修改。

1）部署一个pod，使用4GB内存，pod可以正常运行

```
blueprints # kubectl create -f gcushare-pod-1.json
pod/gcushare-pod-1 created
blueprints # kubectl get pod -A
NAMESPACE          NAME                                          READY   STATUS    RESTARTS   AGE
......
kube-system        gcushare-device-plugin-n6c5w                  1/1     Running   0          27h
kube-system        gcushare-pod-1                                1/1     Running   0          4s
kube-system        gcushare-scheduler-extender-9b57bd745-rxd6r   1/1     Running   0          27h
blueprints # kubectl exec -it gcushare-pod-1 -n kube-system bash
kubectl exec [POD] [COMMAND] is DEPRECATED and will be removed in a future version. Use kubectl exec [POD] -- [COMMAND] instead.
# ls /dev
core  gcu0  fd  full  mqueue  null  ptmx  pts  random  shm  stderr  stdin  stdout  termination-log  tty  urandom  zero
```

pod运行正常，并且成功挂载了GCU0。

gcushare-scheduler-extender组件提供了inspect接口可以用来查询集群所有节点的GCU使用情况。使用示例：

```
inspect.sh <node-name>
```

其中，node-name为可选参数，为空将输出所有节点的gcu使用信息；否则输出指定节点的gcu使用信息。

```
gcushare-scheduler-extender_{VERSION} # ./inspect.sh
inspect GCU usage of all nodes from scheduler cache
{
  "nodes": [
        {
            "name": "sse-lab-inspur-048",
            "totalGCU": 128,                # 节点总的GCU显存
            "usedGCU": 4,                   # 已经使用的GCU显存
            "availableGCU": 124,            # 当前节点剩余的总GCU显存
            "devices": [                    # 每个GCU设备的使用情况
                {
                    "id": 0,
                    "totalGCU": 16,         # GCU0卡总显存
                    "usedGCU": 4,           # GCU0卡已使用显存
                    "availableGCU": 12,     # 当前设备剩余的GCU显存
                    "pods": [               # 使用GCU0卡的全部pod信息
                        {
                            "name": "gcushare-pod-1",
                            "namespace": "kube-system",
                            "uid": "2713cdab-6ec7-40be-a0e6-bf144efb0098",
                            "createTime": "2022-09-06T07:40:43Z",
                            "usedGCU": 4
                        }
                    ]
                },
                {
                    "id": 1,
                    "totalGCU": 16,
                    "usedGCU": 0,
                    "availableGCU": 16,
                    "pods": []
                },
                ......
                {
                    "id": 7,
                    "totalGCU": 16,
                    "usedGCU": 0,
                    "availableGCU": 16,
                    "pods": []
                }
            ]
        }
    ]
}
```

2）再部署一个pod，申请8GB内存，则该pod将优先使用节点1的GCU0卡。
```
blueprints # kubectl create -f gcushare-pod-1.json
pod/gcushare-pod-1 created
blueprints # kubectl get pod -A
NAMESPACE          NAME                                          READY   STATUS    RESTARTS   AGE
......
kube-system        gcushare-device-plugin-n6c5w                  1/1     Running   0          27h
kube-system        gcushare-pod-1                                1/1     Running   0          4s
kube-system        gcushare-pod-2                                1/1     Running   0          5s
kube-system        gcushare-scheduler-extender-9b57bd745-rxd6r   1/1     Running   0          27h
blueprints # kubectl exec -it gcushare-pod-2 -n kube-system bash
kubectl exec [POD] [COMMAND] is DEPRECATED and will be removed in a future version. Use kubectl exec [POD] -- [COMMAND] instead.
# ls /dev
core  gcu0  fd  full  mqueue  null  ptmx  pts  random  shm  stderr  stdin  stdout  termination-log  tty  urandom  zero
```

可以看到，pod2也成功被调度到了节点1，并绑定了GCU0卡。查看节点的GCU使用信息：

```
gcushare-scheduler-extender_{VERSION} # ./inspect.sh sse-lab-inspur-048
inspect GCU usage of node sse-lab-inspur-048 from scheduler cache
{
  "nodes": [
        {
            "name": "sse-lab-inspur-048",
            "totalGCU": 128,
            "usedGCU": 12,
            "availableGCU": 116,
            "devices": [
                {
                    "id": 0,
                    "totalGCU": 16,
                    "usedGCU": 12,
                    "availableGCU": 4,
                    "pods": [
                        {
                            "name": "gcushare-pod-1",
                            "namespace": "kube-system",
                            "uid": "2713cdab-6ec7-40be-a0e6-bf144efb0098",
                            "createTime": "2022-09-06T07:40:43Z",
                            "usedGCU": 4
                        },
                        {
                            "name": "gcushare-pod-2",
                            "namespace": "kube-system",
                            "uid": "62b070a4-0c77-4371-adfa-b20b17e90657",
                            "createTime": "2022-09-06T08:28:09Z",
                            "usedGCU": 8
                        }
                    ]
                },
                {
                    "id": 1,
                    "totalGCU": 16,
                    "usedGCU": 0,
                    "availableGCU": 16,
                    "pods": []
                },
                {
                    "id": 2,
                    "totalGCU": 16,
                    "usedGCU": 0,
                    "availableGCU": 16,
                    "pods": []
                },
        ......
}
```

3）部署第3个pod，申请12G显存，此时尽管GCU0卡仍剩余4GB显存，该pod也无法使用它，因为GCUShare不支持跨卡分配。该pod将使用另一张GCU卡。

```
blueprints # kubectl create -f gcushare-pod-3.json
pod/gcushare-pod-3 created
blueprints # kubectl get pod -A
NAMESPACE          NAME                                          READY   STATUS    RESTARTS   AGE
......
kube-system        gcushare-device-plugin-n6c5w                  1/1     Running   0          27h
kube-system        gcushare-pod-1                                1/1     Running   0          4s
kube-system        gcushare-pod-2                                1/1     Running   0          5s
kube-system        gcushare-pod-3                                1/1     Running   0          3s
kube-system        gcushare-scheduler-extender-9b57bd745-rxd6r   1/1     Running   0          27h
blueprints # kubectl exec -it gcushare-pod-3 -n kube-system bash
kubectl exec [POD] [COMMAND] is DEPRECATED and will be removed in a future version. Use kubectl exec [POD] -- [COMMAND] instead.
# ls /dev
core  gcu1  fd  full  mqueue  null  ptmx  pts  random  shm  stderr  stdin  stdout  termination-log  tty  urandom  zero
```

可以看到pod3使用了GCU1卡。查看节点GCU使用情况：

```
gcushare-scheduler-extender_{VERSION} # ./inspect.sh sse-lab-inspur-048
inspect GCU usage of node sse-lab-inspur-048 from scheduler cache
{
  "nodes": [
    {
        "name": "sse-lab-inspur-048",
        "totalGCU": 128,
        "usedGCU": 24,
        "availableGCU": 104,
        "devices": [
            {
                "id": 0,
                "totalGCU": 16,
                "usedGCU": 12,
                "availableGCU": 4,
                "pods": [
                    {
                        "name": "gcushare-pod-1",
                        "namespace": "kube-system",
                        "uid": "2713cdab-6ec7-40be-a0e6-bf144efb0098",
                        "createTime": "2022-09-06T07:40:43Z",
                        "usedGCU": 4
                    },
                    {
                        "name": "gcushare-pod-2",
                        "namespace": "kube-system",
                        "uid": "62b070a4-0c77-4371-adfa-b20b17e90657",
                        "createTime": "2022-09-06T08:28:09Z",
                        "usedGCU": 8
                    }
                ]
            },
            {
                "id": 1,
                "totalGCU": 16,
                "usedGCU": 12,
                "availableGCU": 4,
                "pods": [
                    {
                        "name": "gcushare-pod-3",
                        "namespace": "kube-system",
                        "uid": "b2b0701d-9664-4763-a1eb-5892579bcb7a",
                        "createTime": "2022-09-06T08:33:52Z",
                        "usedGCU": 12
                    }
                ]
            },
            {
                "id": 2,
                "totalGCU": 16,
                "usedGCU": 0,
                "availableGCU": 16,
                "pods": []
            },
    ......
}
```

4）部署pod4，使用20G显存，此时pod将无法调度到节点，因为GCUShare限定了单个pod最大可申请的显存数为单张整卡。

```
blueprints # kubectl create -f gcushare-pod-4.json
pod/gcushare-pod-4 created
blueprints # kubectl get pod -A
NAMESPACE          NAME                                          READY   STATUS    RESTARTS   AGE
......
kube-system        gcushare-device-plugin-n6c5w                  1/1     Running   0          27h
kube-system        gcushare-pod-1                                1/1     Running   0          4s
kube-system        gcushare-pod-2                                1/1     Running   0          5s
kube-system        gcushare-pod-3                                1/1     Running   0          3s
kube-system        gcushare-pod-4                                0/1     Pending   0          6s              # pod无法被调度到节点
kube-system        gcushare-scheduler-extender-9b57bd745-rxd6r   1/1     Running   0          27h
```

查看pod事件，显示单个GCU卡显存不足。

```
blueprints # kubectl describe pod gcushare-pod-4 -n kube-system
Name:         gcushare-pod-4
Namespace:    kube-system
......
Events:
  Type     Reason            Age   From               Message
  ----     ------            ----  ----               -------
  Warning  FailedScheduling  22s   default-scheduler  0/1 nodes are available: 1 insufficient GCU Memory in one gcu device.
  Warning  FailedScheduling  22s   default-scheduler  0/1 nodes are available: 1 insufficient GCU Memory in one gcu device.
```


### 以更细粒度共享GCU

GCUShare支持多种GCU共享粒度，分别为1GB（默认模式）、0.1GB、0.01GB以及0.001GB。如果你想要以0.1GB模式共享GCU设备，那么在安装gcushare-device-plugin组件前，只需要修改gcushare-device-plugin-chart中values文件的memoryUnit字段即可：

```
gcushare-device-plugin_{VERSION} # vim gcushare-device-plugin-chart/values.yaml
# Default values for gcushare-device-plugin-chart.
# This is a YAML-formatted file.
# Declare variables to be passed into your templates.

replicaCount: 1

image:
  repository: artifact.enflame.cn/enflame_docker_images/enflame
  name: gcushare-device-plugin
  pullPolicy: IfNotPresent
  # Overrides the image tag whose default is the chart appVersion.
  tag: "latest"
 
imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""
deployName: "gcushare-device-plugin"
memoryUnit: "0.1"              # 该字段表示共享GCU粒度，枚举值：1，0.1，0.01和0.001，默认为1
```

修改后保存values文件，然后再部署gcushare-device-plugin组件即可生效。使用方式仍然为：

```
"enflame.com/gcu-memory": 40             # 表示申请GCU显存大小为40*0.1GB=4GB。
```

注意：
1）gcushare-device-plugin提供了4种粒度的共享方式分别为1GB，0.1GB，0.01GB和0.001GB。但推荐使用1GB或者0.1GB，因为共享粒度越小，gcushare-device-plugin向k8s注册共享设备时发送的GRPC消息越大，这可能导致注册失败。如，当在8卡*32G的机器上使用0.001GB共享GCU时，发送的GRPC消息约为5MB，这超过了GRPC接收的最大消息限制4MB，因此将会注册失败。通过推算，按0.001GB的共享方式最多可以在8卡, 单卡16GB的机器上部署。因此，该粒度不建议使用，当前仅做扩展功能保留。
2）memoryUnit是在部署组件前要确定好的，如果你已经部署成功了gcushare相关组件，那么在运行过程中，请不要再去修改memoryUnit字段，否则可能会造成缓存数据混乱。
3）如果在运行过程中一定要修改memoryUnit字段，由于两种共享粒度是不可以共存的，所以你必须先把正在使用共享GCU的pod全部清除，然后重新安装gcushare的两个组件。



### 查看日志

通常组件的运行日志都存放在容器中，这样就容易带来一些问题，一是一旦容器重启或者组件被卸载，会造成日志的丢失；二是难以根据异常信息找到代码的调用栈。这些问题都将增大故障排查的复杂程度，而用户则完全无从下手进行排查，进而大大增大了开发人员的运维负担。而GCUShare提供了日志的本地存储功能，如果你的组件运行异常，或者使用GCUShare出现了问题，都可以通过查看日志，进行初步定位。

日志存放目录：/var/log/topscloud/gcushare。

1）查看gcushare-scheduler-extender组件日志

```
/var/log/topscloud/gcushare # vim gcushare-scheduler-extender.log
......
2022/09/06 08:35:16 INFO /workspace/pkg/resource/pod.go:46 pod gcushare-pod-2 in namespace kube-system is gcushare pod
2022/09/06 08:35:16 INFO /workspace/controller/gcushare_controller.go:199 No need to update pod name gcushare-pod-2 in ns kube-system and old status is Running, new status is Running; its old annotation {"enflame.com/gcu-device-assign-time":"1662452889753946045","enflame.com/gcu-device-assigned":"true","enflame.com/gcu-device-id":"0","enflame.com/gcu-device-memory":"16","enflame.com/gcu-device-memory-request":"8"} and new annotation {"enflame.com/gcu-device-assign-time":"1662452889753946045","enflame.com/gcu-device-assigned":"true","enflame.com/gcu-device-id":"0","enflame.com/gcu-device-memory":"16","enflame.com/gcu-device-memory-request":"8"}
2022/09/06 08:35:33 INFO /workspace/routers/routers.go:155 listen access url: /gcushare-scheduler/inspect/:nodename, method: GET, request body:&{GET /gcushare-scheduler/inspect/sse-lab-inspur-048 HTTP/1.1 1 1 map[Accept:[*/*] User-Agent:[curl/7.47.0]] {} <nil> 0 [] false 0:32766 map[] map[] <nil> map[] 10.12.110.166:12748 /gcushare-scheduler/inspect/sse-lab-inspur-048 <nil> <nil> <nil> 0xc00039ec80}
2022/09/06 08:35:33 INFO /workspace/pkg/cache/cache.go:124 GetNodeInfo uses the existing cache nodeInfo for sse-lab-inspur-048
2022/09/06 08:35:33 INFO /workspace/pkg/cache/cache.go:126 node sse-lab-inspur-048 with devices {"0":{},"1":{},"2":{},"3":{},"4":{},"5":{},"6":{},"7":{}}
2022/09/06 08:35:33 INFO /workspace/pkg/cache/gcu_device_info.go:54 gcu device 0 is used by pods: {"2713cdab-6ec7-40be-a0e6-bf144efb0098":{"metadata":{"name":"gcushare-pod-1","namespace":"kube-system","uid":"2713cdab-6ec7-40be-a0e6-bf144efb0098","resourceVersion":"287072","creationTimestamp":"2022-09-06T07:40:43Z","annotations":{"enflame.com/gcu-device-assign-time":"1662450043664667707","enflame.com/gcu-device-assigned":"true","enflame.com/gcu-device-id":"0","enflame.com/gcu-device-memory":"16","enflame.com/gcu-device-memory-request":"4"},"managedFields":[{"manager":"gcushare-device-plugin"
......
```

2）查看gcushare-device-plugin组件日志

```
/var/log/topscloud/gcushare # vim gcushare-device-plugin.log
......
2022/09/06 08:33:52 INFO /workspace/pkg/server/server.go:205 ----Allocating GCU for gcu memory is started----
2022/09/06 08:33:52 INFO /workspace/pkg/kube/kube.go:169 init client-go client success
2022/09/06 08:33:52 INFO /workspace/pkg/server/server.go:213 container 0 request fake gcu ids list: ["GCU-U53000090104-_-4","GCU-U53000090104-_-14","GCU-U43000051005-_-14","GCU-U53000090104-_-10","GCU-U53000090503-_-6","GCU-U53000081005-_-0","GCU-U53000081005-_-14","GCU-U53000090104-_-13","GCU-U53000070712-_-6","GCU-U53000090104-_-1","GCU-U53000080510-_-11","GCU-U53000081005-_-2"]
2022/09/06 08:33:52 INFO /workspace/pkg/server/server.go:216 container RequestGCUs memory: 12
2022/09/06 08:33:52 INFO /workspace/pkg/resource/pod.go:111 list pod gcushare-pod-3 in ns kube-system in node sse-lab-inspur-048 and status is Pending
2022/09/06 08:33:52 INFO /workspace/pkg/resource/pod.go:76 Found GCUSharedAssumed assumed pod gcushare-pod-3 in namespace kube-system
2022/09/06 08:33:52 INFO /workspace/pkg/resource/pod.go:201 candidatePods list after order by assigned time: [{"metadata":{"name":"gcushare-pod-3","namespace":"kube-system","uid":"b2b0701d-9664-4763-a1eb-5892579bcb7a","resourceVersion":"295923","creationTimestamp":"2022-09-06T08:33:52Z","annotations":{"enflame.com/gcu-device-assign-time":"1662453232515055293","enflame.com/gcu-device-assigned":"false","enflame.com/gcu-device-id":"1","enflame.com/gcu-device-memory":"16","enflame.com/gcu-device-memory-request":"12"},"managedFields":[{"manager":"gcushare-sche-extender","operation":"Update","apiVersion":"v1","time":"2022-09-06T08:33:52Z","fieldsType":"FieldsV1","fieldsV1":{"f:metadata":{"f:annotations":{".":{},"f:enflame.com/gcu-device-assign-time":{},"f:enflame.com/gcu-device-assigned":{},"f:enflame.com/gcu-device-id":{},"f:enflame.com/gcu-device-memory":{},"f:enflame.com/gcu-device-memory-request":{}}}}},{"manager":"kubectl-create","operation":"Update","apiVersion":"v1","time":"2022-09-06T08:33:52Z","fieldsType":"FieldsV1","fieldsV1":{"f:spec":{"f:containers":{"k:{\"name\":\"pod-gcu-example\"}":{".":{},"f:args":{},"f:command":{},"f:image":{},"f:imagePullPolicy":{},"f:name":{},"f:resources":{".":{},"f:limits":{".":{},"f:enflame.com/gcu-memory":{}},"f:requests":{".":{},"f:enflame.com/gcu-memory":{}}},"f:terminationMessagePath":{},"f:terminationMessagePolicy":{},"f:volumeMounts":{".":{},"k:{\"mountPath\":\"/home\"}":{".":{},"f:mountPath":{
......
```



## 组件卸载

通过release目录下delete.sh一键卸载gcushare组件。

### gcushare-scheduler-extender卸载

```
gcushare-scheduler-extender_{VERSION} # ./delete.sh
start recover scheduler config...
check current k8s cluster version is v1.20.0
recover default scheduler success, for detail, see /etc/kubernetes/manifests/kube-scheduler.yaml
stat /etc/kubernetes/kube-scheduler.back.yaml: no such file or directory
stat /etc/kubernetes/kube-scheduler.yaml: no such file or directory
stat /etc/kubernetes/scheduler-extender-config.json: no such file or directory
stat /etc/kubernetes/scheduler-extender-config-v1.23+.yaml: no such file or directory
delete all scheduler config file success, for detail, see /etc/kubernetes
uninstall gcushare scheduler extender release in namespace:kube-system start...
release "gcushare-scheduler-extender" uninstalled
```

组件卸载后，k8s集群将自动恢复默认的调度器配置。

### gcushare-device-plugin卸载

```
gcushare-device-plugin_{VERSION} # ./delete.sh
uninstall gcushare device plugin release in namespace:kube-system start...
release "gcushare-device-plugin" uninstalled
node/sse-lab-inspur-048 labeled
```

组件卸载后，节点上的"enflame.com/gcushare": "true"标签将自动清除。



## 常见问题

1）按卡调度和共享调度为什么不可以共存？

整卡调度和共享调度采用的是两个完全没有关联的k8s设备插件。二者有自己的调度逻辑，而且无法感知到对方的调度缓存，若同时存在，将造成调度混乱。

2）gcushare如何实现底层的显存分配？

gcushare不关注pod实际使用时的显存分配。gcushare是通过存储在gcushare-scheduler-extender组件的GCU缓存，实现pod在调度级别的GCU设备共享，而不是在底层对GCU显存进行划分。所以用户需要自己保证业务实际使用的GCU显存不超过pod声明的申请数目。

3）组件卸载后，已经使用共享GCU的pod业务会受影响吗？

不会。如问题2，gcushare只负责调度层级的GCU共享，已经调度并分配过GCU设备的pod不再受GCUShare影响。

4）gcushare组件重启后缓存会消失吗？如果消失的话是否会影响后续的调度？

组件重启会导致缓存丢失，但这并不会影响后续调度，因为gcushare组件上电后会先进行缓存同步。

5）gcushare支持单个pod内多个容器申请共享GCU吗？

支持，但单个pod内所有申请共享GCU的容器的申请总和不得超过单张GCU卡的内存大小。
