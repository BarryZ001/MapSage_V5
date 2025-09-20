
## 版本申明

| 版本 | 修改内容 | 修改时间  |
| ---- | -------- | --------- |
| v1.0 | 初始化   | 10/08/2022 |
| v1.1 | 格式调整 | 10/21/2022 |
|      |          |           |
|      |          |           |


## 简介

gcu-upgrade-manager是一款上k8s部署的应用，用于配合gcu-operator 2.0组件，共同管理enflame gcu相关软件，如：
- Node Feature Discovery
- GCU Driver
- GCU Container Toolkit
- GCU Device Plugin
- GCU Exporter
- GCU Feature Discovery

其中，gcu-operator 2.0主要负责这些组件的一键部署和一键卸载工作，而gcu-upgrade-manager主要负责升级工作。

使用gcu-upgrade-manager前，你需要先了解gcu-operator 2.0组件的基本原理和使用方法。同时也要对CRD（Custom Resource Definition，自定义资源定义）有一定了解，因为要通过CR（Custom Resource，自定义资源）对上述组件进行部署、卸载、升级等操作。



## 接口说明

gcu-upgrade-manager v1.0.0以http接口调用的方式进行服务，当前提供了四种类型接口：List，Get，Put和Delete。



### List接口

URL：GET /gum/v1/gr/

说明：该List接口使用方法和Get一样，但这里会返回集群中全部的gr（gcu-resource， 属于gcu-operator 2.0自定义资源）列表

示例：curl http://<node-ip>:32765/gum/v1/gr

备注：示例中node-ip为当前节点ip，32765为gcu-upgrade-manager组件默认监听端口，下同



### Get接口

URL：GET /gum/v1/gr/<name>

说明：返回指定名称gr的详情

参数：
- name：要查询的gr名称

示例：curl http://<node-ip>:32765/gum/v1/gr/enflame-gcu-resource



URL：GET /gum/v1/gr/<name>/resource/<kind>

说明：返回指定名称gr部署的某个组件的应用详情

参数：
- name：要查询的gr名称
- kind：要查询的应用，枚举值：driver，toolkit，plugin，exporter，gfd

示例：curl http://<node-ip>:32765/gum/v1/gr/enflame-gcu-resource/resource/driver



### Put接口

URL：PUT /gum/v1/gr/<name>

说明：对指定名称的gr进行升级

参数：
- name：要升级的gr名称

请求体：
- file：必选字段，指定要升级到的gr蓝图名称，可以是json或yaml格式，但必须存放在当前节点的/topsdata/dcp/gcu-upgrade-manager/blueprints/路径下

示例：curl http://<node-ip>:32765/gum/v1/gr/enflame-gcu-resource -X PUT -d '{"file": "gr.json"}'



### Delete接口

URL：DELETE /gum/v1/gr/<name>

说明：删除指定名称的gr

参数：
- name：要删除的gr名称

请求体：
- policy：可选字段，指定删除策略。枚举值：
- background：默认值。收到请求后便返回请求结果，后台删除集群中的gr资源
- foreground：等集群中的gr资源删除完成后，返回请求结果

示例：curl http://<node-ip>:32765/gum/v1/gr/enflame-gcu-resource -X DELETE -d '{"policy": "foreground"}'



## 安装gcu-upgrade-manager组件

安装说明：
- 环境需要安装好k8s，docker，helm3等组件
- gcu-upgrade-manager组件依赖于gcu-operator 2.0组件，因此你需要先安装gcu-operator 2.0，否则gcu-upgrade-manager的pod将不会就绪。


### gcu-upgrade-manager发布包

在topscloud目录下可以找到gcu-upgrade-manager发布包：

```
topscloud_{VERSION}/gcu-upgrade-manager_{VERSION} # ll
total 39860
drwxrwxr-x 3 zxx zxx     4096 Oct 24 01:51 ./
drwxrwxr-x 3 zxx zxx     4096 Oct 24 01:51 ../
-rwxrwxr-x 1 zxx zxx     2083 Oct 24 01:51 build-image.sh*
-rwxrwxr-x 1 zxx zxx     1144 Oct 24 01:51 delete.sh*
-rwxrwxr-x 1 zxx zxx     3909 Oct 24 01:51 deploy.sh*
-rw-rw-r-- 1 zxx zxx      180 Oct 24 01:51 Dockerfile
-rwxrwxr-x 1 zxx zxx 40784660 Oct 24 01:51 gcu-upgrade-manager*
drwxrwxr-x 4 zxx zxx     4096 Oct 24 01:51 gcu-upgrade-manager-chart/
```


### 构建gcu-upgrade-manager镜像

直接执行发布包中的build-image.sh文件一键构建镜像，镜像构建后将自动载入到当前节点，并将镜像包保存到./images目录下。

```
topscloud_{VERSION}/gcu-upgrade-manager_{VERSION} # ./build-image.sh
1. Clear old image if exist
[sudo] password for zxx:
Untagged: artifact.enflame.cn/enflame_docker_images/enflame/gcu-upgrade-manager:latest
Deleted: sha256:d1e180b730681a6c301c8a5409d699b1c482dd8d089cc2767877bc4290ad6d0d
Deleted: sha256:6d8f7cdcb9b5c7f88bf15860eacbe92e1c29e28b83deb85aa2fb126d9c177d42
Deleted: sha256:82bf12aa17dc23e8d6fbcb04eb9daf40ab4eae9e4990fe0e3911cef26897d446
2. Build image start...
image name:artifact.enflame.cn/enflame_docker_images/enflame/gcu-upgrade-manager, image version:latest
Sending build context to Docker daemon  40.81MB
Step 1/3 : FROM debian:bullseye-slim
 ---> 2bb87fe5a1f3
Step 2/3 : COPY gcu-upgrade-manager /usr/bin/gcu-upgrade-manager
 ---> 847a0c3baf5b
Step 3/3 : CMD ["gcu-upgrade-manager"]
 ---> Running in 09cc4787469a
Removing intermediate container 09cc4787469a
 ---> 3099352e1634
Successfully built 3099352e1634
Successfully tagged artifact.enflame.cn/enflame_docker_images/enflame/gcu-upgrade-manager:latest
build image success
3. save image to ./images
build success, you can deploy gcu upgrade manager use ./deploy.sh
```


### 安装gcu-upgrade-manager

使用deploy.sh一键安装gcu-upgrade-manager。

```
topscloud_{VERSION}/gcu-upgrade-manager_{VERSION} # ./deploy.sh
1. Try to push component image to enflame repo...
Untagged: artifact.enflame.cn/enflame_docker_images/enflame/gcu-upgrade-manager:latest
Deleted: sha256:c3092dcf1413766660943fece0823b7fe26ea19f19bde2ca24f5bc0d0012018a
Deleted: sha256:1d9cc144aa69afeaef5a57a905764c13cb60e9e58de14e9d69ef9c59917a34dc
15ae6d36287b: Loading layer [==================================================>]  40.76MB/40.76MB
Loaded image: artifact.enflame.cn/enflame_docker_images/enflame/gcu-upgrade-manager:latest
The push refers to repository [artifact.enflame.cn/enflame_docker_images/enflame/gcu-upgrade-manager]
15ae6d36287b: Preparing
fe7b1e9bf792: Layer already exists
unauthorized: User is unauthorized to upload to enflame_docker_images/enflame/gcu-upgrade-manager/_uploads
Push images to repo failed, will load image to all nodes
2. Load images to cluster nodes start...
cluster node name list:
sse-lab-inspur-048
load image to cluster nodes success
3. Deploy gcu upgrade manager release start...
NAME: gcu-upgrade-manager
LAST DEPLOYED: Mon Oct 24 16:55:56 2022
NAMESPACE: kube-system
STATUS: deployed
REVISION: 1
TEST SUITE: None
```

查看组件运行正常：

```
# kubectl get pod -A
kube-system        gcu-operator-7578bc9dd4-2bc47                1/1     Running   0          3d1h
kube-system        gcu-upgrade-manager-6c6b5c4fbc-lqngz         1/1     Running   0          58s
kube-system        node-feature-discovery-ds-qhkc2              2/2     Running   0          3d1h
```



## 使用示例

先部署一个gr资源:

```
# kubectl create -f ../gcu-operator_2.0.1/example/gcu-resource.json
gcuresource.topsops.enflame.com/enflame-gcu-resource created

# kubectl get gr
NAME                   READY   STATUS    AGE
enflame-gcu-resource   5/5     Running   2m46s

# kubectl get pod -A
kube-system        gcu-operator-7578bc9dd4-2bc47                1/1     Running   0          3d1h
kube-system        gcu-upgrade-manager-6c6b5c4fbc-lqngz         1/1     Running   0          58s
kube-system        node-feature-discovery-ds-qhkc2              2/2     Running   0          3d1h
kube-system        enflame-gcu-docker-plugin-s9vrc              1/1     Running   0          2m37s
kube-system        enflame-gcu-driver-dgcvm                     1/1     Running   0          2m40s
kube-system        enflame-gcu-exporter-lprf8                   1/1     Running   0          54s
kube-system        enflame-gcu-feature-discovery-6g8xm          1/1     Running   0          51s
kube-system        enflame-gcu-k8s-plugin-rbp4f                 1/1     Running   0          63s
```


### 查询gr列表

```
# curl http://<node-ip>:32765/gum/v1/gr
{
    "operation": "List",
    "result": "Success",
    "content": {
        "apiVersion": "topsops.enflame.com/v1",
        "items": [
            {
                "apiVersion": "topsops.enflame.com/v1",
                "kind": "GcuResource",
                "metadata": {
                    "creationTimestamp": "2022-10-19T08:28:29Z",
                    "finalizers": [
                        "topsops.enflame.com/v1"
                    ],
                    "generation": 1,
    ......
}
```

### 查询指定gr详情

```
# curl http://<node-ip>:32765/gum/v1/gr/enflame-gcu-resource
{
    "operation": "Get",
    "result": "Success",
    "content": {
        "apiVersion": "topsops.enflame.com/v1",
        "kind": "GcuResource",
        "metadata": {
            "creationTimestamp": "2022-10-19T08:28:29Z",
            "finalizers": [
                "topsops.enflame.com/v1"
            ],
            "generation": 1,
    ......
}
```

### 查询驱动信息

```
# curl http://<node-ip>:32765/gum/v1/gr/enflame-gcu-resource/resource/driver
{
    "operation": "Get",
    "result": "Success",
    "content": {
        "apiVersion": "apps/v1",
        "kind": "DaemonSet",
        "metadata": {
            "annotations": {
                "deprecated.daemonset.template.generation": "1"
            },
            "creationTimestamp": "2022-10-19T08:28:29Z",
            "generation": 1,
            "labels": {
                "app.kubernetes.io/name": "enflame-gcu-driver"
            },
            "name": "enflame-gcu-driver",
            "namespace": "kube-system",
            "resourceVersion": "3803229",
            "uid": "50771276-1f1c-4d24-ae1e-28b0785d9a48"
        },
        "spec": {
            "revisionHistoryLimit": 10,
            "selector": {
                "matchLabels": {
                    "app.kubernetes.io/name": "enflame-gcu-driver"
                }
            },
            "template": {
                "metadata": {
                    "creationTimestamp": null,
                    "labels": {
                        "app.kubernetes.io/name": "enflame-gcu-driver"
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "env": [
                                {
                                    "name": "WITH_REBOOT",
                                    "value": "false"
                                }
                            ],
                            "image": "artifact.enflame.cn/galaxy_docker_hub/gcu/gcu-driver:latest",
                            "imagePullPolicy": "IfNotPresent",
                            "name": "enflame-gcu-driver",
    ......
}
```

### 升级gr

先准备新的gr文件。由于当前升级直接采用先删后建的方式，你可以修改gr文件中，如：驱动镜像、驱动版本号、gcu-runtime镜像，k8s-plugin镜像等任意子资源字段。但不能修改gr资源本身的名称。

并将修改后的gr文件放在/topsdata/dcp/gcu-upgrade-manager/blueprints目录下。

由于篇幅问题，下面不列出新旧gr的蓝图，通过查询得知当前gr部署了驱动、runtime等全部资源：

```
# kubectl get gr
NAME                   READY   STATUS    AGE
enflame-gcu-resource   5/5     Running   6m11s
 
 
# kubectl get gr -o json|jq .items[].status
{
	"clusterGCUNodes": [
		"sse-lab-inspur-048"
	],
	"conditions": [
		{
			"action": "deploy",
			"lastUpdateTime": "2022-10-19T08:30:24Z",
			"message": "success",
			"progress": 100,
			"reason": "success",
			"status": "success"
		}
	],
	"observedGeneration": 1,
	"phase": "Running",
	"readyResource": "5/5",
	"resourceStatuses": [
		{
			"name": "driver",
			"namespace": "kube-system",
			"status": "Running"
		},
		{
			"name": "toolkit",
			"namespace": "kube-system",
			"status": "Running"
		},
		{
			"name": "devicePlugin",
			"namespace": "kube-system",
			"status": "Running"
		},
		{
			"name": "exporter",
			"namespace": "kube-system",
			"status": "Running"
		},
		{
			"name": "gfd",
			"namespace": "kube-system",
			"status": "Running"
		}
	]
}
```

修改后的gr这里只部署gcu-driver和gcu-runtime两个资源。

```
# curl http://<node-ip>:32765/gum/v1/gr/enflame-gcu-resource -X PUT -d '{"file": "gr.json"}'
{
    "operation": "Upgrade",
    "result": "Success"
}
```

等待gr升级完成。你可以通过查看日志了解gr的升级进度：

```
/topsdata/dcp/gcu-upgrade-manager/op-log # vim gcu-upgrade-manager.log
2022/10/20 03:33:48 INFO /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:239 Accept reqeust: &{PUT /gum/v1/gr/enflame-gcu-resource HTTP/1.1 1 1 map[Accept:[*/*] Content-Length:[19] Content-Type:[application/x-www-form-urlencoded] User-Agent:[curl/7.47.0]] 0xc00074a700 <nil> 19 [] false <node-ip>:32765 map[] map[] <nil> map[] <node-ip>:45464 /gum/v1/gr/enflame-gcu-resource <nil> <nil> <nil> 0xc00074a740}
2022/10/20 03:33:48 INFO /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:277 upgrade request body: {"file": "gr.json"}
......
2022/10/20 03:33:48 INFO /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:435 send delete request for gr: enflame-gcu-resource success
2022/10/20 03:33:48 INFO /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:258 send upgrade gr enflame-gcu-resource request success
2022/10/20 03:33:48 INFO /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:448 check cluster old gr: enflame-gcu-resource delete finish retry times: 0
2022/10/20 03:33:48 WARNING /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:460 check cluster old gr: enflame-gcu-resource is deleting
2022/10/20 03:33:51 INFO /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:448 check cluster old gr: enflame-gcu-resource delete finish retry times: 1
2022/10/20 03:33:51 WARNING /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:460 check cluster old gr: enflame-gcu-resource is deleting
2022/10/20 03:33:54 INFO /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:448 check cluster old gr: enflame-gcu-resource delete finish retry times: 2
2022/10/20 03:33:54 WARNING /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:460 check cluster old gr: enflame-gcu-resource is deleting
......
2022/10/20 03:35:19 INFO /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:452 check cluster old gr: enflame-gcu-resource delete finish, wait new gr recreating...
2022/10/20 03:35:19 INFO /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:511 check cluster new gr: enflame-gcu-resource run status retry times: 0
2022/10/20 03:35:19 WARNING /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:537 check new cluster gr: enflame-gcu-resource run status is
2022/10/20 03:35:23 INFO /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:208 Accept reqeust: &{GET /gum/v1/gr/enflame-gcu-resource HTTP/1.1 1 1 map[Accept:[*/*] User-Agent:[curl/7.47.0]] {} <nil> 0 [] false <node-ip>:32765 map[] map[] <nil> map[] <node-ip>:46414 /gum/v1/gr/enflame-gcu-resource <nil> <nil> <nil> 0xc0003c0900}
2022/10/20 03:35:24 INFO /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:511 check cluster new gr: enflame-gcu-resource run status retry times: 1
2022/10/20 03:35:24 WARNING /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:537 check new cluster gr: enflame-gcu-resource run status is deploying
2022/10/20 03:35:29 INFO /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:511 check cluster new gr: enflame-gcu-resource run status retry times: 2
2022/10/20 03:35:29 WARNING /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:537 check new cluster gr: enflame-gcu-resource run status is deploying
......
2022/10/20 03:36:54 WARNING /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:537 check new cluster gr: enflame-gcu-resource run status is deploying
2022/10/20 03:36:59 INFO /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:511 check cluster new gr: enflame-gcu-resource run status retry times: 20
2022/10/20 03:36:59 INFO /home/zxx/go/src/gcu-upgrade-manager/routers/routers.go:534 check new cluster gr: enflame-gcu-resource run status is Running
```

升级完成后，可以查看新的gr资源运行状态：

```
# kubectl get gr
NAME                   READY   STATUS    AGE
enflame-gcu-resource   2/5     Running   6m11s
 
 
# kubectl get gr -o json|jq .items[].status
{
	"clusterGCUNodes": [
		"sse-lab-inspur-048"
	],
	"conditions": [
		{
			"action": "deploy",
			"lastUpdateTime": "2022-10-20T03:36:59Z",
			"message": "success",
			"progress": 100,
			"reason": "success",
			"status": "success"
		}
	],
	"observedGeneration": 1,
	"phase": "Running",
	"readyResource": "2/5",
	"resourceStatuses": [
		{
			"name": "driver",
			"namespace": "kube-system",
			"status": "Running"
		},
		{
			"name": "toolkit",
			"namespace": "kube-system",
			"status": "Running"
		}
	]
}
```

### 删除gr

使用后台删除的方式可以不用携带请求体，请求被接收后会立即返回结果，并在后台执行gr的删除逻辑。

```
# curl http://<node-ip>:32765/gum/v1/gr/enflame-gcu-resource -X DELETE
{
    "operation": "Delete",
    "result": "Success"
}
 
 
# 删除完成
# curl http://<node-ip>:32765/gum/v1/gr/enflame-gcu-resource
{
    "operation": "Get",
    "result": "Failed",
    "error": "gcuresources.topsops.enflame.com \"enflame-gcu-resource\" not found"
}
```

当然，如果你想让程序等待gr删除完成后再返回执行结果，你可以携带policy参数：

```
# curl http://<node-ip>:32765/gum/v1/gr/enflame-gcu-resource -X DELETE -d '{"policy": "foreground"}'
```



## Q&A
- 可以部署多个gr吗？
实现上是支持的，但不建议这么做。因为gr属于gcu-operator 2.0组件的自定义资源，这个operator组件设计上是针对上述各个组件的一键部署进行定制的。一个gr可以将这些组件部署完成，所以没必要部署多个gr，避免不必要的冲突。

- 支持灰度升级吗？
gcu-upgrade-manager v1.0.0版本暂不支持滚动升级，灰度发布等功能，当前仅支持先删后建的升级方式。

- 可以单独升级驱动或其它的单个组件吗？
暂时不可以，对于驱动而言，后续组件是依赖于它的，所以升级驱动也必然涉及赖它的组件的升级。
