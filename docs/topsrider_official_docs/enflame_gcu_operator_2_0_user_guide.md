
## 版本申明

| 版本 | 修改内容               | 修改时间   |
| ---- | --------------------- | ---------- |
| v1.0 | 初始化                 | 8/01/2022  |
| v1.1 | 格式调整               | 8/10/2022  |
| v1.2 | 更新说明               | 10/13/2022 |
| v1.3 | 支持Nightingale       | 11/01/2022 |
| v1.4 | NFD纳入CR进行管理      | 11/08/2022 |
| v1.5 | 支持构建软件栈默认镜像  | 11/08/2022 |
| v1.6 | 支持gcushare组件部署   | 12/01/2022 |



## 背景
在Kubernetes中支持GCU设备调度，依赖于以下工作：

- 节点上安装enflame gcu 驱动
- 节点上安装enflame container-toolkit
- 集群部署k8s-device-plugin，用于为调度到该节点的pod分配GCU设备。

除此之外，如果我们需要查看集群GCU资源使用情况，可能还需要安装gcu-exporter结合Prometheus输出GCU的运行指标信息。
要安装和管理这么多的组件，对于运维人员来说压力不小。而gcu-operator便是用于自动化管理上面我们提到的这些组件。


## gcu-operator简介

### gcu-operator是什么

gcu-operator是一款基于operator-framework开发的go operator项目，其提供了自定义资源GcuResource用于自动化管理gcu软件

### gcu-operator提供哪些能力

1） 面向云原生，通过编写一张GcuResource（简写gr）资源yaml或json文件（组件提供了示例CR文件，后续详细分析），便可使用kubectl一键部署gcu相关软件：

- Node Feature Discovery；
- GCU Driver；
- GCU Container Toolkit；
- GCU Device Plugin；
- GCU Exporter；
- Nightingale；
- GCU Feature Discovery；
- GCU Share Device Plugin；
- GCU Share Scheduler Extender；
- Enflame Volcano；

2） 支持gcu软件一键部署和一键卸载能力

3） 自动识别k8s集群中enflame gcu设备，管理GCU节点标签

4） 支持GCU软件的依赖部署，前一个组件部署成功之前，后续组件不会部署

5） 日志的持久化存储，即便operator组件重启或者被卸载，都不会删除组件日志

6） 提供更好的运维信息，用户可自行定位问题



## 部署示例

### 环境要求

- 安装好docker组件

- 安装好k8s组件

- 集群有GCU设备

- 安装好helm3组件


### gcu-operator 发布包

在topscloud目录下可以找到gcu-operator 2.0 发布包：

```shell
topscloud_{VERSION}/gcu-operator_{VERSION} # ll
total 46768
drwxrwxr-x  8 root root     4096 May 25 03:22 ./
drwxr-xr-x 20 1003 1003     4096 May 25 03:14 ../
-rw-r--r--  1 root root     1076 May 25 03:15 Dockerfile
-rw-r--r--  1 root root     2730 May 25 03:14 README.md
-rwxr-xr-x  1 root root     8492 May 25 03:15 build-component-image.sh*
-rwxr-xr-x  1 root root     2032 May 25 03:15 build-operator-image.sh*
drwxrwxr-x 12 root root     4096 May 25 03:15 component-images/
-rw-r--r--  1 root root     1269 May 25 03:15 config.json
-rwxr-xr-x  1 root root     1889 May 25 03:15 delete-operator.sh*
-rwxr-xr-x  1 root root     8329 May 25 03:15 deploy-operator.sh*
drwxrwxr-x 12 root root     4096 May 25 03:15 enflame-resources/
drwxrwxr-x  2 root root     4096 May 25 03:15 example/
drwxrwxr-x  4 root root     4096 May 25 03:15 gcu-operator-chart/
drwxr-xr-x  2 root root     4096 May 25 03:19 images/
-rwxr-xr-x  1 root root 47806729 May 25 03:15 manager*
-rwxr-xr-x  1 root root     2028 May 25 03:22 restart-docker.sh*
drwxrwxr-x  2 root root     4096 May 25 03:15 utils/
```

请注意，你必须在topscloud或者topsrider的发布包中使用gcu-operator。


### 构建operator组件镜像


在release包下构建gcu-operator_2.0组件镜像

```shell
gcu-operator_{VERSION} # ./build-operator-image.sh
1. Clear old image if exist
Untagged: artifact.enflame.cn/enflame_docker_images/enflame/gcu-operator:latest
Deleted: sha256:b6d1de45cb9c6c22af70a285a62e6067c25144633e6ca14e40c335d42f4ae98a
Deleted: sha256:629699562c19b498a2a4605dfd03605c5db8f6ded02a7872d924885e85883935
Deleted: sha256:8c09069a663636ae5de41e1f2d477614bce4572b888d0b5d14e038230af35593
Deleted: sha256:f3f0dad84c737f7811b88a28bc473e97f3c9e340ef7195a00be613c216c94478
Deleted: sha256:f98c9ffe27bdaaba532c31beb1f5fee78268e11c3a1661f6763fcdbf1f559fa7
Deleted: sha256:c6824606f0e906614f22aa037548ca7b13dd7e4927e9e7e6d5f000d1987a892c
Deleted: sha256:d3f9d64a4dbb48fe868cc8adc81e14314a76890027d94fea03f3c3928ac51ec5
Deleted: sha256:6381a612541a8ae3c16059f8bb1a3c37682cd1864c15d1b51a5b9a3ca49c87e8
Deleted: sha256:bebb804387c588440b8e73565b568e133a7437dfb963625f445fa559a183ad50
Deleted: sha256:3762dc7e35105e7c54a5a11c189ca708a9f8fdf438b619510e82b459e1b51ff8
Deleted: sha256:a265b66740baa02d4f6741424f4b620724ecad809eab8dfa1db8ac8dd4eefaca
2. Build image start...
image name:artifact.enflame.cn/enflame_docker_images/enflame/gcu-operator, image version:latest
Sending build context to Docker daemon  47.45MB
Step 1/9 : FROM debian:bullseye-slim
 ---> 6a8065e4ba13
Step 2/9 : WORKDIR /
 ---> Running in 24c008780023
Removing intermediate container 24c008780023
 ---> 313a9c668a1f
......
```

镜像构建成功后会自动载入到当前节点，并将镜像包保存在./images目录下。

```shell
gcu-operator_{VERSION} # docker images|grep gcu-operator
artifact.enflame.cn/enflame_docker_images/enflame/gcu-operator           latest              0544b00554cf        2 minutes ago       128MB

gcu-operator_{VERSION} # ll images/
total 202292
drwxr-xr-x 2 root root      4096 10月 14 09:52 ./
drwxrwxr-x 7 root root      4096 10月 14 09:52 ../
-rw------- 1 root root 131396096 10月 14 09:52 gcu-operator.tar
```


### 构建operator软件栈镜像

gcu-operator2.0提供了软件栈默认镜像的一键构建脚本：build-component-image.sh和配置文件config.json:
```shell
gcu-operator_{VERSION}/component-images# ll
total 60
drwxrwxr-x 12 root root 4096 May  8 06:25 ./
drwxrwxr-x  7 root root 4096 May  8 06:24 ../
-rwxr-xr-x  1 root root 7153 May  8 06:24 build-component-image.sh*
-rw-r--r--  1 root root 1269 May  8 06:25 config.json
drwxrwxr-x  4 root root 4096 May  8 06:25 driver/
drwxrwxr-x  2 root root 4096 May  8 06:23 exporter/
drwxrwxr-x  2 root root 4096 May  8 06:23 gdp/
drwxrwxr-x  2 root root 4096 May  8 06:24 gfd/
drwxrwxr-x  2 root root 4096 May  8 06:24 gse/
drwxrwxr-x  2 root root 4096 May  8 06:24 nfd/
drwxrwxr-x  2 root root 4096 May  8 06:24 nightingale/
drwxrwxr-x  2 root root 4096 May  8 06:24 plugin/
drwxrwxr-x  4 root root 4096 May  8 06:23 toolkit/
drwxrwxr-x  2 root root 4096 May  8 06:23 volcano/
```

config.json配置了将要构建的软件镜像的部分信息，你可以根据自己的需求修改它(如：配置文件中的repo地址是enflame的内部访问地址，如果镜像推送到该仓库，外网将不可访问。你可以将repo修改为自己的镜像仓库地址以方便镜像推送；如果仅是本地使用这些镜像，则无需修改)，然后再构建组件镜像。
需要注意的是，config.json默认不构建driver的镜像，这就意味着安装gcu-operator之前，你的机器应该已经安装了driver。当然如果你的机器没有安装driver，你也使用gcu-operator来安装驱动，但你需要做两件事情来支持driver镜像的构建：
- 将config.json中的driver.build改为true
- 如果你不在topsrider的发布包而是在topscloud的发布包下使用gcu-operator，那么你需要下载topsrider包并解压，然后将driver的安装包放到./component-images/driver/目录下，示例：

```bash
chmod +x TopsRider_t2x_*_deb_internal_amd64.run
./TopsRider_t2x_*_deb_internal_amd64.run -x
cp TopsRider_t2x_*_deb_internal_amd64/driver/* ./component-images/driver/
```

执行build-component-image.sh构建默认镜像：

``` shell
gcu-operator_{VERSION} # ./build-component-image.sh build
INFO: detected working in the topscloud package
Building component images for system nameID: ubuntu18.04, arch: x86_64 start...
Building artifact.enflame.cn/enflame_docker_images/enflame/node-feature-discovery:v0.11.3 for system nameID: ubuntu18.04, arch: x86_64 start...
WARN: image artifact.enflame.cn/enflame_docker_images/enflame/node-feature-discovery:v0.11.3 already exist, do you want to overwrite this old image?(y/yes, n/no)
n           # 要构建的镜像已存在，此处会询问是否要覆盖构建镜像，输入n将跳过该镜像的构建流程，继续构建后续镜像
Exit the build process
Note: Maybe you can refer to the usage to modify the build parameters and rebuild this image
/home/zxx/workspace/topscloud_bcbd8f0/gcu-operator_2.1.20230625
......
```

查看构建的软件栈默认镜像：

```shell
gcu-operator_{VERSION} # docker images | grep enflame
artifact.enflame.cn/enflame_docker_images/enflame/node-feature-discovery        v0.11.3             53686ffa349c        6 minutes ago       150MB
artifact.enflame.cn/enflame_docker_images/enflame/gcu-gfd                       default             e35cf30bae86        4 days ago          93.1MB
artifact.enflame.cn/enflame_docker_images/enflame/gcu-exporter                  default             a2d2d91eb691        4 days ago          152MB
artifact.enflame.cn/enflame_docker_images/enflame/gcu-plugin                    default             b3cea4d25978        4 days ago          143MB
artifact.enflame.cn/enflame_docker_images/enflame/gcu-runtime                   default             25d6e1555c46        4 days ago          175MB
artifact.enflame.cn/enflame_docker_images/enflame/gcu-driver                    default             a3ae29e7d3d9        4 days ago          866MB
```


### 部署gcu-operator 

镜像构建完成后，直接在release下面执行一键部署即可。

请注意，deploy-operator.sh脚本提供了参数：--restart-docker，该参数用于控制部署container-toolkit的节点是否需要重启docker，默认值为true，表示需要重启docker。但重启docker可能会破坏当前环境上所有正在使用docker服务的进程。如果你不想让operator重启docker，请将该参数的值设置为false（如：./deploy-operator.sh --restart-docker false）。在这种情况下，container-toolkit部署成功之后，operator并不会重启宿主机的docker，这也意味着container-toolkit可能将不会生效（除非docker的default runtime已经正确配置为enflame），进而导致后续k8s-device-plugin，gcu-feature-discovery等依赖container-toolkit的组件部署失败。因此，请根据你的需求选择如何设置这个参数。

```shell
gcu-operator_{VERSION} # ./deploy-operator.sh
Try to push operator image to enflame repo...
Untagged: artifact.enflame.cn/enflame_docker_images/enflame/gcu-operator:latest
Deleted: sha256:881227ec821d7dc5e7d871bbe09b1cc9f219ee742691b72f671fb13f83cd45d1
Deleted: sha256:66f6c6b3f7297bc313e417b86d744fd1990987b9167228913ffa6ebd736b5b43
Deleted: sha256:30114d3347234035aab976c90ec9bc086bd9048998226582f26c609f769c590a
Deleted: sha256:7574f7236768df8f529177b899034293dfd4bdf6abba993e4f7989cbc572b2f5
Deleted: sha256:9064f75c93266d08120c07f4ac5a811a53872efa9d00ece15dc578653849430f
Deleted: sha256:1faa6491602bfb1fe5cfb67115709ff08fe71da64f88052ddac90e9acc56b5e1
Deleted: sha256:748d1ed83f778a54baa34d447724db49b1e4a3e94d7ea5b0086b22bb47569094
9dcd143382a2: Loading layer [==================================================>]  47.34MB/47.34MB
9e4fdabf212d: Loading layer [==================================================>]   2.56kB/2.56kB
0f7f8cb200ac: Loading layer [==================================================>]  20.99kB/20.99kB
Loaded image: artifact.enflame.cn/enflame_docker_images/enflame/gcu-operator:latest
invalid reference format
Push images to repo failed, will load operator image to all nodes
Load images to cluster nodes start...
cluster node name list:
sse-lab-inspur-048
load image to cluster nodes success
3. Deploy gcu-operator release start...
NAME: gcu-operator
LAST DEPLOYED: Fri Oct 14 09:53:25 2022
NAMESPACE: kube-system
STATUS: deployed
REVISION: 1
TEST SUITE: None
Are you want to create a example gcu-resource?(y/yes, n/no)?y        # 此处会询问是否部署示例CR，选择确认后，会自动部署CR
gcuresource.topsops.enflame.com/enflame-gcu-resource created
create gcu resource example success
Process execute docker restart created
```

如果你没有选择自动部署CR，你也可以使用kubectl手动部署它：

```shell
kubectl create -f example/gcu-resource.json
```

通过上述deploy-operator.sh文件，可以实现CR的自动部署。除此之外，你也可以先修改gcu-operator的chart中的values文件配置，然后再执行deploy-operator.sh来自动部署operator组件和CR。
编辑gcu-operator-chart中的values文件：

```shell
gcu-operator_{VERSION} # vim gcu-operator-chart/values.yaml
...
autoscaling:
  enabled: false
  minReplicas: 1
  maxReplicas: 100
  targetCPUUtilizationPercentage: 80
  # targetMemoryUtilizationPercentage: 80
nodeSelector: {}
tolerations: []
affinity: {}
gcuResource:
  deploy: false   # 该字段默认值为false，表示helm部署operator的chart包时将不会自动部署CR；若将其改为true，那么helm部署operator的chart包时将自动部署CR
  driver:
    notDeploy: true   # 该字段仅在gcuResource.deploy为true时生效，用于可以控制是否部署driver，默认值为true
```

需要注意的是，如果你采用修改chart包的方式进行CR的自动部署，那么在后续执行deploy-operator.sh一键部署脚本时，将不会再询问是否部署示例CR。另外，使用该方式时，你如果想先修改CR文件，请编辑gcu-operator-chart/templates/gcu-resource.yaml文件后保存。


### 查看资源信息

查看pod信息，全部正常运行

```shell
# 查看pod信息，全部正常运行
gcu-operator_{VERSION} # kubectl get pod -A
NAMESPACE          NAME                                         READY   STATUS    RESTARTS   AGE
kube-system        enflame-gcu-docker-plugin-sdxg5              1/1     Running     0          116s    # gcu软件的pod
kube-system        enflame-gcu-driver-rbksz                     1/1     Running     0          119s
kube-system        enflame-gcu-exporter-vfbf6                   1/1     Running     0          10s
kube-system        enflame-gcu-feature-discovery-kf2fx          1/1     Running     0          7s
kube-system        enflame-gcu-k8s-plugin-pptg9                 1/1     Running     0          19s
enflame            agentd-77cbfb9dd9-cgp59                      1/1     Running     0          11s
enflame            mysql-54c4d4d595-x2xnz                       1/1     Running     0          11s
enflame            nserver-c6bdbcb94-6jbxg                      1/1     Running     1          11s
enflame            nwebapi-5d74d686fb-6qqlk                     1/1     Running     1          11s
enflame            prometheus-57f689d889-w72hx                  1/1     Running     0          11s
enflame            redis-88b69646b-2g79q                        1/1     Running     0          11s
kube-system        gcu-operator-6868944d97-wrvns                1/1     Running     0          3m22s   # operator的pod
kube-system        node-feature-discovery-ds-z5hvh              2/2     Running     0          3m22s   # nfd的pod
```

现在就可以正常使用GCU了。可以通过访问gcu-exporter来测试各组件是否正常工作。gcu-exporter可以通过service的方式访问，默认映射端口是30940，采用node port的方式进行访问。浏览器输入http://<nodeIP>:30940/metrics即可，例如：

```shell
# HELP enflame_gcu_clock gcu clock as reported by the device
# TYPE enflame_gcu_clock gauge
enflame_gcu_clock{Host="enflame-gcu-exporter-qjmrf",ID="0"} 1150
enflame_gcu_clock{Host="enflame-gcu-exporter-qjmrf",ID="1"} 1150
enflame_gcu_clock{Host="enflame-gcu-exporter-qjmrf",ID="2"} 1150
enflame_gcu_clock{Host="enflame-gcu-exporter-qjmrf",ID="3"} 1150
enflame_gcu_clock{Host="enflame-gcu-exporter-qjmrf",ID="4"} 1150
enflame_gcu_clock{Host="enflame-gcu-exporter-qjmrf",ID="5"} 1150
enflame_gcu_clock{Host="enflame-gcu-exporter-qjmrf",ID="6"} 1150
enflame_gcu_clock{Host="enflame-gcu-exporter-qjmrf",ID="7"} 1150
# HELP enflame_gcu_cluster_usage gcu cluster usage as reported by the device
# TYPE enflame_gcu_cluster_usage gauge
enflame_gcu_cluster_usage{Cluster="0",Host="enflame-gcu-exporter-qjmrf",ID="0",Metrics="Cluster_Usage",Name="gcu0"} 0
enflame_gcu_cluster_usage{Cluster="0",Host="enflame-gcu-exporter-qjmrf",ID="1",Metrics="Cluster_Usage",Name="gcu1"} 0
enflame_gcu_cluster_usage{Cluster="0",Host="enflame-gcu-exporter-qjmrf",ID="2",Metrics="Cluster_Usage",Name="gcu2"} 0
enflame_gcu_cluster_usage{Cluster="0",Host="enflame-gcu-exporter-qjmrf",ID="3",Metrics="Cluster_Usage",Name="gcu3"} 0
enflame_gcu_cluster_usage{Cluster="0",Host="enflame-gcu-exporter-qjmrf",ID="4",Metrics="Cluster_Usage",Name="gcu4"} 0
enflame_gcu_cluster_usage{Cluster="0",Host="enflame-gcu-exporter-qjmrf",ID="5",Metrics="Cluster_Usage",Name="gcu5"} 0
enflame_gcu_cluster_usage{Cluster="0",Host="enflame-gcu-exporter-qjmrf",ID="6",Metrics="Cluster_Usage",Name="gcu6"} 0
enflame_gcu_cluster_usage{Cluster="0",Host="enflame-gcu-exporter-qjmrf",ID="7",Metrics="Cluster_Usage",Name="gcu7"} 0
enflame_gcu_cluster_usage{Cluster="1",Host="enflame-gcu-exporter-qjmrf",ID="0",Metrics="Cluster_Usage",Name="gcu0"} 0
......
```

另外还可以通过查看节点的标签信息，确认nfd和gfd组件已生效：

```yaml
#kubectl get node -o yaml
apiVersion: v1
items:
- apiVersion: v1
  kind: Node
  metadata:
    labels:
      beta.kubernetes.io/arch: amd64
      beta.kubernetes.io/os: linux
      enflame.com/gcu.count: "8"
      enflame.com/gcu.family: LEO
      enflame.com/gcu.machine: NF5468M5
      enflame.com/gcu.memory: "16384"
      enflame.com/gcu.model: T10
      enflame.com/gcu.present: "true"
      enflame.com/gfd.timestamp: "1660635854"
      feature.node.kubernetes.io/cpu-cpuid.ADX: "true"
      feature.node.kubernetes.io/cpu-cpuid.AESNI: "true"
      feature.node.kubernetes.io/cpu-cpuid.AVX: "true"
      feature.node.kubernetes.io/cpu-cpuid.AVX2: "true"
      feature.node.kubernetes.io/pci-1e36.present: "true"
      ......
```



## 功能分析

### CR编排

gcu-operator组件提供了默认的CR蓝图（release/gcu-resource.json）：

```json
{
	"apiVersion": "topsops.enflame.com/v1",
	"kind": "GcuResource",
	"metadata": {
		"name": "enflame-gcu-resource"
	},
	"spec": {
		"nfd": {
			"notDeploy": false,
			"name": "enflame-node-feature-discovery",
			"namespace": "kube-system",
			"master": {
				"image": "node-feature-discovery",
				"version": "v0.11.3",
				"imagePullPolicy": "IfNotPresent",
				"repository": "artifact.enflame.cn/enflame_docker_images/enflame"
			},
			"worker": {
				"image": "node-feature-discovery",
				"version": "v0.11.3",
				"imagePullPolicy": "IfNotPresent",
				"repository": "artifact.enflame.cn/enflame_docker_images/enflame"
			}
		},
		"driver": {
			"notDeploy": true,
			"env": [
				{
					"name": "WITH_REBOOT",
					"value": "false"
				}
			],
			"image": "gcu-driver",
			"version": "latest",
			"name": "enflame-gcu-driver",
			"namespace": "kube-system",
			"imagePullPolicy": "IfNotPresent",
			"nodeSelector": {
				"enflame.com/gcu.present": "true"
			},
			"repository": "artifact.enflame.cn/galaxy_docker_hub/gcu"
		},
		"toolkit": {
			"image": "gcu-runtime",
			"version": "latest",
			"name": "enflame-gcu-docker-plugin",
			"namespace": "kube-system",
			"imagePullPolicy": "IfNotPresent",
			"nodeSelector": {
				"enflame.com/gcu.present": "true"
			},
			"repository": "artifact.enflame.cn/galaxy_docker_hub/gcu"
		},
		"devicePlugin": {
			"image": "gcu-plugin",
			"version": "latest",
			"name": "enflame-gcu-k8s-plugin",
			"namespace": "kube-system",
			"imagePullPolicy": "IfNotPresent",
			"nodeSelector": {
				"enflame.com/gcu.present": "true"
			},
			"repository": "artifact.enflame.cn/galaxy_docker_hub/gcu"
		},
		"exporter": {
			"notDeploy": false,
			"image": "gcu-exporter",
			"version": "latest",
			"name": "enflame-gcu-exporter",
			"namespace": "kube-system",
			"imagePullPolicy": "IfNotPresent",
			"nodeSelector": {
				"enflame.com/gcu.present": "true"
			},
			"repository": "artifact.enflame.cn/enflame_docker_images/enflame"
		},
		"gfd": {
			"image": "gcu-gfd",
			"version": "latest",
			"name": "enflame-gcu-feature-discovery",
			"namespace": "kube-system",
			"imagePullPolicy": "IfNotPresent",
			"nodeSelector": {
				"enflame.com/gcu.present": "true"
			},
			"repository": "artifact.enflame.cn/galaxy_docker_hub/gcu"
		},
		"nightingale": {
			"notDeploy": true,
			"namespace": "enflame"
		},
		"gcusharePlugin": {
			"notDeploy": true,
			"name": "gcushare-device-plugin",
			"image": "gcushare-device-plugin",
			"version": "default",
			"namespace": "kube-system",
			"imagePullPolicy": "IfNotPresent",
			"nodeSelector": {
				"enflame.com/gcu.present": "true"
			},
			"memoryUnit": "1",
			"gcushareNodes": []
		},
		"gcushareScheduler": {
			"notDeploy": true,
			"name": "gcushare-scheduler-extender",
			"image": "gcushare-scheduler-extender",
			"version": "default",
			"namespace": "kube-system",
			"imagePullPolicy": "IfNotPresent",
			"nodeSelector": {
				"enflame.com/gcu.present": "true"
			}
		},
		"volcano": {
			"notDeploy": true
		}
	}
}
```

这个CR蓝图默认将会部署node-feature-discovery，container-toolkit，k8s-plugin，gcu-exporter，gcu-feature-discovery等5个组件。当然你也可以编辑它，然后使用kubectl create -f example/gcu-resource.json进行部署。

CR部署成功后可以查看一下状态：

```shell
gcu-operator_{VERSION} # kubectl get gr
NAME                   READY   STATUS    AGE
enflame-gcu-resource   5/5     Running   6m44s
```

说明：
- ready字段表示正在运行的GCU软件数目/当前CR要部署的全部软件数目;
- status字段表示CR的运行状态，running说明全部GCU软件已就绪，否则表示未全部就绪;
- gcu-operator已经支持GCUShare的一键部署，但gcushare和k8s device plugin是不可以共存的。默认的CR蓝图将会部署k8s device plugin，而不会部署gcushare。如果你想部署gcushare，请提前将CR蓝图中k8s device plugin的notDeploy字段值改为true，并将gcushare的notDeploy字段值改为false;


### CR 详情

要查看各GCU软件的运行状态，可以查看CR详情：

```shell
gcu-operator_{VERSION} # kubectl get gr enflame-gcu-resource -o json|jq .status
{
	"clusterGCUNodes": [
		"sse-lab-inspur-048"        # 当前集群所有的gcu节点列表
	],
	"conditions": [
		{
			"action": "deploy",
			"lastUpdateTime": "2022-11-01T06:02:23Z",
			"message": "success",
			"progress": 100,
			"reason": "success",
			"status": "success"
		}
	],
	"observedGeneration": 1,
	"phase": "Running",             # gr的运行状态
	"readyResource": "6/6",         # 已经就绪的软件数目
	"resourceStatuses": [           # 每个组件的状态信息
		{
			"name": "nfd",          # 组件名称
			"namespace": "kube-system",          # 组件所属命名空间
			"ownResources": [                    # 当前组件拥有的全部资源信息
				{
					"group": "rbac.authorization.k8s.io",
					"kind": "ClusterRole",
					"name": "nfd-clusterrole",
					"version": "v1"
				},
				{
					"group": "rbac.authorization.k8s.io",
					"kind": "ClusterRoleBinding",
					"name": "nfd-clusterrolebinding",
					"version": "v1"
				},
				{
					"group": "apps",
					"kind": "DaemonSet",
					"name": "enflame-node-feature-discovery",
					"version": "v1"
				},
				{
					"group": "apiextensions.k8s.io",
					"kind": "CustomResourceDefinition",
					"name": "nodefeaturerules.nfd.k8s-sigs.io",
					"version": "v1"
				},
				{
					"kind": "ServiceAccount",
					"name": "nfd-service-account",
					"version": "v1"
				}
			],
			"status": "Running"                 # 当前组件的运行状态
		},
		{
			"name": "driver",
			"namespace": "kube-system",
			"ownResources": [
				{
					"group": "apps",
					"kind": "DaemonSet",
					"name": "enflame-gcu-driver",
					"version": "v1"
				}
			],
			"status": "Running"
		},
		{
			"name": "toolkit",
			"namespace": "kube-system",
			"ownResources": [
				{
					"group": "apps",
					"kind": "DaemonSet",
					"name": "enflame-gcu-docker-plugin",
					"version": "v1"
				}
			],
			"status": "Running"
		},
		{
			"name": "devicePlugin",
			"namespace": "kube-system",
			"ownResources": [
				{
					"group": "apps",
					"kind": "DaemonSet",
					"name": "enflame-gcu-k8s-plugin",
					"version": "v1"
				}
			],
			"status": "Running"
		},
		{
			"name": "nightingale",
			"namespace": "enflame",
			"ownResources": [
				{
					"group": "apps",
					"kind": "Deployment",
					"name": "agentd",
					"version": "v1"
				},
				{
					"group": "apps",
					"kind": "Deployment",
					"name": "mysql",
					"version": "v1"
				},
				{
					"kind": "PersistentVolume",
					"name": "mysql-pv",
					"version": "v1"
				},
				{
					"kind": "PersistentVolumeClaim",
					"name": "mysql-pvc",
					"version": "v1"
				},
				{
					"kind": "Service",
					"name": "mysql",
					"version": "v1"
				},
				{
					"group": "apps",
					"kind": "Deployment",
					"name": "nserver",
					"version": "v1"
				},
				{
					"kind": "Service",
					"name": "nserver",
					"version": "v1"
				},
				{
					"group": "apps",
					"kind": "Deployment",
					"name": "nwebapi",
					"version": "v1"
				},
				{
					"kind": "Service",
					"name": "nwebapi",
					"version": "v1"
				},
				{
					"group": "apps",
					"kind": "Deployment",
					"name": "prometheus",
					"version": "v1"
				},
				{
					"kind": "Service",
					"name": "prometheus",
					"version": "v1"
				},
				{
					"group": "apps",
					"kind": "Deployment",
					"name": "redis",
					"version": "v1"
				},
				{
					"kind": "Service",
					"name": "redis",
					"version": "v1"
				}
			],
			"status": "Running"
		},
		{
			"name": "gfd",
			"namespace": "kube-system",
			"ownResources": [
				{
					"group": "apps",
					"kind": "DaemonSet",
					"name": "enflame-gcu-feature-discovery",
					"version": "v1"
				}
			],
			"status": "Running"
		}
	]
}
```



### CR 字段说明

CR详细字段解析见下表：

| FieldName                           | Type        | Discription                                         | Remark                                                                         |
| :---------------------------------- | :---------- | :-------------------------------------------------- | :----------------------------------------------------------------------------- |
| apiVersion                          | string      | Group and version information of gcu-resource       | Required, msut be topsops.enflame.com/v1                                       |
| kind                                | string      | The kind of gcu-resource                            | Required, msut be GcuResource                                                  |
| metadata.name                       | string      | The name of gcu-resource                            | Required, user defined, same below                                             |
| spec                                | dict        | Information of GCU software to deploy               | Optional, if empty no components will be deployed                              |
| spec.driver                         | BaseFields  | Custom drive component configuration                | Optional, if empty driver will not be deployed                                 |
| spec.toolkit                        | BaseFields  | Custom toolkit component configuration              | Optional, if empty toolkit will not be deployed                                |
| spec.plugin                         | BaseFields  | Custom plugin component configuration               | Optional, if empty plugin will not be deployed                                 |
| spec.exporter                       | BaseFields  | Custom exporter component configuration             | Optional, if empty exporter will not be deployed                               |
| spec.gfd                            | BaseFields  | Custom gfd component configuration                  | Optional, if empty gfd will not be deployed                                    |
| spec.nightingale                    | dict        | Custom nightingale component configuration          | Optional, if empty nightingale will not be deployed                            |
| spec.nightingale.notDeploy          | bool        | Whether nightingale needs to be deployed            | Optional, default: false, means that nightingale will be deployed              |
| spec.nightingale.namespace          | string      | Custom nightingale namespace                        | Optional, default: enflame                                                     |
| spec.nightingale.agentd-deploy      | BaseFields  | Custom nightingale agentd-deploy configuration      | Optional, see underlying BaseFields for detail                                 |
| spec.nightingale.mysql-deploy       | BaseFields  | Custom nightingale mysql-deploy configuration       | Optional, see underlying BaseFields for detail                                 |
| spec.nightingale.nserver-deploy     | BaseFields  | Custom nightingale nserver-deploy configuration     | Optional, see underlying BaseFields for detail                                 |
| spec.nightingale.nwebapi-deploy     | BaseFields  | Custom nightingale nwebapi-deploy configuration     | Optional, see underlying BaseFields for detail                                 |
| spec.nightingale.prometheus-deploy  | BaseFields  | Custom nightingale prometheus-deploy configuration  | Optional, see underlying BaseFields for detail                                 |
| spec.nightingale.redis-deploy       | BaseFields  | Custom nightingale redis-deploy configuration       | Optional, see underlying BaseFields for detail                                 |
| spec.nightingale.mysql-pv           | dict        | Custom nightingale pv configuration                 | Optional, default: enter operator container, cd /opt/gcu-operator/ for detail  |
| spec.nightingale.mysql-pv.name      | string      | Custom nightingale pv name                          | Required                                                                       |
| spec.nightingale.mysql-pv.hostpath  | string      | Custom nightingale pv hostpath                      | Required                                                                       |
| spec.nightingale.mysql-pvc          | dict        | Custom nightingale pvc configuration                | Optional, default: enter operator container, cd /opt/gcu-operator/ for detail  |
| spec.nightingale.mysql-pvc.name     | string      | Custom nightingale pvc name                         | Required                                                                       |
| spec.nfd                            | dict        | Custom nfd component configuration                  | Optional, if empty nfd will not be deployed                                    |
| spec.nfd.notDeploy                  | bool        | Whether nfd needs to be deployed                    | Optional, default: false, means that nfd will be deployed                      |
| spec.nfd.name                       | string      | Custom nfd controller name                          | Optional, default: node-feature-discovery-ds                                   |
| spec.nfd.namespace                  | string      | Custom nfd namespace                                | Optional, default: kube-system                                                 |
| spec.nfd.nodeSelector               | dict        | Deploy the nfd to the node with the specified labels| Optional, default: null                                                        |
| spec.nfd.master                     | BaseFields  | Custom nfd master configuration                     | Optional, see underlying BaseFields for detail                                 |
| spec.nfd.worker                     | BaseFields  | Custom nfd worker configuration                     | Optional, see underlying BaseFields for detail                                 |
| spec.gcusharePlugin                 | BaseFields  | Custom gcusharePlugin worker configuration          | Optional, see underlying BaseFields for detail                                 |
| spec.gcusharePlugin.memoryUnit      | string      | Custom gcu share level                              | Optional, default: 1. Enumeration: 1, 0.1, 0.01, 0.001                         |
| spec.gcusharePlugin.gcushareNodes   | list        | Custom nodes which want to share gcu                | Optional, default: all nodes                                                   |
| spec.gcushareScheduler              | BaseFields  | Custom gcushareScheduler worker configuration       | Optional, see underlying BaseFields for detail                                 |
| spec.volcano                        | dict        | Custom volcano component configuration              | Optional, if empty volcano will not be deployed                                |
| spec.volcano.notDeploy              | bool        | Whether volcano needs to be deployed                | Optional, default: false, means that volcano will be deployed                  |
| spec.volcano.namespace              | string      | Custom volcano namespace                            | Optional, default: volcano-system                                              |
| spec.volcano.controller             | BaseFields  | Custom volcano controller configuration             | Optional, see underlying BaseFields for detail                                 |
| spec.volcano.admission              | BaseFields  | Custom volcano admission configuration              | Optional, see underlying BaseFields for detail                                 |
| spec.volcano.scheduler              | BaseFields  | Custom volcano scheduler configuration              | Optional, see underlying BaseFields for detail                                 |
| status                              | dict        | Status information for gcu-resource                 | The following fields users should not edit them                                |
| status.clusterGCUNodes              | list        | Cluster GCU Nodes List                              |                                                                                |
| status.conditions                   | list        | gcu-resource status information about each action   |                                                                                |
| status.conditions.action            | string      | Action type                                         | Enumeration: deploy, delete, upgrade, rollback                                 |
| status.conditions.lastUpdateTime    | string      | The last update time of the current action          |                                                                                |
| status.conditions.message           | string      | Some information about the action                   | Such as abnormal information display during execution                          |
| status.conditions.progress          | int         | Current action execution progress                   |                                                                                |
| status.conditions.reason            | string      | Some information about the action                   | Such as abnormal information display during execution                          |
| status.conditions.status            | string      | The execution phase of the current action           | Enumeration: deploying, deleting, upgrading, rollbacking, success, ...         |
| status.observedGeneration           | int         | gcu-resource generation                             |                                                                                |
| status.phase                        | string      | Run status of gcu-resource                          | Enumeration: deploying, deleting, upgrading, rollbacking, running              |
| status.readyResource                | string      | Ready number and total number of component          |                                                                                |
| status.resourceStatuses             | list        | Status information for each component               |                                                                                |
| status.resourceStatuses.name        | string      | Component name                                      |                                                                                |
| status.resourceStatuses.namespace   | string      | Component namespace                                 |                                                                                |
| status.resourceStatuses.status      | string      | Component run status                                | Enumeration: deploying, deleting, upgrading, rollbacking, running              |
| status.resourceStatuses.ownResources| list        | Cluster resources owned by the component            | See underlying ResourceInfo for detail                                         |


扩展字段定义：
BaseFields
| FieldName        | Type    | Discription                                                | Remark                                                                          |
| :--------------- | :------ | :--------------------------------------------------------- | :------------------------------------------------------------------------------ |
| notDeploy        | bool    | Whether the component needs to be deployed                 | Optional, Default: false, means that the component will be deployed             |
| image            | string  | The component image name                                   | Optional, Default: enter operator container, cd /opt/gcu-operator/ for detail   |
| imagePullPolicy  | string  | The component image pull policy                            | Optional, Default: IfNotPresent                                                 |
| name             | string  | The component controller name                              | Optional, Default: enter operator container, cd /opt/gcu-operator/ for detail   |
| namespace        | string  | The component controller namespace                         | Optional, Default: nightingale in enflame, other components in kube-system      |
| repository       | string  | The component image repository                             | Optional, Default: artifact.enflame.cn/enflame_docker_images/enflame            |
| version          | string  | The component image version                                | Optional, Default: latest                                                       |
| env              | list    | Environment variables will be injected into the container  | Optional, Default: null                                                         |
| nodeSelector     | dict    | Deploy the component to the node with the specified labels | Optional, Default: null                                                         |


ResourceInfo
| FieldName        | Type    | Discription                                                | Remark                                                                          |
| :--------------- | :------ | :--------------------------------------------------------- | :------------------------------------------------------------------------------ |
| group            | string  | The group to which the resource belongs                    |                                                                                 |
| version          | string  | The version to which the resource belongs                  |                                                                                 |
| kind             | string  | The resource kind                                          |                                                                                 |
| name             | string  | The resource name                                          |                                                                                 |


说明：对于内部定义了包含扩展字段BaseFields类型的软件，则这些BaseFields类型字段中的与软件第一层定义的同名字段将不再起作用。如spec.nightingale.notDeploy，此时spec.nightingale.<resource>.notDeploy是无效的。


用户可以根据自己的需要指定gcu软件的相关配置，编排CR蓝图。



## 常见问题

### 如何查看日志

gcu-operator的日志存放在宿主机上，通过日志用户可自行快速定位问题,例如：

```shell
/var/log/topscloud/gcu-operator/ # ll
total 380
drwxr-xr-x 2 root root   4096 May 26 08:57 ./
drwxr-xr-x 3 root root   4096 May 24 10:02 ../
-rw-r--r-- 1 root root 370080 May 25 03:23 gcu-operator.log               # operator组件的运行日志
-rw-r--r-- 1 root root   2011 May 25 03:23 restart-docker.log             # 重启docker的进程的运行日志
 
/var/log/topscloud/gcu-operator/ # vim gcu-operator.log
2022/10/10 09:42:09 INFO /home/zxx/go/src/topscloud/cloud/k8s-operator/gcu-operator/main.go:91 setup gcu resource manager success...
2022/10/10 09:42:09 INFO /home/zxx/go/src/topscloud/cloud/k8s-operator/gcu-operator/main.go:103 Starting Controller:gcuresource, Group:topsops.enflame.com, Version:v1, Kind:GcuResource
2022/10/10 09:43:33 INFO /home/zxx/go/src/topscloud/cloud/k8s-operator/gcu-operator/model/gcu-resource.go:191 add finalizers for gcuResource(name:enflame-gcu-resource, uuid:1f83296b-10c4-4f8c-9a81-ef2860472980) success
2022/10/10 09:43:33 INFO /home/zxx/go/src/topscloud/cloud/k8s-operator/gcu-operator/model/gcu-resource.go:166 init gcuResource(name:enflame-gcu-resource, uuid:1f83296b-10c4-4f8c-9a81-ef2860472980) deploy condition success
......

/var/log/topscloud/gcu-operator/ # vim restart-docker.log
[Wed May 24 10:02:07 UTC 2023] Work start...
[Wed May 24 10:02:07 UTC 2023] ----------Wait file /var/log/topscloud/gcu-operator/hostIP.list created----------
[Wed May 24 10:03:18 UTC 2023] It is detected that container toolikit deploy success, ready restart docker
[Wed May 24 10:03:19 UTC 2023] Host 10.12.110.166 docker default runtime is enflame, need not restart
[Wed May 24 10:03:19 UTC 2023] End of current restart cycle, remove /var/log/topscloud/gcu-operator/hostIP.list
[Wed May 24 10:03:19 UTC 2023] ----------Wait file /var/log/topscloud/gcu-operator/hostIP.list created----------
[Wed May 24 10:04:24 UTC 2023] Received kill signal, process exist
......

```


### 如何卸载gcu-resource和operator

如果只想卸载gcu-resource，可以通过kubectl命令一键卸载:

```shell
gcu-operator_{VERSION} # kubectl delete -f example/gcu-resource.json
```

如果也要卸载operator，则可以直接执行一键卸载脚本，该脚本会将CR和operator全部卸载，并终止重启docker的进程:

```shell
gcu-operator_{VERSION} # ./delete-operator.sh
```


### gcu-operator如何获取gcu软件的镜像

支持用户通过build-component-image.sh脚本一键构建全部软件栈镜像


### 卸载的时候可以单独只卸载某个组件吗？

可以通过部署的时候编辑多个CR进行实现。
单个CR在卸载的时候，该CR所部署的组件将会全部卸载，不可以指定只卸载某个组件。但如果部署的时候将不同组件编辑到不同的CR下进行部署，那么便可以通过只卸载某个CR的方式达到卸载单个组件的目的。


### operator在部署container-toolkit之后如果docker重启了，会影响后续组件的部署流程吗？

不会。docker重启虽然会导致已经部署的组件，包括k8s集群等相关pod重启，但重启后，gcu-operator会从断点处继续执行后续的部署流程。


### 常见报错

```shell
gcu-operator_{VERSION}/component-images# ./build-component-image.sh build
Building component images for system nameID: ubuntu18.04, arch: x86_64 start...
Building artifact.enflame.cn/enflame_docker_images/enflame/gcu-driver:default for system nameID: ubuntu18.04, arch: x86_64 start...
cp: cannot stat './enflame-x86_64-*.run': No such file or directory
```

该错误提示构建driver镜像的时候缺少相关文件，请参照上述文档下载topsrider包，并将这些文件拷贝到相应目录下后执行构建脚本。


```shell
gcu-operator_{VERSION}# ./deploy-operator.sh
...
STATUS: deployed
REVISION: 1
TEST SUITE: None
Are you want to create gcu-resource now?(y/yes, n/no)?
y
Error: You should install the driver on the node in advance, or use gcu-resource to install the driver.
                You can also get more information from the user guide.
```

该错误提示你的机器没有安装gcu driver，也没有使用operator安装gcu driver。
解决办法二选一：
- 提前在机器上安装好driver
- 使用./build-component-image.sh构建驱动镜像，并将example/gcu-resource.json文件中的spec.driver.notDeploy字段的值改为false


```shell
gcu-operator_{VERSION}# ./deploy-operator.sh
WARN: It is detected that KMD is not installed on the system, the following information is useful for you:
#####################
1. It is recommended that you install KMD on all GCU nodes in the k8s cluster before deploying gcu-operator
2. You can also use gcu-operator to automatically install KMD for all GCU nodes in the cluster, then you need to do two things:
            1) The gcu-resource.yaml file and gcu-resrouce.json file provided by gcu-operator do not install drivers by default.
                * If you deploy gcuResource together with gcu-operator(values.gcuResource.deploy=true), make sure 'gcuResource.driver.notDeploy=false' in values.yaml;
                * If you deploy gcuResource after installing gcu-operator(values.gcuResource.deploy=false), make sure 'spec.driver.notDeploy=false' in example/gcu-resource.json;
            2) If you are not using gcu-operator in the topsrider package, then you should download the topsrider package and unzip it,
                then copy all the files in the driver directory to the ./component-images/driver/ directory
                Example:
                    # chmod +x TopsRider_t2x_*_deb_internal_amd64.run
                    # ./TopsRider_t2x_*_deb_internal_amd64.run -x
                    # cp TopsRider_t2x_*_deb_internal_amd64/driver/* ./component-images/driver/
            Next, you can build the driver image according to the user guide.
#####################
Error: It is detected that gcuResource will be installed together with gcu-operator, but the driver is not installed in the system.
            You should install the driver on the node in advance, or use gcu-resource to install the driver.
            You can also get more information from the user guide.

```

该错误提示安装软件栈时，你的机器没有安装gcu driver，也没有使用operator安装gcu driver。
解决办法二选一：
- 提前在机器上安装好driver
- 使用./build-component-image.sh构建驱动镜像，并将gcu-operator-chart/values.yaml文件中的gcuResource.driver.notDeploy字段的值改为false




## 附录

gcu-operator及其相关组件依赖于helm3和go1.18+，以下安装方式供您参考。

### helm3安装示例

1、下载需要的版本并解压

下载链接：https://github.com/helm/helm/releases
如：
```shell
$ wget https://get.helm.sh/helm-v3.12.0-linux-amd64.tar.gz      # 下载软件包
$ tar -zxf helm-v3.12.0-linux-amd64.tar.gz                      # 解压

```

2、将可执行文件移动到环境变量PATH下
```shell
$ mv linux-amd64/helm /usr/local/bin/helm

```

3、查看helm安装成功
```shell
$ helm version
version.BuildInfo{Version:"v3.12.0", GitCommit:"c9f554d75773799f72ceef38c51210f1842a1dea", GitTreeState:"clean", GoVersion:"go1.20.5"}

```


### go1.18+安装示例

1、下载需要的版本并解压到/usr/local下

下载链接：https://golang.google.cn/dl/
如：
```shell
$ wget https://golang.google.cn/dl/go1.20.5.linux-amd64.tar.gz           # 下载软件包
$ tar xfz go1.20.5.linux-amd64.tar.gz -C /usr/local                      # 解压到/usr/local目录下

```

2、配置go环境变量

编辑安装用户空间下.profile文件，文件末尾追加以下内容后保存：
```shell
$ vim /root/.profile
......
export GO111MODULE=on
export GOROOT=/usr/local/go
export GOPATH=$HOME/go
export GOBIN=$GOPATH/bin
export PATH=$GOPATH:$GOBIN:$GOROOT/bin:$PATH

```

使配置生效：
```shell
$ source /root/.profile

```

3、查看go安装成功

```shell
$ go version
go version go1.20.5 linux/amd64

```