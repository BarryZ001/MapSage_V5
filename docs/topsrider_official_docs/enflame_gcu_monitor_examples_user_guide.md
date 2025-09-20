
## 版本申明

| 版本 | 修改内容                                   | 修改时间   |
| ---- | ------------------------------------------ | ---------- |
| v1.0 | 将之前的版本更新为Prometheus + K8S示例说明 | 2022/10/10 |
| v1.1 | 简化内容，更新说明 | 2023/6/30 |
|      |                                            |            |
|      |                                            |            |



## 简介

Gcu-monitor-examples是一个基于prometheus + grafana + gcu-exporter的可观测方案应用简单示例。从gcu-exporter采集的指标通过 Prometheus展示到Grafana以便用户获取或设置GCU设备的运行指标与告警信息。 而gcu-exporter 依赖于EFML（Enflame Management Library）来获取Enflame GCU的运行指标信息，该exporter除了支持Enflame训练加速卡之外同时也支持推理加速卡，相应的运行指标说明参考《enflame_gcu-exporter_user_guide》，用户可以根据自己的具体需要进行可观测方案的定制化二次开发。



## 应用示例

当前topscloud_\<VERSION\>/gcu-monitor-examples\_\<VERSION\>发布包里提供了基于K8S和docker的部署示例，当前示例对应的prometheus与Grafana版本如下：

-   prom/prometheus:v2.7.1

-   grafana/grafana:6.6.0


### gcu-exporter 镜像构建

注：如果可以直接从内网docker hub 拉取gcu-exporter 镜像包，则无需这一步。

cd topscloud_xx/gcu-exporter_xx
执行，镜像构建脚本：

`./docker-image-build.sh`

```

Sending build context to Docker daemon  16.83MB
Step 1/5 : FROM ubuntu:18.04
 ---> 35b3f4f76a24
Step 2/5 : RUN apt-get update && apt-get install -y dmidecode
 ---> Using cache
 ---> ea0682fd7490
Step 3/5 : COPY gcu-exporter /usr/bin/
 ---> 07025c2b165e
Step 4/5 : EXPOSE 9400
 ---> Running in af3d2fd53bbd
Removing intermediate container af3d2fd53bbd
 ---> 7d5e12fe94fa
Step 5/5 : ENTRYPOINT ["/usr/bin/gcu-exporter"]
 ---> Running in 3e748bd92997
Removing intermediate container 3e748bd92997
 ---> b45a48a7b486
Successfully built b45a48a7b486
Successfully tagged artifact.enflame.cn/enflame_docker_images/enflame/gcu-exporter:latest

```


### k8s部署示例

在K8s集群已经部署好的前提下，采用`gcu-monitor-examples/k8s/k8s-monitor.sh`脚本，如 ：

```
# cd k8s
#./k8s-monitor.sh --help
Usage: k8s-monitor.sh [command]
command:
    apply    Apply k8s yaml
    delete   Delete k8s yaml
example:
    k8s-monitor.sh apply
    k8s-monitor.sh delete
```


拉起运行指标观测示例，执行：

```bash
 # ./k8s-monitor.sh apply
```
下线运行指标观测示例，执行：

```bash
 # ./k8s-monitor.sh delete
```


以上步骤需要注意先配置`yaml/gcu-exporter.yaml`里的`gcu-exporter`镜像路径，镜像路径需要根据本地的实际情况修改，如下：

```
      containers:
        - name: gcu-exporter
          image: artifact.enflame.cn/enflame_docker_images/enflame/gcu-exporter:latest
          imagePullPolicy: IfNotPresent #Always
          securityContext:
            privileged: true
```

另外用户还可以按需自我定制yaml目录下的yaml文件：

```
yaml/gcu-exporter.yaml
yaml/grafana.yaml
yaml/namespace.yaml
yaml/prometheus.yaml
```

以上文件中prometheus 与 grafana的镜像如果连不了外网需要先下载后再导入。


### docker部署示例

docker部署示例，采用`gcu-monitor-examples/docker/docker-monitor.sh`，如 ：

```
# cd docker
#./docker-monitor.sh --help
Usage: docker-monitor.sh [command]
command:
    init        init docker compose
    up          docker compose up -d
    down        docker compose down
example:
    docker-monitor.sh up
    docker-monitor.sh down
```


拉起运行指标观测示例，执行：
```bash
 # ./docker-monitor.sh up
```

这一步如果出现 `docker-compose: command not found` 这样的log，如下：

```
FO] Action start : up
[INFO] docker-compose up -d
./docker-monitor.sh: line 60: docker-compose: command not found
[ERROR] Action is failed : up

```

则需要先安装docker-compose 命令：
```bash
# cd topscloud_xxx/gcu-monitor-examples_xxx/docker
# ./docker-monitor.sh init
```


关闭运行指标观测示例，执行：

```bash
 # ./docker-monitor.sh down
```

以上示例需要先根据本地实际情况配置docker-compose.yaml里 `gcu-exporter`的镜像路径，例如`image: artifact.enflame.cn/enflame_docker_images/enflame/gcu-exporter:latest`。
prometheus 与 grafana的镜像如果连不了外网需要先下载后再导入，docker-compose.yaml 内容如下：

```
version: '2.0'

services:
    prometheus:
        container_name: prometheus
        image: prom/prometheus:v2.7.1
        volumes:
            - ./prom/prometheus.yml:/etc/prometheus/prometheus.yml:ro
        ports:
            - 9090:9090
        network_mode: host

    grafana:
        container_name: grafana
        image: grafana/grafana:7.5.4 #6.6.0
        volumes:
            - /var/lib/grafana:/var/lib/grafana
        ports:
            - 3000:3000
        network_mode: host

    gcu-exporter:
        container_name: gcu-exporter
        image: artifact.enflame.cn/enflame_docker_images/enflame/gcu-exporter:latest
        privileged: true
        volumes:
            - /usr/lib/libefml.so:/usr/lib/libefml.so
            - /usr/local/efsmi:/usr/local/efsmi
        ports:
            - 9400:9400
        network_mode: host

```




### 通过Prometheus 查看运行指标

通过浏览器访问prometheus服务，访问http://\<NodeIP\>:9090, prometheus默认端口9090（注意配置K8S的端口映射）， 依次选择status-\> target 查看endpoint status，如果每个服务的status 为 UP代表节点运行程序正常启动，如果为DOWN 则代表节点运行程序异常。



### 通过Grafana查看运行指标

注：如果Grafana版本不一致，以下步骤与过程也可能会不一致，需要根据具体情况进行调整。



#### 登录进Grafana

在浏览器地址栏输入grafana服务的IP和端口，**http://\<NodeIp\>:3000** , grafana的默认端口是3000， 默认 **Username: admin**， 默认：**Password: admin**。



### 添加Grafana数据源

在Grafana的首页里点击 `Add your first data source`选择`Prometheus` 作为数据源，再根据Prometheus的配置选项提示配置相应信息，比如在 Prometheus 的配置选项 URL 里 填写 `http://localhost:9090` , 然后再点 左下角的 `Save & Test` ，即可完成Prometheus的简单配置。



### 导入Grafana UI模板文件

当前gcu-monitor-examples 提供了一个简单的Grafana UI模板示例，Grafana UI模板json文件在目录gcu-monitor的目录 dashboard下 。

模板导入过程如下：

```
在Grafana首页左边的 Dashboards选项里，依次点击 -> Manage -> Import -> Upload JSON file， 根据界面提示即可导入模板。
```

导入模板成功后在Grafana的主页面左下角的dashboards --> Recently viewed dashboards 下即可看到导入的模板，点击进去即可查看运行指标示例。


## 注意事项

- 当前gcu-monitor 只是提供了gcu-exporter 的Prometheus 简单示例。如果要在生产上使用，建议用户可以根据自己的具体要求参考运行指标说明文档 《enflame_gcu-exporter_user_guide》，进行合理的二次开发；
- **注：本应用示例仅供参考，而非一键开箱即用方案。**



