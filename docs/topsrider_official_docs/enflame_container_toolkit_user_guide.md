
## 版本申明

| 版本   | 修改内容 | 修改时间      |
|:----:|:----:|:---------:|
| v1.0 | 初始化  | 2023/5/11 |
|      |      |           |


## 简介

Enflame-container-toolkit 是一个容器扩展工具套件，通过这一工具套件使得燧原的GCU卡可以在docker、containerd等容器内使能。k8s-device-plugin 与enflame-container-toolkit组合使用以支撑Enflame GCU 在K8S系统内的资源调度，本文档主要描述enflame-container-toolkit的单独使用方式。

Enflame-container-toolkit依赖于以下三个条件:

- Enflame Driver已经在OS里安装好，缺少这一步会找不到Enflame GCU加速卡；

- Docker-ce 已经在OS里安装好；

- 用户: root, 或者 `sudo [command]` ， 安装需要root权限。



## container-toolkit安装包

在topscloud/container-toolkit安装包里涵盖以下几个文件，如下：

```
container-toolkit_1.0.1/
├── daemon.json.template
├── enflame-container-toolkit_1.0.1_amd64.deb
├── enflame-container-toolkit_1.0.1_x86_64.rpm
├── install.sh
├── LICENSE.md
└── README.md
```

1） enflame-container-toolkit_1.0.1_amd64.deb 是 ubuntu 系统安装包，安装命令：

```
# dpkg -i enflame-container-toolkit_1.0.1_amd64.deb
```

2）enflame-container-toolkit_1.0.1_x86_64.rpm 是 tlinux ，centos，redhat系统安装包，安装命令：

```
# rpm -ivh enflame-container-toolkit_1.0.1_x86_64.rpm
```

3）daemon.json.template为 docker /etc/docker/daemon.json 配置模板，内容如下:

```bash
{
    "default-runtime": "enflame",
    "runtimes": {
        "enflame": {
            "path": "/usr/bin/enflame-container-runtime",
            "runtimeArgs": []
        }
    },

    "registry-mirrors": ["https://mirror.ccs.tencentyun.com", "https://docker.mirrors.ustc.edu.cn"],
    "insecure-registries": ["127.0.0.1/8"],
    "max-concurrent-downloads": 10,
    "log-driver": "json-file",
    "log-level": "warn",
    "log-opts": {
        "max-size": "10m",
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

可以根据自己的具体需求，更改daemon.json.template 的内容，然后copy到 /etc/docker下，这个配置文件会将docker的default-runtime 设置成enflame container runtime。需要注意的是考虑到安全问题，安装 enflame-container-toolkit 时，当OS 的 /etc/docker 下已经有 daemon.json 存在，则不会覆盖；

4） install.sh， 一键安装脚本, 执行 ./install.sh 会将enflame-container-toolkit自动安装到系统。

5） LICENSE.md ，版权说明;

6） README.md，readme 文件



## 安装container-toolkit

在topscloud_xxx内执行
```
cd container-toolkit_x.x.x
./install.sh
```

以上步骤会把enflame-container-toolkit 安装进系统，例如：

```
topscloud_2.0.3/container-toolkit_1.0.1# ./install.sh
(Reading database ... 235488 files and directories currently installed.)
Preparing to unpack enflame-container-toolkit_1.0.1_amd64.deb ...
Unpacking enflame-container-toolkit (1.0.1) over (1.0.1) ...
Setting up enflame-container-toolkit (1.0.1) ...
[INFO] enflame-container-toolkit had been installed
[INFO] log dir: /var/log/enflame
[INFO] config dir: /etc/enflame-container-runtime
[INFO] ldconfig...
[INFO] Docker service is restarting...
[INFO] Docker service had been restarted
Processing triggers for libc-bin (2.23-0ubuntu11) ...
[INFO] systemctl restart docker
```

> 注：

1）这一步需要注意daemon.json 的内容要根据自己的实际情况按需修改，再安装进/etc/docker下，默认如果/etc/docker/daemon.json 已存在则不会覆盖；

2）要确保deamon.json 里 \"default-runtime\": 为\"enflame\"， 不然docker识别不了enflame GCU；

3）对应的logs目录为/var/log/enflame/enflame-container-runtime，可以从这里获取container-toolkit 的运行日志信息；




## 环境变量

enflame-container-toolkit 采用  `ENFLAME_VISIBLE_DEVICES`  这个环境变量挂载GCUs进容器内。

### 这个变量支持的值

- `0,1,2`, `GCU-0,GCU-1,GCU-2` …:  采用逗号`,` 分割.
- `all`: 默认值，挂载所有的GCUs设备进容器内.
- `none`: 全部不挂载
- `void` 为 *空* 或 未设置，这时 `enflame-container-runtime` 等同 `runc`.

### GCU挂载用例

直接docker挂载用例如下：

```
### all gcus
$ docker run -it --network host -e ENFLAME_VISIBLE_DEVICES=all enflame/enflame:latest /bin/bash

### gcu 0,1,2
$ docker run -it --network host -e ENFLAME_VISIBLE_DEVICES=0,1,2 enflame/enflame:latest /bin/bash

### gcu 0
$ docker run -it --network host -e ENFLAME_VISIBLE_DEVICES=0 enflame/enflame:latest /bin/bash

### enflame runtime
$ docker run -it --network host --runtime enflame  enflame/enflame:latest /bin/bash

### enflame-docker
$ enflame-docker run  -it --network host enflame/enflame:latest /bin/bash
```

## 配置文件与日志


1） 配置文件路径 ：`/etc/enflame-container-runtime/config.toml`，内容如下：

```
[enflame-container-cli]
debug = "/var/log/enflame/enflame-container-runtime-hook.log"
environment = []

ldconfig = "@/sbin/ldconfig.real"

[enflame-container-runtime]
# debug = "/var/log/enflame/enflame-container-runtime.log"
```


2）日志路径：`/var/log/enflame`


## 常见问题

### runc 版本

建议的 runc 最低版本为 1.0.0-rc10
