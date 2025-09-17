7
 
 
TopsRider 软件栈安装手册
 
 
V2.1
 
 
2023年1月
 
 
TopsRider 软件栈安装手册
 
 
目录
 
 
目录
 
 
1安装综
 
 
述...
 
 
1.1综
 
 
述...
 
 
··       ..3
 
 
1.2名词解
 
 
释...
 
 
3
 
 
2安装说
 
 
明...
 
 
·              ..4
 
 
2.1使用前说
 
 
明...
 
 
4
 
 
2.2安装使
 
 
用..
 
 
·4
 
 
2.2.1静默安
 
 
装...4
 
 
2.2.2交互式操作界面安
 
 
装.....                                             5
 
 
2.3软件栈卸
 
 
载...
 
 
2.4 TopsInstaller 命令说
 
 
明...
 
 
7
 
 
2.4.1参数解
 
 
释...
 
 
7
 
 
2.4.2重点参数说
 
 
明...
 
 
8
 
 
3附
 
 
录...
 
 
10
 
 
3.1 Docker 制
 
 
作...
 
 
10
 
 
3.2 Docker 启
 
 
动...
 
 
10
 
 
3.3版本历
 
 
史...
 
 
..10
 
 
2/11
 
 
TopsRider 软件栈安装手册
 
 
1
 
 
安装综述
 
 
1
 
 
安装综述
 
 
1.1综述
 
 
TopsRider 是燧原软件栈的总称，覆盖用户运行所需要的驱动、开发框架、应用软件等，为了帮助大家快速安装，节约开发流程，燧原官方提供了TopsInstaller 安装方式，快速实现软件栈的安装。
 
 
1.2名词解释
 
 
名词
 
 
GCU
 
 
TopsInstaller
 
 
解释
 
 
General Compute Unit 的缩写， Enflame 生产的高性
 
 
能并行计算设备。将 TopsRider 软件栈整体封装后的自解压软件安装包
 
 
3/11
 
 
TopsRider 软件栈安装手册
 
 
2安装说明2安装说明
 
 
2.1使用前说明
 
 
当下载好对应的 TopsInstaller 后，需要注意以下事项：
 
 
·安装时请确保具有 root 权限。
 
 
·TopsInstaller 会自动识别用户安装环境是 Host 或 Docker 环境，默认会在Host 环境安装 Driver 相关内容，在 Docker 环境安装除 Driver 外的其他软件内容。用户也可以将全部软件安装在 Host 环境中，但在 Docker 环境中只能安装 Driver 以外的软件
 
 
·TopsInstaller 提供交互式操作界面安装和静默安装
 
 
安装方式解释如下
 
 
安装方式
 
 
静默安装
 
 
交互式操作界面安装
 
 
说明
 
 
一行命令，无需其他操作即可完成 TopsRider 的安装提供可视化交互界面，让用户清晰感知 TopsInstaller 中内容
 
 
适用人群
 
 
已对于燧原软件栈比较了解的开发者首次接触燧原软件栈的开发者
 
 
2.2安装使用
 
 
首先请确认 TopsRider_{filename}.run 为可执行文件，如果不是，需要手动添加执行权限后才可正常运行。
 
 
chmod +x TopsRider_{filename}. run
 
 
2.2.1静默安装
 
 
步骤一：下载安装包后，在 Host 端执行
 
 
#需要使用root权限
 
 
./TopsRider_{filename}.run -y
 
 
#Or
 
 
./TopsRider_{filename}.run --silence
 
 
步骤二：下载安装包后，在 Docker 端执行
 
 
这里默认用户已经有 Docker 的环境，如果没有可参考 Docker 制作
 
 
注解：
 
 
·TopsInstaller 支持多种框架安装，由于不同框架版本对于 protobuf 版本支持情况不一致，用户安装 TopsRider 后需要再次单独进行框架安装，推荐用户在使用不同框架，使用不同的 docker， 避免同一个 docker 环境下安装多个框架。
 
 
4/11
 
 
TopsRider 软件栈安装手册
 
 
2
 
 
安装说明
 
 
1、针对 PaddlePaddle 框架，分两步执行如下命令
 
 
#需要使用root权限#第一步安装除框架之外的基础软件栈./TopsRider_{filename}.run -y # 第二步安装对应的框架 ./TopsRider_{filename}.run -y -C paddlepaddle-gcu，paddle-custom-gcu
 
 
2、针对 Pytorch 框架，分两步执行如下命令
 
 
#需要使用root权限#第一步安装除框架之外的基础软件栈./TopsRider_{filename}.run -y # 第二步安装对应的框架./TopsRider_{filename}.run -y -C torch-gcu
 
 
3、针对 TensorFlow 框架，分两步执行如下命令
 
 
#需要使用root权限#第一步安装除框架之外的基础软件栈./TopsRider {filename}.run -y # 第二步安装对应的框架 ./TopsRider_{filename}. run -y -C tensorflow
 
 
安装示例代码、文档、其它文件到指定目录下，默认安装到/usr/local/topsrider/下，用户也可指定安装路径
 
 
#需要使用：root 权限./TopsRider_{filename}.runn -y --install-dilir/your_dir
 
 
2.2.2交互式操作界面安装
 
 
步骤一：下载安装包后，在 Host 端执行
 
 
#需要使用root 权限./TopsRider_{filename}.run
 
 
在 host 环境下会勾选如下默认组件：用户可通过上下箭头选择来选择菜单项
 
 
·点击空格键可选择对应组件
 
 
·选择 Driver Options ， 并点击回车键，会进入驱动安装高级选项
 
 
·选择 Quit，并点击回车键，会中止并退出安装过程
 
 
·针对已经装好的组件，如在上层 python 调用时报出诸如： ImportError：libxxx. so： cannot open shared object file： No such file or directory 等某个共享库找不到对应的文件或目录的情况，可能是共享库路径未正确配置所致。可依次按如下操作进行配置：
 
 
1.可以使用 ldd [对应共享库名]命令检查共享库的依赖项；
 
 
2.如果依赖项存在 not found ，请尝试在当前环境下运行 ldconfig 命令更新动态链接器的配置。
 
 
·更多可以参考 ldconfig 相关 Linux man pages(man 8 ldconfig)
 
 
2.3软件栈卸载
 
 
使用如下命令，直接卸载本安装包内安装过的软件，在 Host七下会卸载所有包含软件，包含驱动软件，在 Docker 下会卸载除了驱动软件包外所有其它软件。
 
 
#需要使用root权限
 
 
./TopsRider_{filename}. run --uninstall
 
 
2.4 TopsInstaller 命令说明
 
 
2.4.1参数解释
 
 
7/11
 
 
TopsRider 软件栈安装手册
 
 
2
 
 
安装说明
 
 
参数         说明-x， --extract-only 解压安装包的文件到一个目录中，而不执行安装脚本-1，--list       查看安装包内可安装的软件模块列表，其中括号中为模块 id-C， --components 指定软件模块 id 安装软件模块，多个模块 id用逗号分隔-y，--silence    使用静默安装-h， --help      打印帮助指令--python       指定 Python 版本安装--install-dir    指定安装路径--uninstall      卸载
 
 
--cn          中文模式下安装
 
 
--with-dkms    安装 dkms， 默认不安装
 
 
| --no-auto-load   重启默认不自动加载，默认加载--mdev-host    以 mdev_host 模式，即虚拟机 monitor 模式加载 kmd--with-vgcu    以 vgcu 模式加载 kmd|
| ---| 
 
--peermem     安装分布式相关驱动
 
 
| --virt         安装虚拟化插件包|
| ---| 
 
2.4.2重点参数说明
 
 
·参数 peermem ：：安装 Peer Memeory 驱动插件
 
 
#需要使用root权限
 
 
./TopsRider_{filename}.run --peermem
 
 
默认情况下，驱动软件包支持传统模式的 RDMA 功能，即跨机架的两个设备想通过 RDMA 的方式传输数据时，需要将数据从设备 HBM 存储拷贝至系统内存，再通过网卡传输。如果系统中有支持相关功能的 Mellanox 硬件，可以开启GCUDirect RDMA， 来对 RDMA 功能进行加速，使得设备 HBM 存储中的数据可以直接通过网卡进行传输，不再需要拷贝至系统内存。
 
 
安装该驱动插件，即可开启 GCUDirect RDMA 功能。该插件安装在 HOST 端，安装前请确认已安装 Mellanox OFED ， 仅支持版本“5.0-2.1.8.0”、“5.4-3.1.0.0”， 具体请参考 Mellanox OFED 文档，安装命令建议使用：
 
 
#./mlnxofedinstall --add-kernel-support
 
 
注解：
 
 
·TopsRider 驱动插件(如果安装包含有)依赖 Mellanox OFED 以及TopsRider 驱动软件。如果系统中没有安装 Mellanox OFED 或者安装TopsRider 驱动软件失败，该插件都会安装失败。
 
 
·安装好之后插件独立运行，对主模块功能没有影响。可以随时卸载或者重新安装。
 
 
·在卸载后，驱动将不会使用 GCUDirect RDMA ， 转而使用传统方式完成 RDMA功能。
 
 
·在 Host OS 卸载驱动软件(如有必要)：如果有安装驱动插件，首先卸载驱动插件。
 
 
·参数 virt ： 安装虚拟化驱动插件
 
 
8/11
 
 
TopsRider 软件栈安装手册
 
 
2
 
 
安装说明
 
 
#需要使用root 权限
 
 
./TopsRider_{filename}.run --mdev_host --virt
 
 
如果需要开启虚拟机 monitor 模式，使用该命令开启 GCU 的虚拟化模块。(在Host 下安装)
 
 
注解：
 
 
·在 mdev host 模式下，设备可以按空分方式划分，并被虚拟成为最多4个单cluster 的 mdev 虚拟设备，使用 Qemu 透传给不同的虚拟机使用。
 
 
·参数 python ：指定 Python 版本
 
 
#需要使用root权限
 
 
./TopsRider_{filename}.run -y --python /usr/bin/python3.6
 
 
#Or
 
 
./TopsRider_{filename}.run -y --python python3.6
 
 
安装 whl 安装包时，安装到指定的 Python 版本环境中，如不指定参数将会默认选择 python3 命令对应的版本，请通过命令“python3-V”查看对应版本号。
 
 
9/11
 
 
TopsRider 软件栈安装手册
 
 
3
 
 
附录
 
 
3
 
 
TopsRider 软件栈安装手册
 
 
3
 
 
附录
 
 
版本
 
 
V2.0 V2.1
 
 
描述
 
 
初版修订版
 
 
日期
 
 
2022年10月2023年1月
 
 
11/11
 
 
