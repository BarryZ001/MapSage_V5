![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1418740443610374144/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91TopsRider_v2_1.jpg?Expires=1758382107&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=d0uI2Mx7kptAxXtWuP36Nf36FR4%3D) 
 
TopsRider发布说明
 
 
v2.5.136
 
 
2024年1月30日
 
 
(Enflame
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1418740443610374144/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91TopsRider_v2_2.jpg?Expires=1758382107&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=oZtDAIq0EuXR5Xa3MYtCuMeUOHA%3D) 
 
版权所有 © 2022上海燧原科技有限公司保留所有权利
 
 
Page 2 of 6
 
 
1简介...........................................................................................................................3
 
 
2功能优化................................................................................................................... 3
 
 
2.1       新增/修改特性............................................................................................................3
 
 
2.1.1     TopsRuntime..............................................................3
 
 
2.1.2     TopsCloud................................................................ 3
 
 
2.1.3     BigModel................................................................. 3
 
 
2.1.4     TopsTransformer.......................................................... 3
 
 
2.1.5     TopsModelgraph.......................................................... 3
 
 
2.1.6     TopsModel................................................................3
 
 
2.1.7     Framework...............................................错误！未定义书签。
 
 
2.2      I2x模型支持...............................................................................................................4
 
 
2.2.1     AIGC模型-Topstransformer框架支持模型.................................... 4
 
 
2.2.2    AIGC模型支持需要使用分布式推理框架 TopsDistInfer(alpha版本)，详见文档.... 4
 
 
2.2.3     AIGC模型-SD模型.........................................................4
 
 
2.3      T2x支持模型..............................................................................................................4
 
 
2.3.1     PaddlePaddle模型支持..................................................... 4
 
 
2.3.2     PyTorch模型支持.......................................................... 4
 
 
2.3.3     TensorFlow模型支持.......................................................4
 
 
2.3.4     AIGC模型支持.............................................................4
 
 
2.3.5     ARM模型支持（此次无新增）...............................................4
 
 
3问题修复................................................................................................................... 4
 
 
4文档...........................................................................................................................5
 
 
5使用限制................................................................................................................... 5
 
 
6已知问题................................................................................................................... 5
 
 
7     EFSMI版本................................................................................................................5
 
 
8操作系统和 P y thon支持.......................................................................................... 5
 
 
8.1       适配说明.....................................................................................................................5
 
 
8.2      操作系统支持列表......................................................................................................5
 
 
8.3       P ython支持............................................................................................................... 6
 
 
9附录...........................................................................................................................6
 
 
(Enflame
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1418740443610374144/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91TopsRider_v2_3.jpg?Expires=1758382107&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=JmAIrhxcY7UNLUwW6wvym5dJnMs%3D) 
 
版权所有 © 2022上海燧原科技有限公司保留所有权利
 
 
Page 3 of 6
 
 
1简介
 
 
TopsRider v2.5update1发布说明，2.5.136版本适用于 i20、T20、T21设备。新增/修改特性和问题修复部分相对于 v2.5.115的变更。
 
 
2功能优化
 
 
2.1新增/修改特性
 
 
2.1.1    TopsRuntime
 
 
1、 topsruntime package中动态库的 LD Version Script做了修改，其中版本名改为动态库名称加版本号，例如 libtopsrt.so的版本名为 topsrt_x.x，符号使用名称进行精确匹配
 
 
2、有依赖 topsruntime package中动态库的组件都需要基于新的动态库重新编译。否则会遇到类似 “version`CAPS_0.8` not found”的错误
 
 
2.1.2    TopsCloud
 
 
1、 k8s-device-plugin GCU K8S集群资源调度管理插件：新增 PCIe Switch亲和性调度支持
 
 
2、enflame-container-toolkit GCU容器化管理工具套件：修复
 
 
k8s>=1.24系统启用 systemd driver时出现 crash的 bug，支持containerd
 
 
2.1.3    BigModel
 
 
1、 Megatron_Deepspeed
 
 
-新增序列并行支持
 
 
2、 DeepSpeed
 
 
-新增 MoE支持
 
 
2.1.4    TopsTransformer
 
 
1、新增支持 TopsTransformer框架
 
 
2、新增支持 TopsTransformer框架的模型：ChatGLM-1-6B; ChatGLM-2-6B; ChatGLM-3-6B
 
 
2.1.5     To p sModel g ra p h
 
 
1、新增支持 TopsModelgraph工具
 
 
2、支持本地/远程打开 ONNX，PB，HLIR模型图文件，支持图形放大/缩小，还原及属性，输出等信息查看，支持模型子图查看。
 
 
3、支持 ONNX，PB，HLIR模型图缩略图展示及鼠标拖动定位。
 
 
4、支持 ONNX，PB，HLIR模型算子信息查询，支持搜索历史查看；搜索内容可区分大小写；正则表达式，支持搜索类型选择。
 
 
5、支持ONNX，PB，HLIR模型算子属性，输入，输出信息查看，其中属性信息以表格的形式列出属性的名称，类型，值。输入信息以表格的形式列出输入节点的别名，名称，类型，操作（包含详情和复制数据），点击详情可弹出新的窗口用于查看值信息，输出信息以以表格的形式列出输入节点的别名，名称。
 
 
6、支持算子类型统计视图，与对应的模型编辑器视图有联动作用，当切换模型编辑器时，算子类型统计视图自动刷新为新的统计内容，支持表格内容导出为csv文件。
 
 
2.1.6     TopsModel
 
 
1、InternLM-7B--推理--2卡流水性能优化
 
 
2、ChatGLM-6Bfinetune--性能优化
 
 
(/Enflame
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1418740443610374144/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91TopsRider_v2_4.jpg?Expires=1758382107&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=VqqKk9VpOJq%2FGgVQd7gXsO7cSls%3D) 
 
版权所有 © 2022上海燧原科技有限公司保留所有权利
 
 
Page 4 of 6
 
 
2.1.7    Torch_GCU
 
 
1、 Torch_GCU1.10支持编译子图 Executable按需延迟加载，在多子图场景下降低内存开销
 
 
2、 Torch_GCU1.10支持通过环境变量控制是否将计算图中的 Int64和FP64数据类型隐式转换为对应的 Int32以及 FP32数据类型
 
 
2.2     I2x模型支持
 
 
2.2.1    AIGC模型-Topstransformer框架支持模型
 
 
| 模型名称                        精度                                  说明|模型名称                        精度                                  说明|模型名称                        精度                                  说明|
| ---|---|---|
| ChatGLM-1-6B|FP16|新增支持|
| ChatGLM-2-6B|FP16|新增支持|
| ChatGLM-3-6B|FP16|新增支持| 
 
2.2.2   AIGC模型支持需要使用分布式推理框架 TopsDistInfer(alpha版本)，详见文档此次无新增
 
 
2.2.3     AIGC模型-SD模型
 
 
此次无新增
 
 
2.3     T2x支持模型
 
 
2.3.1     PaddlePaddle模型支持
 
 
此次无新增
 
 
2.3.2    PyTorch模型支持
 
 
此次无新增
 
 
2.3.3     TensorFlow模型支持
 
 
此次无新增
 
 
2.3.4    AIGC模型支持
 
 
AIGC模型需要使用 ai develo p ment toolkit套件（beta版本），使用说明详见 ai development toolkit软件包内 Readme
 
 
| 模型名称                          框架                  说明|模型名称                          框架                  说明|模型名称                          框架                  说明|
| ---|---|---|
| Qwen-14Bfinetune|PyTorch|支持 8卡 finetune（lorA）|
| |||
| ||| 
 
2.3.5     ARM模型支持（此次无新增）
 
 
| 模型名称|框架                      数据类型            卡数|框架                      数据类型            卡数|框架                      数据类型            卡数|
| ---|---|---|---|
| Resnet50 v1.5|PyTorch|EFP|2|
| BERT Large|PyTorch|EFP|2|
| BERT Base|PyTorch|EFP|2|
| YOLOv3|PyTorch|EFP|2|
| YOLOv5s|PyTorch|EFP|2|
| Unet|PyTorch|EFP|2|
| SSD|PyTorch|EFP|2| 
 
3问题修复
 
 
| 1、|topsprof运行可执行文件报错，但是能够运行出结果|
| ---|---|
| 2、|i20文档描述与代码存储路径不符|
| 3、|chatglm2-6b子模型（修改模型结构）在特定输入维度推理出错| 
 
(Enflame
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1418740443610374144/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91TopsRider_v2_5.jpg?Expires=1758382107&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=MXsNc8zVmb7VcqC%2BGt5Uys6Eyx4%3D) 
 
版权所有 © 2022上海燧原科技有限公司保留所有权利
 
 
Page 5 of 6
 
 
4文档
 
 
🟥新增 TopsModelgrah文档
 
 
5使用限制
 
 
●PaddlePaddle框架下必须使用 Python 3.8及以上
 
 
●Topscc组件 kernel c++不能支持纯虚成员函数
 
 
●   TopsGDB在 i20设备上支持 C/C++源码调试；在 T20、T21设备上支持汇编级别的调试，对于源码调试功能将在后续版本支持
 
 
●   RHEL 9.2不支持虚拟化功能
 
 
●   TopsTransformer框架的模型，运行时需要使用环境变量:export
 
 
EFRT_MAX_KERNEL_EMIT_PER_STREAM=2
 
 
6已知问题
 
 
i20训练模型
 
 
Pytorch bert_large单卡/8卡 OOM
 
 
Pytorch ssd_mobilenetv2 8卡偶现 OOM
 
 
●   TopsCloud
 
 
deepspeed初始化和 load-ckpt之间切图，否则权重加载不生效，参数没发生改变
 
 
🟥 deepspeed API测试中流水并行的精度不达标，受限于底层实现 2.5release暂无法处理
 
 
deepspeed API批量测试中 cpu-adam必现 hang/runtime_error
 
 
●   TopsPlatform-TopsCC
 
 
kernel使用两个 private DTE比只使用一个带来性能下降
 
 
cmake不能够使用 topscc作为 cc（c语言编译器）
 
 
●      ctrl+c或者 SIP超时后，这之前的printf信息不能输出到控制台
 
 
●当使用-O0对 to p scc kernel进行编译时，执行该 kernel的操作会默认为同步操作
 
 
Launch TopsCC kernel时，线程总数（grid dim* block dim）不能大于 1024
 
 
top scc使用-O0编译报错
 
 
C循环中更新 leaptr的 offset，出现 tar spill happen
 
 
🟥不支持 barrier功能
 
 
7     EFSMI版本
 
 
● 1.21.0
 
 
8操作系统和 P y thon支持
 
 
8.1适配说明
 
 
Host环境：仅Enflame Driver对此OS环境做兼容适配，Docker运行 Ubuntu
 
 
●   Docker环境：软件栈功能已做适配测试，需使用相同OS的Host
 
 
8.2操作系统支持列表
 
 
所有OS支持列表OS
 
 
| 操作系统名称                   架构         内核版本|操作系统名称                   架构         内核版本|操作系统名称                   架构         内核版本|GCC|GLIBC       说明|GLIBC       说明|
| ---|---|---|---|---|---|
| Ubuntu18.04.z(z<=6)|x86|4.15.0   &5.4|7.5|2.27|Host& Docker|
| Ubuntu18.04.6|aarch64|5.4|7.5|2.27|Host& Docker|
| Ubuntu20.04.z(z<=5)|x86|5.4        &5.11     &5.13      &|9.3|2.31|Host& Docker| 
 
(Enflame
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1418740443610374144/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91TopsRider_v2_6.jpg?Expires=1758382107&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=6QfOb3yTIK7yjCcnf1q0rZhjzwQ%3D) 
 
版权所有 © 2022上海燧原科技有限公司保留所有权利
 
 
Page 6 of 6
 
 
| ||5.15||||
| ---|---|---|---|---|---|
| Ubuntu 22.04.z(z<=1)|x86|5.15|11.2|2.35|仅   kmd在   Host上已适配， Docker中请使用其他 OS|
| CentOS 7.9|x86|3.1|5.5|2.17|Host& Docker|
| Kylin v10|x86|4.19.0|7.3|2.28|仅   kmd在  Host上已适配，Docker使用 Ubuntu|
| UOS 20 Server|x86|4.19.0|7.3|2.28|仅 Host适配，Docker使用 Ubuntu|
| OpenEular|X86|5.10.0|10.3.1|2.34|Host& Docker|
| 龙蜥 8.2 QU2|X86|4.18.0|8.3.1|2.28|Host& Docker|
| 龙蜥 8.6|X86|4.19.90|7.3.0|2.28|Host& Docker| 
 
8.3    Python支持
 
 
Python 3.6，Python 3.8，Python 3.10(推理模型支持)
 
 
9附录
 
 
T20/T21训练性能测试机器配置
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1418740443610374144/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91TopsRider_v2_7.jpg?Expires=1758382107&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=S%2FCY7jkVKI1FbthMtwvTByC2Jvs%3D) 
 
