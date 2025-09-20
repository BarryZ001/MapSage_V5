![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_1.jpg?Expires=1758446091&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=t2ZDNetZ3VCNUNUSSpVtw8nuCeE%3D) 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_2.jpg?Expires=1758446091&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=YCBcH5AUUEsk0vCBg6NZ8N8Bo20%3D) 
 
训推一体化平台 TopsDL用户使用手册
 
 
V2.0
 
 
2022年 9月
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_3.jpg?Expires=1758446091&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=o81OZWxU%2FpguukGCxnz1CmZz9J8%3D) 
 
Enflame燧原科技
 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
2/ 47
 
 
目录
 
 
1前言.................................................................................................................. 3
 
 
1.1声明............................................................................................................................................... 3
 
 
1.2版本历史....................................................................................................................................... 3
 
 
1.3词汇表........................................................................................................................................... 3
 
 
2产品概述.......................................................................................................... 4
 
 
2.1产品简介....................................................................................................................................... 4
 
 
2.2架构图........................................................................................................................................... 4
 
 
3用户操作手册................................................................................................... 5
 
 
3.1概览............................................................................................................................................... 5
 
 
3.2数据管理....................................................................................................................................... 5
 
 
3.2.1数据集.................................................................................................................................... 5
 
 
3.2.2标签集.................................................................................................................................. 12
 
 
3.3多人标注..................................................................................................................................... 13
 
 
3.3.1多人标注任务...................................................................................................................... 13
 
 
3.3.2管理多人标注团队.............................................................................................................. 15
 
 
3.4算法开发..................................................................................................................................... 19
 
 
3.4.1算法集.................................................................................................................................. 19
 
 
3.4.2开发环境.............................................................................................................................. 21
 
 
3.5模型管理..................................................................................................................................... 25
 
 
3.5.1模型集.................................................................................................................................. 25
 
 
3.5.2批量推理.............................................................................................................................. 27
 
 
3.6训练管理..................................................................................................................................... 30
 
 
3.6.1训练任务.............................................................................................................................. 30
 
 
3.7部署上线..................................................................................................................................... 34
 
 
3.7.1模型包.................................................................................................................................. 34
 
 
3.7.2在线服务.............................................................................................................................. 36
 
 
3.8系统管理..................................................................................................................................... 39
 
 
3.8.1项目管理.............................................................................................................................. 39
 
 
3.8.2用户管理.............................................................................................................................. 41
 
 
3.8.3镜像管理.............................................................................................................................. 43
 
 
3.8.4资源全局.............................................................................................................................. 45
 
 
3.8.5算力规格.............................................................................................................................. 45
 
 
3.9修改密码..................................................................................................................................... 46
 
 
3.9.1系统中部分字段参数限制说明.......................................................................................... 46
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_4.jpg?Expires=1758446091&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=VB7Nf7jjVtngowz4bRcrlFiAGPY%3D) 
 
Enflame燧原科技
 
 
1前言
 
 
1.1声明
 
 
本文档提供的信息属于上海燧原科技有限公司和/或其子公司（以下统称“燧原科技”）所有，且燧原科技保留不经通知随时对本文档信息或对任何产品和服务做出修改的权利。本文档所含信息和本文档所引用燧原科技其他信息均“按原样”提供。燧原科技不担保信息、文本、图案、链接或本文档内所含其他项目的准确性或完整性。燧原科技不对本文档所述产品的可销售性、所有权、不侵犯知识产权、准确性、完整性、稳定性或特定用途适用性做任何暗示担保、保证。燧原科技可不经通知随时对本文档或本文档所述产品做出更改，但不承诺更新本文档。
 
 
在任何情况下，燧原科技不对因使用或无法使用本文档而导致的任何损害（包括但不限于利润损失、业务中断和信息损失等损害）承担责任。燧原科技不承担因应用或使用本文档所述任何产品或服务而产生的任何责任。
 
 
本文档所列的规格参数、性能数据和等级需使用特定芯片或计算机系统或组件来测量。经该等测试，本文档所示结果反映了燧原科技产品的大概性能。系统配置及软硬件版本、环境变量等的任何不同会影响实际性能，产品实际效果与文档描述存在差异的，均属正常现象。燧原科技不担保测试每种产品的所有参数。客户自行承担对产品适合并适用于客户计划的应用以及对应用程序进行必要测试的责任。客户产品设计的脆弱性会影响燧原科技产品的质量和可靠性并导致超出本文档范围的额外或不同的情况和/或要求。
 
 
燧原科技和燧原科技的标志是上海燧原科技有限公司申请和/或注册的商标。本文档并未明示或暗示地授予客户任何专利、版权、商标、集成电路布图设计、商业秘密或任何其他燧原科技知识产权的权利或许可。
 
 
本文档为版权所有并受全世界版权法律和条约条款的保护。未经燧原科技的事先书面许可，任何人不可以任何方式复制、修改、出版、上传、发布、传输或分发本文档。为免疑义，除了允许客户按照本文档要求使用文档相关信息外，燧原科技不授予其他任何明示或暗示的权利或许可。
 
 
燧原科技对本文档享有最终解释权。
 
 
1.2版本历史
 
 
表 1-1版本历史
 
 
| 文档版本|文档日期|文档说明|
| ---|---|---|
| V2.0|2022年 9月|定稿| 
 
1.3词汇表
 
 
表 1-2词汇表
 
 
| 术语|描述|
| ---|---|
| TopsDL|燧原科技训推一体化平台| 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_5.jpg?Expires=1758446091&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=xD2pTYB9ehcAsAiP4y2h%2FoYFRBI%3D) 
 
Enflame燧原科技
 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
4/ 47
 
 
2产品概述
 
 
2.1产品简介
 
 
随着人工智能和神经网络技术的不断进步，以及 AI算法的应用范围越来越广，客户的需求不仅局限一些主流算法的应用，更希望针对新的算法应用场景快速开发出相应的算法并快速部署到生产环境中。另外面对快速发展的人工智能市场需求，深度学习算法开发工程师在数量上还无法满足市场需求，需要有一种系统或技术能承担一些算法开发工作，减少对专业开发者的依赖。再者 AI技术日新月异，不断涌现出新的硬件芯片、不同的训练与推理框架、新的 AI算法、各种模型压缩技术、新的部署技术等等，这些新的技术都对 AI从业人员产生了新的挑战。基于这些的需求和目前的市场客观条件，算法训练和部署的低门槛、易用性、自动化变得越来越重要了。
 
 
燧原自研的训推一体化平台 TopsDL通过对 AI算法深层打磨，平台将训练和推理无缝结合，统一管理计算资源，提供一键训练部署和专业开发流程相结合的方式，降低了客户使用 AI硬件资源和深度学习算法开发的难度，助力客户搭建属于自己的 AI算法仓库和应用实例，行业覆盖政府、公安、银行、园区、企业等，满足细分行业客户的需求。
 
 
TopsDL打通包含从数据获取、数据处理、数据标注、算法构建、模型训练、到模型部署全流程链路，帮助用户快速创建和部署人工智能算法和应用。实现高可用、可视化操作、丰富的算法支持、多机器学习框架支持、多数据源接入，使得从模型训练、评估、透视，到模型部署和应用落地能全链路业务无感知。从而助力政企单位加速数字化转型并促进人工智能行业生态共建。
 
 
2.2架构图
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_6.jpg?Expires=1758446091&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=N4p9c5p4yMI8i4toYNIO05ySto0%3D) 
 
图 2-1 TopsDL架构图
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_7.jpg?Expires=1758446092&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=WeVpmhZzkA3VqRd7xI4KeQ1nGcQ%3D) 
 
Enflame燧原科技
 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
5/ 47
 
 
3用户操作手册
 
 
3.1概览
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_8.jpg?Expires=1758446092&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=3kJ0%2FpUMMzppJKLgwJHqm8C5VO0%3D) 
 
图3-1概览
 
 
如下图所示，概览展示了当前客户或者项目的资源配额和使用情况，用户可以直观地了解实时和历史资源使用情况。
 
 
3.2数据管理
 
 
K
 
 
3.2.1数据集
 
 
数据集分为“我的数据集”和“公共数据集”。
 
 
•我的数据集：当前客户或者项目内共享，客户或者项目间互相隔离，支持读写权限。
 
 
•公共数据集：所有客户或者项目共享，仅支持只读权限。
 
 
1)我的数据集
 
 
•搜索数据集：如下图所示，支持数据集类型搜索，或搜索框内输入数据集名称+回车，快速定位数据集。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_9.jpg?Expires=1758446092&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=Tjd5Y7B7uBvubrv4D%2B9QWx8LOZw%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
6/ 47
 
 
三                                                                                                                                         test6m
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_10.jpg?Expires=1758446092&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=xSnmIiwIw7JTrTGoH0yJTUCsl48%3D) 
 
共1条          10条/.
 
 
图3-2搜索数据集
 
 
•创建数据集：在数据集列表页，点击“+创建数据集”，输入数据集名称，选择“数据类型”，“标注类型”。
 
 
test
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_11.jpg?Expires=1758446092&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=9wWJTmuWgyYuohXMJoYOxnIqR70%3D) 
 
图3-3创建数据集
 
 
•编辑数据集：编辑数据集，只能修改数据集名称，其它都不能更改。
 
 
三                                                                                                 test m
 
 
数据集
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_12.jpg?Expires=1758446092&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=B58xibWCKc%2Fnc2nN2Zpn9xyPXlo%3D) 
 
图3-4编辑数据集
 
 
•删除数据集：只有当数据集版本为空时，才能删除数据集，删除数据集时，请务必谨慎，目前没有强制检查数据集和其它组件的关联状态。
 
 
•数据集版本：每个数据集默认展示最新的 5个版本，如下图所示。点击某个版本的菜单如详情、自动标注、导入、删除、导出、发布、审核发布可进入相应功能页面进行操作。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_13.jpg?Expires=1758446092&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=qM6B90C3sMaW%2BsyfWB%2FSj5aBsV8%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
7/ 47
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_14.jpg?Expires=1758446092&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=v2bhePUAJegRYgxFkweyjLuBtqg%3D) 
 
图3-5数据集版本
 
 
•数据集详情：点击数据集版本的“详情”，即可打开数据集标注功能。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_15.jpg?Expires=1758446092&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=HOVbsYc3%2FFI9NLCiW3GVPs9zwvo%3D) 
 
图3-6数据集详情
 
 
•导入数据集：点击如图所示“导入”打开导入页面，支持数据集多次导入，可导入有标注信息或无标注信息的数据集，目前支持本地数据集上传。
 
 
•本地数据集上传，如果上传的是有标注信息，可以通过“下载范本样例”，来查看支持的标注格式。
 
 
•注意：1.压缩包仅支持 zip、tar、rar格式，压缩包内文件格式为 jpg/png/jpeg；2.压缩包大小不超过 5G。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_16.jpg?Expires=1758446092&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=GnrQc6lHq0F1YHDkLJRHEnaqtRo%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
8/ 47
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_17.jpg?Expires=1758446092&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=FWydBKbyzroPXfZLef24hcSX%2FCQ%3D) 
 
图3-7导入数据集
 
 
•手动标注：导入图片之后，点击“未标注”，即可开始预处理。
 
 
•预处理：勾选多项预处理同时处理时,会按照排列顺序从上至下执行,去模糊会将不符合的图片移至已清洗,移至已清洗的数据不参与后续的处理。
 
 
<预处理
 
 
园是否启用去模糊                                                            A
 
 
保留清晰度大于等于此值的图片:50       0                       5000
 
 
清晰度参考
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_18.jpg?Expires=1758446092&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=NpHvECwP9rx1ouFHHjYY63GYe7E%3D) 
 
S
 
 
图3-8预处理
 
 
•手动标注点击“未标注”，即可开始手动标注。可以手动建立标签，给数据画框或分类。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_19.jpg?Expires=1758446093&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=N9E4Cl%2F1rEkni0QFRJfBbDkeSJE%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
9/ 47
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_20.jpg?Expires=1758446093&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=x%2FA4MeduMOOQmHRs1s%2BGBTLN0FE%3D) 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_21.jpg?Expires=1758446093&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=iFRtcmvm2rtV1rFg%2FCsL%2FY6qwU0%3D) 
 
图3-9图像分类标注
 
 
•通过“新增标签”创建一个标签，选择图片后，选择标签的箭头按钮，可以将图片导入该分类，从而进行标注。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_22.jpg?Expires=1758446093&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=sfVOQZMEXx5TFE2eq3f6GYhoYf8%3D) 
 
图3-10图像检测标注
 
 
•单击图片，即可打开“手动标注”，通过“新增标签”创建一个新的标签，选中该标签，即可进行画框操作，也可以通过未使用标签来选择一个标签（数据集已有标签），进行画框标注。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_23.jpg?Expires=1758446093&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=J%2FvFqfABVIoGarQX44Hez9KwrTE%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
10/ 47
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_24.jpg?Expires=1758446093&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=puom%2BdnUEp9L42wkCAfagwyoFg0%3D) 
 
图3-11图像分割标注
 
 
Q
 
 
•通过“新增标签”创建一个新的标签，选中该标签，即可进行选点标注操作，也可以通过未使用标签来选择一个标签（数据集已有标签），进行选点标注。可查看 标注示例，结合快捷键快速标注。
 
 
<手动脉注
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_25.jpg?Expires=1758446093&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=9j5Sm1%2BiZyPshePfDYLFONsgPLE%3D) 
 
版汉所有心上海这科技有限公司|备案许编号:PVCP18023946
 
 
图3-12关键点标注
 
 
•通过“新增标签”创建一个新的标签，选中该标签，即可进行选点标注操作，也可以通过未使用标签来选择一个标签（数据集已有标签），进行选点标注。可查看 标注示例，结合快捷键快速标注。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_26.jpg?Expires=1758446093&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=VwxOzMnKNYrfuRDWUD0%2F2IsY9ao%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
11/ 47
 
 
训推一体化平台 TopsDL用户使用手册
 
 
<手动标注
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_27.jpg?Expires=1758446093&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=sG7M8AkqagZSR0MMiz7nk8noRp4%3D) 
 
图3-13 OCR标注
 
 
•通过“新增标签”创建一个新的标签，选中该标签，即可进行选点标注操作，也可以通过未使用标签来选择一个标签（数据集已有标签），选择矩形或者多边形标
 
 
注。在右侧栏中填定标注的文字，点击最左边的按钮进行保存。
 
 
•自动标注：导入图片之后，可以进行自动标注。
 
 
•预置模型：选择设置的数据集类型选择列表中对应的模型类型，点击“确定”，进行自动标注，标注完成后，对应数据集的状态为“自动标注完成”，同时更新标注进度。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_28.jpg?Expires=1758446093&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=mBicQSZ5ohBI9srxinwXLkq87Tc%3D) 
 
图3-14预置模型
 
 
•在线模型：从“部署上线-在线服务”中复制对应类型的模型包的服务地址，点击“确定”，自动标注完成后，对应数据集的状态为“自动标注完成”，同时更新标注进
 
 
度。
 
 
Enfi
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_29.jpg?Expires=1758446093&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=H4WBOEyFos7%2BL9z1Dx8xMAQG9f4%3D) 
 
图 3-15在线模型
 
 
•导出数据：如下图，点击“导出”弹出提示窗，“是否带数据”和“是否重新生成文件”是可选，此时“下载”按钮置灰，点击确定，记录的状态变为“导出数据生成中”，生成数据后，状态变为“导出数据生成完毕”。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_30.jpg?Expires=1758446094&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=6%2Fup%2FIQnmUPVr93t7vQqi0skIok%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
12/ 47
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_31.jpg?Expires=1758446094&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=7ZGJKj158Vzcg6hN%2FQ3mNCedngE%3D) 
 
图3-16生成导出数据
 
 
•如下图，再点击“导出”，弹出提示窗，此时“下载”按钮可点击，点击“下载”，把数据下载到本地。下载的数据包括原始数据(根据选择), json格式的标注数据数据。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_32.jpg?Expires=1758446094&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=Vo%2F4klCYFpi1awZfbKt%2BDqkweSQ%3D) 
 
图3-17导出生成数据
 
 
2)公共数据集
 
 
ntiuential
 
 
预置的公共数据集，对所有客户或者项目可见，可在创建“开发环境”或“训练任务”,“批量推理”时候选择使用，但仅支持“只读”挂载。
 
 
3.2.2标签集
 
 
•搜索标签：如下图所示，搜索框内输入标签集名称+回车，快速定位标签集。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_33.jpg?Expires=1758446094&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=CpGAyJcB%2B3P%2BXCZTAcPgO6mSZrs%3D) 
 
图3-18搜索标签集
 
 
•创建标签集：点击“创建标签集”，输入标签集名称及标签集描述，点击“确认”或“取消”。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_34.jpg?Expires=1758446094&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=CDYbN0J9%2BiE1SBm786ZvL9xbUSg%3D) 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_35.jpg?Expires=1758446094&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=A8itZHTZXFWTbOJ2qe3aj22weUA%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
13/ 47
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_36.jpg?Expires=1758446094&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=2v%2BeO9eBBq%2FFDj2tax31yeEFrqA%3D) 
 
图3-19创建标签集
 
 
•标签管理：点击“标签管理”，进入标签管理页面，可以“添加标签”，“编辑”和“删除”标签。                     NO
 
 
<标签集管理(dog)
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_37.jpg?Expires=1758446094&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=zEa7Q2Q%2F9e5QJG9RS8pmon700hQ%3D) 
 
图3-20标签管理
 
 
•编辑标签集：点击“编辑”，可以修改标签集名称及描述。
 
 
•删除标签集：删除标签集及其下属标签。
 
 
3.3多人标注
 
 
3.3.1多人标注任务
 
 
1)创建多人标注任务
 
 
•如图，通过多人协同标注页面，可以“创建多人标注任务”，可选择标注团队和审核标注团队，也可设置标注截止时间和审核比例。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_38.jpg?Expires=1758446094&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=MesPqBcF1s8pd4E%2B8m%2BTJtfIQq0%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
14/ 47
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_39.jpg?Expires=1758446094&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=%2BGFkdazU0tvxMjDXBvc6nDd4HwI%3D) 
 
图3-21创建多人标注
 
 
编辑多人标注任务：如图，通过多人协同标注页面，可以“编辑多人标注任务”，“任务名称”和“数据集”不可修改，其他信息可编辑。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_40.jpg?Expires=1758446094&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=1%2FpM%2BTt%2B%2B8k03tJ9mohRazD%2By10%3D) 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_41.jpg?Expires=1758446094&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=kopujj%2FRJzpCwOkt2Z70JcemWG8%3D) 
 
图3-22编辑多人标注
 
 
•删除多人标注任务：如下图，通过多人协同标注页面，可以删除多人标注记录，用户只能删除自己创建的多人标注任务。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_42.jpg?Expires=1758446094&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=j8hVoiBu5%2BIvBazOgOyOyI1KxCI%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
15/ 47
 
 
训推一体化平台 TopsDL用户使用手册
 
 
•查看多人标注任务详情：点击任务列表中的“详情”，如下图，可查看团队成员的标注数量信息。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_43.jpg?Expires=1758446094&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=nR3lMc8YbAJyBsbH7YSqkC1RszI%3D) 
 
图3-24多人标注任务详情
 
 
O
 
 
•点击“标注详情”，进入成员个人标注详情页。如下图，可查看团队成员个人的标注数量信息，可按时间查询，默认为当天。
 
 
<返回
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_44.jpg?Expires=1758446095&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=HlTuWieEYRRFJ6uckoLHxIYaf8M%3D) 
 
图3-25标注详情
 
 
3.3.2管理多人标注团队
 
 
1)标注团队
 
 
•创建标注团队：如图，团队成员只显示标注用户，可以添加多个用户，也可以删除用户，但至少要选择一个，点击“确定”，创建的标注团队显示在列表中。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_45.jpg?Expires=1758446095&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=0ZHZ%2BenqBFCWvx25zToXrIR9Gyc%3D) 
 
图3-26创建标注团队
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_46.jpg?Expires=1758446095&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=ainBaGzPELuBdf%2F7N7mu2CL57tM%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
16/ 47
 
 
•编辑标注团队：如图，可以修改团队名称，团队描述以及团队成员，可以添加多个用户，也可以删除用户，但至少要选择一个，点击“确定”，列表更新。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_47.jpg?Expires=1758446095&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=3gFuWADAPktk3yDKl%2Fq6nTN8SMM%3D) 
 
取消    确定
 
 
图3-27编辑标注团队
 
 
•查询标注团队：如图，在右上栏框中输入团队名称，列表中显示符合条件的记录，点击“清除”，查询框置为空。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_48.jpg?Expires=1758446095&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=9B30KMuxTQ2uMMd3SaTTw%2BdfNpc%3D) 
 
共1条<   1  >   20条/页
 
 
图3-28查询标注团队
 
 
•删除标注团队：如图，选择记录中的一条，点击“删除”-“确定”，记录在列表中消失。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_49.jpg?Expires=1758446095&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=VsLx8j2Fc5%2BFkAgHwAav5iUeBWk%3D) 
 
图3-29删除标注团队
 
 
•标注用户进行标注：标注用户登录系统，进入数据管理-数据集，选择“标注任务”-点击一条记录的“详情”进 入标注页面，对分配的标注任务进行标注。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_50.jpg?Expires=1758446095&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=QUhYNE1p8M%2F4HhMyrn8mDJD62EM%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
17/ 47
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_51.jpg?Expires=1758446095&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=4YU6kmsPkpib5n8dlMUm3RViCCw%3D) 
 
图3-30执行标注
 
 
2)审核团队
 
 
•创建审核团队：如图，团队成员只显示标注用户，可以添加多个用户，也可以删除用户，但至少要选择一个，点击“确定”，创建的审核团队显示在审核团队列表中。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_52.jpg?Expires=1758446095&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=ULFXevyig0YYs911%2FNABRWmjOXw%3D) 
 
图3-31创建审核团队
 
 
Entla
 
 
•编辑审核团队：如图，可以修改团队名称，团队描述以及团队成员，可以添加多个用户，也可以删除用户，但至少要选择一个，点击“确定”，审核列表更新。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_53.jpg?Expires=1758446095&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=wf1ywtxYbWxi2bWjR%2BIx73qSK84%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
18/ 47
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_54.jpg?Expires=1758446095&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=UKvompDoBC%2F4GmKJELEspnqCZpg%3D) 
 
图3-32编辑审核团队
 
 
•查询标注团队：如图，支持模糊查询，点击“清除”，查询框置为空。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_55.jpg?Expires=1758446095&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=u%2FJ4biHZ5F1DDgxgaCG3sQ6o6DI%3D) 
 
图3-33查询标注团队
 
 
•删除审核团队：如图，选择审核团队记录中的一条，点击“删除”-“确定”，记录在审
 
 
核列表中消失。
 
 
三  uat                                                                         qing.li@enflame-tech.com le
 
 
<返回
 
 
标注团队总数:5
 
 
标注团队   审核团队                                                              请输入团队名称查询        清除  新建审核团队
 
 
是否确认删除当前任务?
 
 
团队名称                   团队成员数         创建时间                    团队描述                           取消 确定
 
 
audit-name-test-01               2                2022-06-10 18:52:22               audit-name-test-01                编辑团队   除
 
 
图3-34删除审核团队
 
 
•审核标注：审核标注用户登录系统，进入数据管理-数据集，选择“审核任务”-点击一条记录的“详情” 进入审核页面，如下图
 
 
•进入审核标注页面，查看“已标注”，点击图片右下角的“预览”，可以审核图片。
 
 
•点击预览图片的右上角的“通过”或“不通过”，进行标注审核。普通用户可以进入详情页查看标注结果。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_56.jpg?Expires=1758446096&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=yBNXzM1BbrjgduHy9MW5ri5sLVA%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
19/ 47
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_57.jpg?Expires=1758446096&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=4P5uZ%2F1%2BjmMfyzM97w0IX%2FF%2BeJE%3D) 
 
图3-35进入标注审核页面
 
 
已标注   未标注
 
 
图3-36选择图片审核
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_58.jpg?Expires=1758446096&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=WKt3NL4qwu1g%2FY2lIpOi6wQ0xf8%3D) 
 
图3-37标注审核
 
 
3.4算法开发
 
 
3.4.1算法集
 
 
算法集分为“我的算法集”和“公共算法集”。
 
 
•我的算法集：前客户或者项目内共享，客户或者项目间互相隔离，支持读写权限。
 
 
•公共算法集：有客户或者项目共享，仅支持只读权限。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_59.jpg?Expires=1758446096&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=2ksRoaRWkwMdjJ%2F%2BT7dhUZNjufo%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
20/ 47
 
 
1)我的算法集
 
 
•搜索算法集：如下所示，搜索框内输入关键词+回车，快速定位算法集。
 
 
三                                                                                                                                        tesst
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_60.jpg?Expires=1758446096&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=04e95Y%2BMOUxrN%2F0aeEUjrKML7Q0%3D) 
 
12
 
 
图3-38搜索算法集
 
 
•添加算法集：算法集列表页，点击“+添加算法集”，输入算法集名称，选择“应用场景”，“详细内容”可选。
 
 
•编辑算法集：只能修改“详细内容”，其它都不能更改。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_61.jpg?Expires=1758446096&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=SBdngRAdAMCogo8hy3xzIUWZ14A%3D) 
 
图3-39添加算法集
 
 
•删除算法集：点击算法集的“删除”按钮，弹出删除提示窗,提示是否删除，用户只能删除自己创建的算法集，不可删除他人创建的算法集。注意:只有当算法集版本为空时，才能删除算法集，删除算法集时，请务必谨慎，目前没有强制检查算法集和其它组件的关联状态。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_62.jpg?Expires=1758446096&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=Wm37RFYtR1s2AV5Q07h4Rz05ONs%3D) 
 
图3-40删除算法集
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_63.jpg?Expires=1758446096&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=yTyC0%2FyZMc%2FQceSLKB4rYqC9q0Y%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
21/ 47
 
 
训推一体化平台 TopsDL用户使用手册
 
 
•算法集版本：每个算法集默认展示最新的 5个版本，如下图所示。点击某个版本可进入文件管理功能，用户可以上传文件、文件夹，下载文件、文件夹，删除文件、文件夹。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_64.jpg?Expires=1758446096&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=HgSWx%2FjTv6s%2F0EkWl9IhU83DZ4c%3D) 
 
图3-41算法集版本
 
 
2)公共算法集
 
 
•预置的公共算法集，对所有客户或者项目可见，可在创建“开发环境”或“训练任务”时候选择使用，但仅支持“只读”挂载。
 
 
3.4.2开发环境
 
 
开发环境集成了 JupyterLab交互式开发功能，提供在线开发调试功能；同时也提供 SSH用于访问容器，可支持 VSCode和 Pycharm远程开发调试代码；另外 JupyterLab集成了markdown和 Tensorboard功能。
 
 
1)注意事项
 
 
•开发环境的工作目录/workspace是非易失的，用户创建的文件和文件夹在开发环境重启、停止之后，是不会丢失的。
 
 
•开发环境支持多容器工作模式，可支持分布式训练开发调试。
 
 
2)目录结构
 
 
| |-- workspace<br>|-- algorithm<br>|--我的算法集 A<br>|--公共算法集 B<br>|-- dataset<br>|--我的数据集 A<br>|--我的数据集 A（ ID）<br>|--公共数据集 B<br>|--公共数据集 B(ID)<br>|-- model|
| ---| 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_65.jpg?Expires=1758446096&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=%2BE3E4MDQtfTZxLrQP2YKViokdls%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
22/ 47
 
 
•注意：/workspace/persistent-model是模型训练输出模型结果、LOSS日志、评估结果的目录。
 
 
3)创建开发环境
 
 
•点击“+创建开发环境”，按照页面提示配置环境。
 
 
•开发环境参数说明
 
 
表 3-1开发环境参数说明
 
 
| 名称|必填项|说明|
| ---|---|---|
| 开发环境名称|是|最长 50位字符，不能包含/:*?”<’>|字符，且不能都为空格|
| 数据|否|可选“数据集”|
| 算法|否|可选“算法集”|
| 密码|否|建议设置复杂密码，也可以由系统自动生成|
| 镜像|是|根据实际场景，选择镜像|
| 规格|是|根据实际场景，选择容器规格|
| 使用时长|是|建议选择合适的使用时长，闲置时释放资源|
| 计算节点数量|是|可选择多个计算节点，支持分布式训练开发调试|
| 模型|否|可选择“模型集”，支持从某个模型继续训练| 
 
一
 
 
4)查看开发环境详情
 
 
•点击开发环境“名称”，进入开发环境详情页，可查看开发环境配置信息、容器监控、日志和事件。
 
 
Enflame-
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_66.jpg?Expires=1758446096&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=g0Z2Nvrysr8OTMczZZVvr4kTj%2B0%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
23/ 47
 
 
三                                                                                                                        test
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_67.jpg?Expires=1758446097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=094W7ArpDJXr%2BydjV7QfI3Oglz4%3D) 
 
图3-42开发环境详情
 
 
5)登录开发环境
 
 
•   SSH终端登录点击开发环境“名称”，进入开发环境详情页，查看 SSH登录信息。例如，使用终端工具，通过 ssh root@console.labcloud.uat.enflame.cn-p 30853远程登录到开发环境。
 
 
•   JupyterLab网页端登录点击开发环境“名称”，进入开发环境详情页，点击“进入”，即可进入交互式开发环境。创建 Notebook进行算法开发和调试，关于 Notebook的详细操作，请参见 https://jupyter-
 
 
notebook.readthedocs.io/en/stable/notebook.html。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_68.jpg?Expires=1758446097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=R9JJ%2BQtcxo29S2JIEaJw958XPvo%3D) 
 
图3-43多节点登录
 
 
•多节点登录创建多节点的开发环境时，详情页只显示第一个节点的登录信息。使用上述 SSH或 JupyterLab网页方式登录第一个节点之后，可以从第一个节点免密登录其它节点。JupyterLab可以单击 Terminal组件进入命令行，然后使用 SSH命令登录其他节点。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_69.jpg?Expires=1758446097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=GvXc8VlP%2F98MBFr41OpI8hDuu90%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
24/ 47
 
 
•其它节点名称保存在开发环境的/etc/volcano/worker.host文件里，如下图所示。
 
 
•   SSH登录其它节点
 
 
•如下图所示的例子，通过 ssh notebookv2-5755710f7b7b-1637033279-worker-0登录到其中一个 worker节点。
 
 
•   OpenMPI访问其它节点
 
 
•分布式训练时，需要先配置 hostfile文件，格式如下所示。每一行都是以 host slots=N的形式组织，slots=N表示在 host上启动 N个计算进程，一般情况下， slots等于 host上云燧 AI加速卡的个数。
 
 
localhost slots=8
 
 
notebookv2-5755710f7b7b-1637033279-worker-0 slots=8 notebookv2-
 
 
5755710f7b7b-1637033279-worker-1 slots=8 notebookv2-
 
 
5755710f7b7b-1637033279-worker-2 slots=8
 
 
•   hostfile可以通过下面的命令生成，slots根据实际容器申请的云燧 AI加速卡的个数来配置。
 
 
mkdir-p/etc/mpi&&\
 
 
echo"localhost slots=8">/etc/mpi/hostfile&&\
 
 
cat/etc/volcano/worker.host| sed"s/$/& slots=8/g">>/etc/mpi/ hostfile
 
 
•启动 OpenMP的时候指定 hostfile路径，如下所示。
 
 
mpirun--hostfile/etc/mpi/hostfile--allow-run-as-root-mca btl^ openib-np 32\
 
 
python3/workspace/algorithm/resnet50/train.py\
 
 
--data_dir=/workspace/dataset/imagenet\
 
 
--output_dir=/workspace/persistent-model\
 
 
--device=dtu--dataset=imagenet\
 
 
--dtype=bf16--epoch=50\
 
 
6)搜索开发环境
 
 
•如下图所以，搜索框内输入关键词+回车，快速定位开发环境。
 
 
三                                                                                              test6m
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_70.jpg?Expires=1758446097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=4jHTx5BI%2ByohqJnBj25VWs%2FopWQ%3D) 
 
图3-44搜索开发环境
 
 
7)停止开发环境
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_71.jpg?Expires=1758446097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=y8KwTj%2FlRTNa1BvrwahQZVt4F20%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
25/ 47
 
 
•在开发环境列表页，点击要停止的开发环境“名称”，进入开发环境详情页，点击“停止”。
 
 
8)启动开发环境
 
 
•在开发环境列表页，点击要启动的开发环境“名称”，进入开发环境详情页，点击“启动”。启动需要一定时间，当开发环境状态为 running，并且开发环境真正可用时，才能点击“进入”进入交互式开发环境。
 
 
9)复制开发环境
 
 
•在开发环境列表页，点击“复制”，根据需要修改开发环境配置，点击“确定”，创建新的开发环境。
 
 
10)保存镜像
 
 
•当开发环境状态为“running”时，点击“保存镜像”，编辑镜像名称、镜像版本、规格和和详细内容，点击保存，可以到“系统管理-镜像规格”到查看保存的镜像推送情况。
 
 
表 3-2保存镜像参数说明
 
 
| 名称|必填项|说明|
| ---|---|---|
| 镜像名称|是|最长 50位字符，只能输入 a-z,0-9,符号为-._/|
| 镜像版本|是|最长 50位字符，只能输入 a-z,A-Z, 0-9,符号为-._|
| 镜像规格|是|可选择多个镜像规格|
| 详细内容|否|最长 500位字符| 
 
3.5模型管理
 
 
3.5.1模型集
 
 
模型集分为“我的模型集”和“公共模型集”。
 
 
•我的模型集：个客户或者项目内共享，客户或者项目间互相隔离，支持读写权限。
 
 
•公共模型集：有客户或者项目共享，仅支持只读权限。
 
 
1)我的模型集
 
 
•每个模型集默认展示最新的 5个版本，如下图所示。点击某个版本记录，可进入文件管理功能，用户可以上传文件、文件夹，下载文件、文件夹，删除文件、文件夹。
 
 
•如下图所示，搜索框内输入关键词+回车，快速定位模型集。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_72.jpg?Expires=1758446097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=q%2Bj7X2Cjxk2Zu84ST11f13Mr9Fc%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
26/ 47
 
 
三                                                                                                                         test
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_73.jpg?Expires=1758446097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=QU2JlyznN58MtYcaeWBCQVYNGzA%3D) 
 
图3-45搜索模型集
 
 
•添加模型集：在模型集列表页，点击“+添加模型集”，输入模型名称，选择“应用场景”，“详细内容”可选。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_74.jpg?Expires=1758446097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=cw3PxlskJvyrMK7pR2FUFFzdUvU%3D) 
 
图3-46添加模型集
 
 
•编辑模型集：辑模型集，只能修改“详细内容”，其它都不能更改。
 
 
•删除模型集。注意：只有当模型集版本为空时，才能删除模型集，删除模型集时，请务必谨慎，目前没有强制检查模型集和其它组件的关联状态。
 
 
•模型集版本
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_75.jpg?Expires=1758446097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=mSTdQfodr1xpfMFJN4nFDiR8928%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
27/ 47
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_76.jpg?Expires=1758446097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=idGXgCk6aYzTDMKNKIZT%2FbI7IoM%3D) 
 
图3-47模型集版本
 
 
•模型可视化：点击模型集版本的“可视化”，选择目标模型文件，打开模型可视化功能，查看模型网络结构和参数,。
 
 
←返回     resnet50_v1.5-op8.onnx
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_77.jpg?Expires=1758446097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=LFnFZc5vwj1icv8DDAhMP1fgOdU%3D) 
 
图3-48模型可视化
 
 
2)公共模型集
 
 
预置的公共模型集，对所有客户或者项目可见，可在创建“开发环境”或“训练任务”时候选择使用，但仅支持“只读”挂载。
 
 
3.5.2批量推理
 
 
批量推理功能利用训练好的模型和推理代码，批量对数据集进行推理预测，输出结果。
 
 
1)注意事项
 
 
•批量任务的/workspace是易失的，批量推理跑完之后，/workspace临时目录会被清空。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_78.jpg?Expires=1758446098&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=B797U15FhDC6XIVpfzjkrU2aFRs%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
28/ 47
 
 
•用户可以将推理的结果输出到标准输出，通过日志页面可查看结果，并且可下载结果；或者将推理结果输出到数据集目录下保存。
 
 
2)目录结构
 
 
•训练任务的目录结构和开发环境是一致的，这样可以保证在开发环境跑通的算法代码，换成批量推理环境之后也能顺利跑通。
 
 
3)创建批量推理
 
 
•点击“+创建批量推理”，按照页面提示配置环境。
 
 
表 3-3批量推理参数说明
 
 
| 名称|必填项|说明|
| ---|---|---|
| 批量推理名称|是|最长 50位字符，不能包含/:*?”<’>|字符，且不能都为空格|
| 应用场景|是|不同数据类型的标注类型|
| 数据|否|可选“数据集”|
| 算法|否|可选“算法集”|
| 镜像|是|根据实际场景，选择镜像|
| 规格|是|根据实际场景，选择容器规格|
| 运行命令|是|批量推理运行命令|
| 模型|否|可选择“模型集”，支持从某个模型继续训练| 
 
4)运行命令说明
 
 
•批量推理运行命令(仅供格式参考)，通常情况下就是启动 python程序，例子如下。
 
 
| python3/workspace/algorithm/resnet50/inference.py\<br>--data_dir=/workspace/dataset/imagenet\<br>--device=dtu--dataset=imagenet\|
| ---| 
 
5)搜索批量推理
 
 
•如下图所以，搜索框内输入关键词+回车，快速定位批量推理。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_79.jpg?Expires=1758446098&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=HTKGSIqor2fsGXk%2B8LGU4c4BSNM%3D) 
 
图3-49搜索批量推理
 
 
6)启动批量推理
 
 
•在批量推理列表页，点击“启动”，即可启动一个实例。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_80.jpg?Expires=1758446098&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=FgPkoDpqxnssnleF1rxGUYPFF4E%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
29/ 47
 
 
7)复制批量推理
 
 
•在批量推理列表页，点击“复制”，根据需要修改批量推理配置，点击“确定”，创建新的批量推理。
 
 
8)查看批量推理详情
 
 
•在批量推理列表页，点击某个批量推理实例，进入批量推理详情页，可查看批量推理配置、日志、资源监控和事件记录。
 
 
三                                                                                                                                                         test6v
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_81.jpg?Expires=1758446098&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=PJ0WKOOPta3KVi8YaE%2FDwm5ekKg%3D) 
 
C
 
 
图3-50批量推理详情
 
 
9)停止批量推理
 
 
•在批量推理列表页，点击某个批量推理实例的“停止”，即可停止推理实例。
 
 
10)删除批量推理
 
 
•在批量推理列表页，点击某个批量推理实例的“删除”，即可删除批量推理实例。如需删除整个批量推理，确保批量推理实例已清空。
 
 
11)批量推理结果展示
 
 
•以 Resnet_50_inference为例，批量推理结果展示如下图所示。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_82.jpg?Expires=1758446098&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=WKjy8%2BTDEOOHW9aNyL2Tv2rLd8U%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
30/ 47
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_83.jpg?Expires=1758446098&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=Ut8Uo6aULAJYbLGcBb%2FT0PMOqL8%3D) 
 
图3-51批量推理结果
 
 
3.6训练管理
 
 
3.6.1训练任务
 
 
Contide
 
 
训练任务支持单机单卡、单机多卡和多机多卡三种场景，用户可查看训练任务执行记录，查看训练任务日志、资源监控和事件记录。
 
 
1)注意事项
 
 
•训练任务的/workspace是易失的，训练任务跑完之后，/workspace临时目录会被清空。
 
 
•如果训练任务需要保存模型结果，请在创建训练任务的时候选择保存模型，同时模型结果请保存到/workspace/persistent-model。
 
 
2)目录结构
 
 
•训练任务的目录结构和开发环境是一致的，这样可以保证在开发环境跑通的算法代码，换成训练任务环境之后也能顺利跑通。
 
 
3)创建训练任务
 
 
•点击“+创建训练任务”，按照页面提示配置环境。
 
 
表 3-4训练任务参数说明
 
 
| 名称|必填项|说明|
| ---|---|---|
| 训练任务名称|是|最少 50位字符，不能包含/:*?”<’>|字符，且不能都为空格|
| 数据|否|可选“数据集”|
| 算法|否|可选“算法集”|
| 镜像|是|根据实际场景，选择镜像|
| 规格|是|根据实际场景，选择容器规格| 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
31/ 47
 
 
训推一体化平台 TopsDL用户使用手册
 
 
| 运行命令|是|训练任务启动命令|
| ---|---|---|
| 保存模型|否|可选择“我的模型”，保存训练模型结果|
| 计算节点数量|是|可选择多个计算节点，支持分布式训练开发调试|
| 模型|否|可选择“模型集”，支持从某个模型继续训练| 
 
4)运行命令说明
 
 
•普通训练：通训练运行命令相对简单(仅供格式参考)，通常情况下就是启动 python程序，例子如下。
 
 
| python3/workspace/algorithm/resnet50/train.py\<br>--data_dir=/workspace/dataset/imagenet\<br>--output_dir=/workspace/persistent-model\<br>--device=dtu--dataset=imagenet\<br>--dtype=bf16--epoch=50\|
| ---| 
 
•分布式训练：布式训练仅支持基于 OpenMPI和 Horovod的架构，运行命令需指定 slots，并生成 hostfile，例子如下。
 
 
•注意：master节点也是可以参与训练的，所以把 localhost也加入 hostfile。
 
 
| mkdir-p/etc/mpi&&\|
| ---|
| echo"localhost slots=8">/etc/mpi/hostfile&&\|
| cat/etc/volcano/worker.host| sed"s/$/& slots=8/g">>/etc/mpi/ hostfile&&\|
| mpirun--hostfile/etc/mpi/hostfile--allow-run-as-root-mca btl^ openib-np 32\|
| python3/workspace/algorithm/resnet50/train.py\|
| --data_dir=/workspace/dataset/imagenet\|
| --output_dir=/workspace/persistent-model\|
| --device=dtu--dataset=imagenet\| 
 
5)搜索训练任务
 
 
•如下图所以，搜索框内输入关键词+回车，快速定位训练任务。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_84.jpg?Expires=1758446098&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=b4KYrzNjBnGytYdn9ZCLOcJhfEM%3D) 
 
共2条
 
 
图3-52搜索训练任务
 
 
6)启动训练任务
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_85.jpg?Expires=1758446098&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=ZQrYCarlrO7RXnr5hC2qE53hSoQ%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
32/ 47
 
 
•在训练任务列表页，点击“启动”，即可启动一个实例。
 
 
7)复制训练任务
 
 
•在训练任务列表页，点击“复制”，根据需要修改训练任务配置，点击“确定”，创建新的训练任务。
 
 
8)查看训练任务详情
 
 
•在训练任务列表页，点击某个训练任务实例，进入训练任务详情页，可查看训练任务配置、日志、 资源监控和事件记录。
 
 
三                                                                                                                         test6
 
 
←返回1
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_86.jpg?Expires=1758446098&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=SuUBUqDUFcXMTCUh4RUv%2FKGmB2k%3D) 
 
图3-53训练任务详情
 
 
9)停止训练任务
 
 
•在训练任务列表页，点击某个训练任务实例的“停止”，即可停止训练实例。
 
 
10)删除训练任务
 
 
•在训练任务列表页，点击某个训练任务实例的“删除”，即可删除训练实例。如需删除整个训练任务，确保训练任务实例已清空。
 
 
11)断点续训
 
 
•在训练任务列表页，点击处于 stopped, failed, completed的状态的训练任务实例操作栏中的“断点续训”，可以使实例继续训练。当 pending和 running状态时，断点续训按钮置灰，不可用。
 
 
12)打开 TensorBoard
 
 
•在训练任务列表页，点击某个训练任务实例的“创建 Tensorboard”，其状态会依次更新为“启动中”–>”启动 tensorboard”–>”打开 tensorboard”,点击“打开tensorboard”，下图页面会被打开。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_87.jpg?Expires=1758446099&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=n2UpzAcOIB3gm0oNUCPcbuZl3ns%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
33/ 47
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_88.jpg?Expires=1758446099&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=vzDNGlfj9ZrSPOl8t8vQI2POa6E%3D) 
 
图3-54训练任务详情
 
 
13)生成训练数据
 
 
entlo
 
 
•数据训练完成后，生成训练数据（当标注的数据个数少于 10个时，不会生成test.json文件）。
 
 
•图像检测
 
 
train.json文件为coco格式的训练数据集，test.json文件为coco格式的测试数据集。
 
 
•图像分割
 
 
trainval.json文件为coco格式的训练数据集，test.json文件为coco格式的测试数据集。
 
 
•图像分类
 
 
rain_label_file.txt文件是“数据,标签”格式的训练数据集，test_label_file.txt文件是“数据,标签”格式的测试数据集，classes.txt为标签文件。train目录中是 TFRecord格式的训练数据集, evaluate目录中是 TFRecord格式的测试数据集。
 
 
•图像关键点
 
 
train.json文件为训练数据集，test.json文件为测试数据。
 
 
•图像 OCR
 
 
train.json文件为coco格式的训练数据集，test.json文件为测试数据。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_89.jpg?Expires=1758446099&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=Zb%2F9h4aiynThQCJsp2QdyX%2FyeBU%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
34/ 47
 
 
3.7部署上线
 
 
3.7.1模型包
 
 
模型包管理在线服务容器镜像，用户可通过模型包构建在线服务的镜像，比如模型推理服务。
 
 
1)目录结构
 
 
•通过模型包构建的服务镜像，模型、数据、算法的目录结构如下。
 
 
| /<br>|-- workspace<br>|-- algorithm<br>|--我的算法集 A<br>|--公共算法集 B<br>|-- dataset<br>|--我的数据集 A<br>|--公共数据集 B|
| ---| 
 
2)添加模型包版本
 
 
•在模型包列表页，点击“+添加模型包”，输入“名称”，选择“应用场景”，“名称”和“应用场景”为必填，“详细内容”可选。然后点击新建模型包的“新增版本”，按照要求填写模型包参数。
 
 
表 3-5模型包参数说明
 
 
| 名称|必填项|说明|
| ---|---|---|
| 模型|是|如果是模型推理，需要选择模型，可添加多个模型，|
| 镜像|是|新构建的镜像基于此基础镜像|
| 推理代码|否|可选算法集|
| 端口|是|在线服务的请求端口，默认 8080|
| 运行参数|否|在线服务的启动命令，比如：python3-m http.server 8080，输入时务必以空格为分割，分别填写|
| 软件依赖|否|可输入python包，比如：pandas==1.4.0|
| 环境变量|否|环境变量以key:value的形式配置，最终会打入镜像|
| Docker Run命令|是|可运行一些复杂的 shell命令，配置容器镜像，比如：apt-get install-y xxx| 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_90.jpg?Expires=1758446099&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=xr%2BPdRcqE%2BcaI74HtUsG5oJ0020%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
35/ 47
 
 
添加模型包版本
 
 
×
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_91.jpg?Expires=1758446099&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=LjzasfUB%2FZPE%2BioegBxx9jeW%2B%2Bo%3D) 
 
图3-55添加模型包版本
 
 
3)搜索模型包
 
 
•如下图所以，搜索框内输入关键词+回车，快速定位模型包。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_92.jpg?Expires=1758446099&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=0aEfTQy7zlOJ3JmQQQz%2BCavDAps%3D) 
 
4)停止模型包
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_93.jpg?Expires=1758446099&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=eJCfs3aBRjAe6kkKyHHuG4xjYd8%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
36/ 47
 
 
•只有当模型包的状态为 pending或者 building的时候才能停止，点击模型包版本的“停止”即可。
 
 
5)构建模型包
 
 
•点击模型包版本的“构建”，即可开始构建模型包，模型包构建是一个批处理任务，当它的状态为 completed的时候，即表示模型包构建成功。
 
 
3.7.2在线服务
 
 
目前平台支持以 HTTP协议暴露在线服务，用户可选择模型包部署在线服务。支持灰度版本、流量切分以及一键切换版本。建议的在线服务发布的流程是：
 
 
•添加一个服务
 
 
•新增生产版本，运行一段时间
 
 
•业务需要新的版本，新增一个灰度版本，设置 10%的流量到灰度版本，后台验证灰度版本是否满足业务需求
 
 
•灰度版本满足业务需求，一键切换为生产版本，验证服务正常，可删除灰度版本以后的版本迭代都可以按照此过程操作，确保业务稳定。
 
 
1)添加服务
 
 
•在服务列表页，点击“+添加服务”，输入“名称”，“详细内容”可选，点击“确定”。每个服务最多支持一个生产版本和一个灰度版本，灰度版本可配置分流比例。
 
 
Entlame-Tech•新增生产版本：增生产版本，选择模型包、容器规格和副本数，可设置环境变量。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_94.jpg?Expires=1758446099&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=PBy0bQTBLv%2Fyr3QRqQGtcnH3DUs%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
37/ 47
 
 
添加服务版本×
 
 
*模型包
 
 
| 我的模型包|python simple server|版本|1|
| ---|---|---|---| 
 
*规格
 
 
| 选择|型号|基础配置|加速卡|IB|RoCE|
| ---|---|---|---|---|---|
| ○|i10x1|4/16Gi|enflame.com/dtu:1|禁用|禁用|
| ○|T10x1|4/16Gi|enflame.com/dtu:1|禁用|禁用|
| ◎|Dev Only|1/1Gi|无|禁用|禁用|
| ○|T10x8|16/32Gi|enflame.com/dtu:8|禁用|禁用|
| |T20x1|4/16Gi|enflame.com/dtu:1|禁用|禁用|
| ○|i20x1|4/16Gi|enflame.com/dtu:1|禁用|禁用| 
 
环境变量
 
 
| TEST:VALUE X|+添加环境变量|
| ---|---| 
 
副本
 
 
2
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_95.jpg?Expires=1758446099&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=%2F3xkFyCYRyuq%2FhNa7Lw9Pfnt0Gw%3D) 
 
图3-57新增生产版本
 
 
•新增灰度版本：在已有生产版本的基础上，可新增一个灰度版本，选择模型包、容器规格和副本数，可设置环境 变量，另外可设置流量切分比例，默认 10%。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_96.jpg?Expires=1758446099&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=YVJjMvrxvhos8O64aCI2xWQS5xQ%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
38/ 47
 
 
添加服务版本
 
 
×
 
 
*模型包
 
 
我的模型包       python simple server                    版本   2
 
 
*规格
 
 
| 选择|型号|基础配置|加速卡|IB|RoCE|
| ---|---|---|---|---|---|
| ○|i10x1|4/16Gi|enflame.com/dtu:1|禁用|禁用|
| ○|T10x1|4/16Gi|enflame.com/dtu:1|禁用|禁用|
| ◎|Dev Only|1/1Gi|无|禁用|禁用|
| ○|T10x8|16/32Gi|enflame.com/dtu:8|禁用|禁用|
| ○|T20x1|4/16Gi|enflame.com/dtu:1|禁用|禁用|
| ○|i20x1|4/16Gi|enflame.com/dtu:1|禁用|禁用| 
 
环境变量
 
 
TEST:VALUE×    +添加环境变量
 
 
副本
 
 
一        2        十
 
 
分流
 
 
确定      取消
 
 
图3-58新增灰度版本
 
 
2)搜索在线服务
 
 
•如下图所以，搜索框内输入关键词+回车，快速定位在线服务。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_97.jpg?Expires=1758446100&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=RSput5lc2bna5swh0sdnsEWo4sc%3D) 
 
三
 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
39/ 47
 
 
test
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_98.jpg?Expires=1758446100&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=vd7mFMKmK%2Fbl7dOi6UnocfBYpZQ%3D) 
 
图3-59搜索在线服务
 
 
3)一键切换
 
 
•当灰度版本验证通过之后，可点击“切换生产版本”，将灰度和生产版本对换。验证无误之后，可 将灰度版本删除，只保留最新的生产版本。
 
 
4)设置分流
 
 
•只有灰度版本上可设置分流比例，点击分流“修改”，调整比例，点击“提交”。
 
 
5)设置副本
 
 
•所有版本都可以调整服务副本，点击副本“修改”，调整副本数，点击“提交”。注意：只有当服务版本状态为** serving**的时候，才表示副本数满足实际设置，可正常提供服务调用。副本那一列展示类似“2/ 1”这样的组合，2表示期望的副本数，1表示目前可用的副本数。
 
 
3.8系统管理
 
 
3.8.1项目管理
 
 
心
 
 
提供项目管理功能，支持统一管理您的项目。当前客户或者项目内资源共享，客户或者项目间互相隔离。
 
 
•搜索项目：如下图所以，搜索框内输入关键词+回车，快速定位项目。
 
 
三                                                                                                  admin
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_99.jpg?Expires=1758446100&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=J16r0l1vAF9h6JzXo6ghrBFJt0o%3D) 
 
图3-60搜索项目
 
 
•添加项目：在项目管理列表页，点击“+添加项目”，输入项目“名称”，“存储”，点击“创建”；点击 “+添加资源配额”，并输入如 cpu：48， memory:128Gi,requests.enflame.com/dtu:4等，并点击“确认”；选择“用户”，点击“添加”。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_100.jpg?Expires=1758446100&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=AvFNd2FlF027FzuelddrcTJqoTk%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
40/ 47
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_101.jpg?Expires=1758446100&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=lpeLtd5nCzHs%2FRn5h%2FabjmskFfI%3D) 
 
图3-61添加项目
 
 
ideitial
 
 
•编辑项目：在项目管理列表页，点击“编辑”，编辑项目“名称”，“存储”，点击“更新”；编辑资源配额或点击“+添加资源配额”，并输入如 cpu：48，
 
 
Entlame-Tech C
 
 
memory:128Gi,requests.enflame.com/dtu:4等，并点击“更新”；添加或删除“用户”，点击“添加”或”删除“。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_102.jpg?Expires=1758446100&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=LT%2Flz56mv6ibuGKX0MUSJSQfXJo%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
41/ 47
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_103.jpg?Expires=1758446100&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=np%2B5GFwTmQ%2FltvEJkPbHiRgWjN0%3D) 
 
图3-62编辑项目
 
 
•删除项目：在项目管理列表页，点击某个项目的“删除”，即可删除项目。
 
 
3.8.2用户管理
 
 
提供用户管理功能，支持统一管理您的用户。用户管理分为“平台用户管理”和“数据用户管理”。
 
 
•平台用户管理：平台用户,支持除数据管理中心所有模块的使用,支持用户的管理,密码修改操作。据用户管理：据用户,支持数据管理的操作使用,可对数据标注用户进行管理,设置用户类型,
 
 
1)平台用户管理
 
 
•搜索用户：如下图所以，搜索框内输入关键词+回车，快速定位用户。
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_104.jpg?Expires=1758446100&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=Bm%2BgtRLPT72BlUb1iAKe3med%2BpU%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
42/ 47
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_105.jpg?Expires=1758446100&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=NixWnALaEEM6QHeHSXoUexBBK6Q%3D) 
 
图3-63搜索用户
 
 
•添加用户：在平台用户管理列表页，点击“+添加用户”，输入用户名称，密码，邮箱后，点击“确定”或“取消”。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_106.jpg?Expires=1758446100&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=Jl%2F5Oi07l9EJYx2LFouwACoZvh8%3D) 
 
图3-64添加用户
 
 
•编辑用户：用户管理列表页，点击“编辑”，编辑用户名称，密码或邮箱，点击“确定”或“取消”。
 
 
•删除用户：用户管理列表页，点击某个用户的“删除”，即可删除用户。
 
 
2)数据用户管理
 
 
•修改用户权限：在数字用户管理列表页，点击“修改权限”，修改用户名或权限，点击“确定”或“取消”。
 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
43/ 47
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_107.jpg?Expires=1758446101&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=zJQxw8QTR0HkSlIbR4EUIaP96Oc%3D) 
 
图3-65修改权限
 
 
3.8.3镜像管理
 
 
0
 
 
提供镜像管理功能，支持统一管理我的镜像集、公共镜像集。镜像分为“我的镜像集”和“公共 镜像集”。
 
 
•我的镜像集：前客户或者项目内共享，客户或者项目间互相隔离，支持读写权限。
 
 
•公共镜像集：有客户或者项目共享，仅支持只读权限。管理员页面显示增删改查的功能。
 
 
1)我的镜像集
 
 
•搜索镜像：如下图所以，搜索框内输入关键词+回车，快速定位镜像。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_108.jpg?Expires=1758446101&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=7EwcyWocemPRGrzvgP3H4bzb5Uc%3D) 
 
图3-66搜索镜像
 
 
•添加镜像：在镜像管理列表页，点击“+添加镜像”，输入镜像名称，选择“镜像规格”，“镜像版本”，可复选，添加“镜像描述”。
 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
44/ 47
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_109.jpg?Expires=1758446101&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=BdhcHCXPvaodekNHuhxpC0pBzQA%3D) 
 
图3-67添加镜像
 
 
表 3-6添加镜像参数说明
 
 
| 名称|必填项|说明|
| ---|---|---|
| 镜像名称|是|最长 50位字符，只能输入 a-z,0-9,符号为-._/|
| 镜像版本|是|最长 50位字符，只能输入 a-z,A-Z, 0-9,符号为-._|
| 镜像规格|是|可选择多个镜像规格|
| 镜像描述|否|最长 500位字符| 
 
•编辑镜像：镜像管理列表页，点击“编辑”，只可编辑镜像描述，镜像名称、镜像版本和镜像规格置灰，不可修改。
 
 
•删除镜像：镜像管理列表页，点击“删除”即可。
 
 
•推送镜像：在镜像管理列表页，点击“推送镜像”，按照图示步骤进行操作。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_110.jpg?Expires=1758446101&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=VH1%2Bvtj2Y5Jwmn5OnX8s4exKtvs%3D) 
 
图3-68推送镜像
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_111.jpg?Expires=1758446101&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=4x3ccBP9ahiSVGhKZkJVRuufHg4%3D) 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
45/ 47
 
 
2)公共镜像集
 
 
•搜索、添加、编辑、删除、推送镜像，请参考我的镜像集。
 
 
3.8.4资源全局
 
 
显示所有用户算力分配和使用情况。
 
 
三
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_112.jpg?Expires=1758446101&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=KG2WQHkxVILZ%2Bsdksn1n4g6ZQuQ%3D) 
 
图3-69资源全局
 
 
3.8.5算力规格
 
 
提供算力规格功能，支持统一管理算力规格。只有管理员账号可以使用这个功能，配置内容包括 DTU卡、CPU数量、内存数量。
 
 
•搜索：如下图所示，搜索框内输入关键词+回车，快速定位相应记录。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_113.jpg?Expires=1758446101&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=472kBmETb7w7g6JFcFKmjZez3mA%3D) 
 
•     共8条
 
 
图3-70搜索算力规格
 
 
•添加规格：在算力规格列表页，点击“+添加规格”，输入算力规格名称，选择“型号”，数量，选择是否开启 IB,ROCE，最后点击“确定”或“取消”。
 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
46/ 47
 
 
训推一体化平台 TopsDL用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_114.jpg?Expires=1758446101&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=%2FV1ZOQk0Qk%2BHW7RFW4HE73Wit0w%3D) 
 
图3-71添加规格
 
 
•编辑规格：算力规格列表页，点击“编辑”，可编辑算力规格名称，“型号”，数量，选择是否开启 IB,ROCE。
 
 
•删除规格：镜像管理列表页，点击“删除”即可。
 
 
3.9修改密码
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419008688082948096/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E5%8C%96%E5%B9%B3%E5%8F%B0TopsDL%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_115.jpg?Expires=1758446101&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=ei%2Bkm3g3dfzYm6EfQLnOsfFjWRI%3D) 
 
图3-72修改用户信息
 
 
点击右上角的下拉按钮，选择“修改密码”，弹窗如下图所示，除 admin用户外，其他用户（如普通用户，标注用户）都可以修改自己的密码和邮箱。修改密码成功后，系统返回到登录页面用 户使用修改后的密码重新登录。
 
 
文中菜单涉及“我的...”（譬如“我的数据集”）均以项目为单位，包含项目范围内不同用户的所有数据记录；“显示自己”开关以用户为单位，过滤本用户相关记录。
 
 
3.9.1系统中部分字段参数限制说明
 
 
表 3-7参数说明
 
 
| 名称|说明|
| ---|---|
| 创建名称|不能包含/:*?”<’>|字符，且不能为纯空格(字数限制 50)|
| 用户名|不能包含/:*?”<’>|字符，且不能为纯空格(字数限制 50)|
| 密码|不限制输入内容(字数限制 6-16)|
| 描述内容|不限制输入内容(字数限制 500)| 
 
版权所有©2023上海燧原科技有限公司保留所有权利
 
 
47/ 47
 
 
训推一体化平台 TopsDL用户使用手册
 
 
| 运行命令|不限制输入内容(字数限制 3000)|
| ---|---|
| 环境变量|不限制输入内容（字数限制 500）|
| 镜像名称|最长 50位字符，只能输入 a-z,0-9,符号为-._/|
| 镜像版本|最长 50位字符，只能输入 a-z,A-Z, 0-9,符号为-._|
| key value|不限制输入内容（字数限制 500）| 
 
