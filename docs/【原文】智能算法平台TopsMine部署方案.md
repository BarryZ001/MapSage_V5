![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_1.jpg?Expires=1758447007&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=tQ3tevXR0wlEnrx6Z4FUMmbmAU0%3D) 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_2.jpg?Expires=1758447007&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=2HBmAuV5HNsX6UUrY5B6Seaj%2FyU%3D) 
 
nfidential
 
 
智能算法平台 TopsMine部署方案
 
 
Entlame-Ted
 
 
V1.1
 
 
2022年 9月 5日
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_3.jpg?Expires=1758447007&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=VVOw0meOdRJb7MpVXv32UqF%2F%2BpM%3D) 
 
Enflame燧原科技
 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
2/ 9
 
 
目录
 
 
1前言.................................................................................................................. 3
 
 
1.1声明............................................................................................................................................... 3
 
 
2 TopsMine........................................................................................................... 4
 
 
2.1部署概要....................................................................................................................................... 4
 
 
2.1.1工程名称................................................................................................................................ 4
 
 
2.1.2部署环境................................................................................................................................ 4
 
 
2.1.3软件版本................................................................................................................................ 4
 
 
2.1.4部署地点................................................................................................................................ 4
 
 
2.1.5部署时间................................................................................................................................ 4
 
 
2.1.6部署调试人员........................................................................................................................ 4
 
 
2.2部署情况....................................................................................................................................... 4
 
 
2.2.1预检查.................................................................................................................................... 4
 
 
2.2.2部署步骤................................................................................................................................ 5
 
 
2.2.3调试步骤................................................................................................................................ 5
 
 
2.2.4部署结果................................................................................................................................ 8
 
 
今
 
 
版本历史
 
 
| 文档版本|文档日期|文档说明|
| ---|---|---|
| V 1.1|2022/9/5|第一次正式发布| 
 
Entlame-
 
 
智能算法平台 TopsMine部署方案
 
 
1前言
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_4.jpg?Expires=1758447007&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=iYt2CJbVECItSIvm2Wr3sh3GEUM%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
3/ 9
 
 
1前言
 
 
Enflame燧原科技
 
 
1.1声明
 
 
本文档提供的信息属于上海燧原科技有限公司和/或其子公司所有，且燧原科技保留不经通知随时对本文档信息或对任何产品和服务做出修改的权利。本文档所含信息和本文档所引用燧原科技其他信息均“按原样”提供。燧原科技不担保信息、文本、图案、链接或本文档内所含其他项目的准确性或完整性。燧原科技不对本文档所述产品的可销售性、所有权、不侵犯知识产权、准确性、完整性、稳定性或特定用途适用性做任何暗示担保、保证。燧原科技可不经通知随时对本文档或本文档所述产品做出更改，但不承诺更新本文档。
 
 
在任何情况下，燧原科技不对因使用或无法使用本文档而导致的任何损害（包括但不限于利润损失、业务中断和信息损失等损害）承担责任。燧原科技不承担因应用或使用本文档所述任何产品或服务而产生的任何责任。
 
 
本文档所列的规格参数、性能数据和等级需使用特定芯片或计算机系统或组件来测量。经该等测试，本文档所示结果反映了燧原科技产品的大概性能。系统配置及软硬件版本、环境变量等的任何不同会影响实际性能，产品实际效果与文档描述存在差异的，均属正常现象。燧原科技不担保测试每种产品的所有参数。客户自行承担对产品适合并适用于客户计划的应用以及对应用程序进行必要测试的责任。客户产品设计的脆弱性会影响燧原科技产品的质量和可靠性并导致超出本文档范围的额外或不同的情况和/或要求。
 
 
燧原科技和燧原科技的标志是上海燧原科技有限公司申请和/或注册的商标。本文档并未明示或暗示地授予客户任何专利、版权、商标、集成电路布图设计、商业秘密或任何其他燧原科技知识产权的权利或许可。
 
 
本文档为版权所有并受全世界版权法律和条约条款的保护。未经燧原科技的事先书面许可，任何人不可以任何方式复制、修改、出版、上传、发布、传输或分发本文档。为免疑义，除了允许客户按照本文档要求使用文档相关信息外，燧原科技不授予其他任何明示或暗示的权利或许可。
 
 
Entlame燧原科技对本文档享有最终解释权。
 
 
智能算法平台 TopsMine部署方案
 
 
2 TopsMine
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_5.jpg?Expires=1758447007&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=04WfMh634qU8JgrpKqG2Vz9gbn0%3D) 
 
2 TopsMine
 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
4/ 9
 
 
Enflame燧原科技
 
 
2.1部署概要
 
 
2.1.1工程名称
 
 
国产化 AI视频基础设施平台项目。
 
 
2.1.2部署环境
 
 
表 2-1部署环境
 
 
| 主机名|IP|CPU(core)|内存(G)|系统盘|数据盘挂载点和大小|用途<br>说明|
| ---|---|---|---|---|---|---|
| Master01|172.17.180.24|16|64|200G|/var/ lib/ docker500G|k8s master|
| Master02|172.17.180.25|16|64|200G|/var/ lib/ docker500G|k8s master|
| Master03|172.17.180.26|16|64|200G|/var/ lib/ docker500G|k8s master|
| Node* 78||96|512|200G|1T|k8s worknode|
| Node*100||128|512|200G|1T|k8s worknode| 
 
2.1.3软件版本
 
 
v1.1
 
 
2.1.4部署地点
 
 
e-lec
 
 
智算中心 301机房
 
 
2.1.5部署时间
 
 
2022-10-12
 
 
2.1.6部署调试人员
 
 
吴建强
 
 
2.2部署情况
 
 
2.2.1预检查
 
 
1.检查 ubuntu18.04.6操作系统是否安装正确，经检查正确；
 
 
2.检查 hostname配置，ip配置是否正确，经检查正确；
 
 
3.检查系统分区是否正确，经检查正确；
 
 
4.检查是否将内核版本升级到 kernel5.4.0.113，经检查已升级到该版；
 
 
5.检查内核版本是否已经锁定，经检查内核版本已锁定；
 
 
智能算法平台 TopsMine部署方案
 
 
2 TopsMine
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_6.jpg?Expires=1758447008&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=bDH4OZROVAxPJc6ozWRRBQRBoF4%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
5/ 9
 
 
6.检查防火墙是否关闭，经检查防火墙已关闭；
 
 
7.检查网络模式设置是否正确，经检查正确；
 
 
8.检查网络速率是否正常，经检查正常
 
 
9.安装加速卡驱动 2.0.122版本，完成后，检查无误；
 
 
10.安装推理服务器解码卡驱动，完成后，检查无误；
 
 
11.经检查，所有预装配置都正确，板卡相关信息正常，板卡驱动安及解码卡驱动安装正确，板卡压测通过
 
 
2.2.2部署步骤
 
 
1.登陆 172.17.180.27，ssh root@172.17.180.27，密码 enflame@123
 
 
2.进入/home/deploy-doc/topsstack目录,执行安装命令
 
 
3.修改 topsmine部署文件，运行 docker run-v/root/.kube:/root/.kube-v●`pwd`/config.yaml:/topsstation-installer/config.yaml  harbor.proxima.com/topsstack/topsstation-installer:release-v1.0.0，等待部署完成
 
 
4.浏览器打开：http://topsmine.proxima.com
 
 
1)输入用户名/密码：test-admin/test-admin123
 
 
2)查看页面（表示安装成功） idene
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_7.jpg?Expires=1758447008&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=1Vp3DHQM%2FmVGP5P9XxaCw%2BZTaHk%3D) 
 
图 2-1部署成功页面
 
 
5.查看重要组件运行状态
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_8.jpg?Expires=1758447008&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=a30Xih%2Betedz0UmLJLAsCU6e08o%3D) 
 
LA..
 
 
图 2-2组件状态
 
 
2.2.3调试步骤
 
 
1.预检查
 
 
智能算法平台 TopsMine部署方案
 
 
2 TopsMine
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_9.jpg?Expires=1758447008&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=NalKvnZM4bltJmVKV6smeaRl5AE%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
6/ 9
 
 
172.17.180.102(root)
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_10.jpg?Expires=1758447008&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=ecWNiGx87pEjXd6K2DATs8BhJA4%3D) 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_11.jpg?Expires=1758447008&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=BAfprUH8pMs7tcmCcQlH5W13MH4%3D) 
 
图 2-3预检查
 
 
智能算法平台 TopsMine部署方案
 
 
2 TopsMine
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_12.jpg?Expires=1758447008&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=%2FliUqkoKIOc0Xlk6cmo6qSkCNwI%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
7/ 9
 
 
2.打开网页，以如下方式成功登录
 
 
地址：http://topsmine.proxima.com
 
 
输入用户名/密码：test-admin/test-admin123
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_13.jpg?Expires=1758447008&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=uw8R%2BHzyXKyREZoGPnYp0g1Z5Mc%3D) 
 
图 2-3登录成功页面
 
 
3.创建算法应用
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_14.jpg?Expires=1758447008&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=dGaQCK3%2FcK1hm%2FO6GuX0w0vStW4%3D) 
 
图 2-5创建算法应用
 
 
4.模型与镜像上传
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_15.jpg?Expires=1758447008&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=5I5TyzH%2BiaV4JNjgAzOgjY%2Fmp84%3D) 
 
图 2-6模型与镜像上传
 
 
5.算法应用部署
 
 
2 TopsMine
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_16.jpg?Expires=1758447009&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=g7bIc46f5thhPG7qsCsSueFRqi8%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
8/ 9
 
 
智能算法平台 TopsMine部署方案
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_17.jpg?Expires=1758447009&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=EOIvhuiT%2BINlLkRoH%2FcJMpX5HNI%3D) 
 
图 2-7算法应用部署
 
 
6.算法应用推理
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_18.jpg?Expires=1758447009&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=bBdIUNP8RgfLrrpgJtNRVef6RFI%3D) 
 
图 2-8算法应用推理
 
 
7.查看部署日志及资源信息
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_19.jpg?Expires=1758447009&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=tihHjm4HIEGfnwn4v5AvZ0PNGeI%3D) 
 
图 2-9查看部署日志及资源信息
 
 
2.2.4部署结果
 
 
基于部署方案和功能设计，各项指标满足方案要求，完成智能算法平台 TopsMine部署。
 
 
智能算法平台 TopsMine部署方案
 
 
2 TopsMine
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_20.jpg?Expires=1758447009&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=%2B%2Fi3ib8lLe83MyGEgsh0%2Bp8HzTk%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
9/ 9
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741997727744/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%B9%B3%E5%8F%B0TopsMine%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88_21.jpg?Expires=1758447009&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=58suFIwqJ9fgOTt%2BeIDLjBBKkJU%3D) 
 
图 2-10部署结果
 
 
