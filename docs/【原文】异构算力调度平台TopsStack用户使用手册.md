![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_1.jpg?Expires=1758447096&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=ajefbC6%2FQP%2FoR9gdVcwrKdq1ezs%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
1/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_2.jpg?Expires=1758447096&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=ykycd5xv9QIXlMo2VKslyov1IoM%3D) 
 
ntidential
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
Entlame-Te
 
 
V 1.1
 
 
2022年 9月
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_3.jpg?Expires=1758447096&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=%2FjFAG%2Fa0zlxwu9RF7KjFNSSae0s%3D) 
 
Enflame燧原科技
 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
2/ 57
 
 
目录
 
 
1前言.................................................................................................................. 4
 
 
1.1声明............................................................................................................................................... 4
 
 
1.2版本历史....................................................................................................................................... 4
 
 
1.3词汇表........................................................................................................................................... 4
 
 
2产品概述.......................................................................................................... 5
 
 
2.1产品简介....................................................................................................................................... 5
 
 
2.2架构图........................................................................................................................................... 5
 
 
3功能介绍.......................................................................................................... 6
 
 
3.1多租户管理................................................................................................................................... 6
 
 
3.2系统登陆....................................................................................................................................... 6
 
 
3.2.1创建账号................................................................................................................................ 6
 
 
3.2.2创建企业空间........................................................................................................................ 7
 
 
3.2.3创建项目................................................................................................................................ 9
 
 
3.3存储管理..................................................................................................................................... 10
 
 
3.3.1创建存储类型...................................................................................................................... 10
 
 
3.3.2创建存储卷.......................................................................................................................... 12
 
 
3.4配置中心..................................................................................................................................... 13
 
 
3.4.1创建保密字典...................................................................................................................... 13
 
 
3.4.2创建配置字典...................................................................................................................... 14
 
 
3.4.3服务账户.............................................................................................................................. 16
 
 
3.5应用............................................................................................................................................. 17
 
 
3.5.1应用管理.............................................................................................................................. 17
 
 
3.6工作负载..................................................................................................................................... 20
 
 
3.6.1创建部署.............................................................................................................................. 20
 
 
3.6.2有状态副本集...................................................................................................................... 22
 
 
3.6.3守护进程集.......................................................................................................................... 24
 
 
3.7任务............................................................................................................................................. 26
 
 
3.7.1任务...................................................................................................................................... 26
 
 
3.7.2定时任务.............................................................................................................................. 30
 
 
3.8网络与服务................................................................................................................................. 32
 
 
3.8.1服务管理.............................................................................................................................. 32
 
 
3.8.2灰度发布.............................................................................................................................. 35
 
 
3.8.3应用路由.............................................................................................................................. 42
 
 
3.9监控告警..................................................................................................................................... 44
 
 
3.9.1通知管理.............................................................................................................................. 44
 
 
3.9.2告警策略.............................................................................................................................. 45
 
 
3.9.3告警消息.............................................................................................................................. 47
 
 
3.10集群管理................................................................................................................................... 48
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_4.jpg?Expires=1758447096&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=QnwGSFajVlXOko6tJerDFtK5c3I%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
3/ 57
 
 
3.10.1添加集群............................................................................................................................ 48
 
 
3.10.2集群节点............................................................................................................................ 49
 
 
3.10.3网关设置............................................................................................................................ 54
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_5.jpg?Expires=1758447096&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=1YgRR0Y3o4dchnYwkpJoeEH9H%2Bg%3D) 
 
Enflame燧原科技
 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
4/ 57
 
 
1前言
 
 
1.1声明
 
 
本文档提供的信息属于上海燧原科技有限公司和/或其子公司（以下统称“燧原科技”）所有，且燧原科技保留不经通知随时对本文档信息或对任何产品和服务做出修改的权利。本文档所含信息和本文档所引用燧原科技其他信息均“按原样”提供。燧原科技不担保信息、文本、图案、链接或本文档内所含其他项目的准确性或完整性。燧原科技不对本文档所述产品的可销售性、所有权、不侵犯知识产权、准确性、完整性、稳定性或特定用途适用性做任何暗示担保、保证。燧原科技可不经通知随时对本文档或本文档所述产品做出更改，但不承诺更新本文档。
 
 
在任何情况下，燧原科技不对因使用或无法使用本文档而导致的任何损害（包括但不限于利润损失、业务中断和信息损失等损害）承担责任。燧原科技不承担因应用或使用本文档所述任何产品或服务而产生的任何责任。
 
 
本文档所列的规格参数、性能数据和等级需使用特定芯片或计算机系统或组件来测量。经该等测试，本文档所示结果反映了燧原科技产品的大概性能。系统配置及软硬件版本、环境变量等的任何不同会影响实际性能，产品实际效果与文档描述存在差异的，均属正常现象。燧原科技不担保测试每种产品的所有参数。客户自行承担对产品适合并适用于客户计划的应用以及对应用程序进行必要测试的责任。客户产品设计的脆弱性会影响燧原科技产品的质量和可靠性并导致超出本文档范围的额外或不同的情况和/或要求。
 
 
燧原科技和燧原科技的标志是上海燧原科技有限公司申请和/或注册的商标。本文档并未明示或暗示地授予客户任何专利、版权、商标、集成电路布图设计、商业秘密或任何其他燧原科技知识产权的权利或许可。
 
 
本文档为版权所有并受全世界版权法律和条约条款的保护。未经燧原科技的事先书面许可，任何人不可以任何方式复制、修改、出版、上传、发布、传输或分发本文档。为免疑义，除了允许客户按照本文档要求使用文档相关信息外，燧原科技不授予其他任何明示或暗示的权利或许可。
 
 
燧原科技对本文档享有最终解释权。
 
 
1.2版本历史
 
 
表1-1版本历史
 
 
| 文档版本|文档日期|文档说明|
| ---|---|---|
| V1.1|2022年9月|定稿| 
 
1.3词汇表
 
 
表1-2词汇表
 
 
| 术语|描述|描述|
| ---|---|---|
| TopsStack|燧原科技异构算力调度平台|燧原科技异构算力调度平台| 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_6.jpg?Expires=1758447096&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=Xpj5n%2B29DqV64jRDX%2FmaXTkcRkU%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
5/ 57
 
 
2产品概述
 
 
Enflame燧原科技
 
 
2.1产品简介
 
 
燧原科技异构算力调度平台 TopsStack是在 Kubernetes之上构建的以应用为中心的企业级分布式容器平台，提供简单易用的操作界面以及向导式操作方式，在降低用户使用容器调度平台学习成本的同时，极大减轻开发、测试、运维的日常工作的复杂度，旨在解决 Kubernetes本身存在的存储、网络、安全和易用性等痛点。除此之外，平台已经整合并优化了多个适用于容器场景的功能模块，以完整的解决方案帮助企业轻松应对敏捷开发与自动化运维、DevOps、微服务治理、灰度发布、多租户管理、工作负载和集群管理、监控告警、日志查询与收集、服务与网络、应用商店、镜像构建与镜像仓库管理和存储管理等多种业务场景。后续版本还将提供和支持多集群管理、大数据、人工智能等更为复杂的业务场景。
 
 
2.2架构图
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_7.jpg?Expires=1758447096&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=zrhdv9N2FLVoCDTTdTuGi%2B5NXhI%3D) 
 
图2-1 TopsStack架构图
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_8.jpg?Expires=1758447097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=UZyvZTP3j9f4eExWKWikgLsTUmo%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
6/ 57
 
 
3功能介绍                     Enflame燧原科技
 
 
燧原科技异构算力调度平台 TopsStack作为开源的企业级全栈化容器平台，为用户提供了一个具备极致体验的 Web控制台，让您能够像使用任何其他互联网产品一样，快速上手各项功能与服务。异构算力调度平台 TopsStack目前提供了工作负载管理、微服务治理、DevOps工程、多租户管理、多维度监控、日志查询与收集、告警通知、服务与网络、应用管理、基础设施管理、镜像管理、应用配置密钥管理等功能模块，开发了适用于物理机部署 Kubernetes的负载均衡器插件 Porter，并支持对接多种开源的存储与网络方案，支持高性能的商业存储与网络服务。
 
 
3.1多租户管理
 
 
平台的资源一共有三个层级，包括集群(Cluster)、企业空间(Workspace)、项目(Project)和 DevOps Project(DevOps工程)，一个集群中可以创建多个企业空间，而每个企业空间，可以创建多个项目和 DevOps工程，而集群、企业空间、项目和 DevOps工程中，默认有多个不同的内置角色。
 
 
3.2系统登陆
 
 
用户可以使用 chrome、IE浏览器，输入异构算力调度平台 TopsStack即可进入登入页面。初始默认用户名为 admin，初始密码为 P@88w0rd。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_9.jpg?Expires=1758447097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=2bdsptO5DVjLoFDIQReXKCK4PTo%3D) 
 
图3-1系统登陆
 
 
3.2.1创建账号
 
 
平台中的 cluster-admin角色可以为其他用户创建账号并分配平台角色，平台内置了集群层级的以下三个常用的角色，同时支持自定义新的角色。
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
7/ 57
 
 
表3-1常用角色
 
 
| 内置角色|描述|
| ---|---|
| workspaces-manager|集群中企业空间管理员，仅可创建、删除企业空间，维护企业空间中的成员列<br>表。|
| platform-admin|管理 TopsStack平台上的所有资源。|
| platform-regular|被邀请加入企业空间之前无法访问任何资源。| 
 
本示例首先新建一个账号，为该账号授予管理员的权限。
 
 
⚫点击工作台 →用户管理 →用户，点击创建。
 
 
⚫填写用户信息，并将平台角色设置为 platform-admin。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_10.jpg?Expires=1758447097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=yD8f7Q4R4OqEZsb9puU6nkvS3Lg%3D) 
 
图3-2添加用户
 
 
⚫点击确定，用户创建成功。
 
 
3.2.2创建企业空间
 
 
⚫点击工作台 →企业空间管理，可以看到当前的企业空间列表，点击创建。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_11.jpg?Expires=1758447097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=wSFJmwBeg8t18sieI2mVJEq2chM%3D) 
 
图3-3企业空间列表
 
 
⚫填写企业空间的基本信息，并选择指定用户作为该企业空间的管理员
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_12.jpg?Expires=1758447097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=xlgzDwCKrLzBGpeeEVaMsm4rk%2B0%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
8/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_13.jpg?Expires=1758447097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=YbqbqKoAGBxFHxmP2%2Fgr0JxbW%2Fs%3D) 
 
图3-4填写企业空间信息
 
 
⚫点击下一步，并选择企业空间需要使用的集群
 
 
S
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_14.jpg?Expires=1758447097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=uDATmcdh6j50hg5jIULku8TnOXM%3D) 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_15.jpg?Expires=1758447097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=9ajwalCW3slVkK9yk1razLrALqU%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
9/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_16.jpg?Expires=1758447097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=FZSrlw%2BgQ9GzRyt3N2hb99n33Gs%3D) 
 
图3-5企业空间集群设置
 
 
⚫点击创建即可。
 
 
3.2.3创建项目
 
 
⚫点击工作台 →企业空间管理，点击对应的企业空间
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_17.jpg?Expires=1758447097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=Jz1PgQUrLWOL0S9RtORdxhjP550%3D) 
 
图3-6企业空间列表
 
 
⚫查看菜单栏中的项目，点击创建
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_18.jpg?Expires=1758447097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=AkJGXTi8vojqxpZXeaVdx3hsIu0%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
10/ 57
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_19.jpg?Expires=1758447097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=rGnt%2BdzwqDtJdfn0%2FlZC30jXUSM%3D) 
 
图3-7创建项目
 
 
⚫填写项目的名称等信息，点击确定即完成创建
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_20.jpg?Expires=1758447097&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=ClfF5iloWBQS1wasqcxhNeypCwg%3D) 
 
图3-8填写项目信息
 
 
3.3存储管理
 
 
飞
 
 
存储是为异构算力调度平台 TopsStack的容器运行的工作负载(Pod)提供存储的组件，支持多种类型的存储，并且同一个工作负载中可以挂载任意数量的存储卷。
 
 
3.3.1创建存储类型
 
 
存储类型(StorageClass)是由集群管理员配置存储服务端的参数，并按类型提供存储给集群用户使用。通常情况下创建存储卷之前需要先创建存储类型，目前支持的存储类型如GlusterFS、Ceph RBD、NFS、Local Volume等。需要注意的是，当系统中存在多种存储类型时，只能设定一种为默认的存储类型。
 
 
⚫选择工作台 →集群管理-> 选择集群->存储 →存储类型，进入存储类型列表页面。作为集群管理员，可以查看当前集群下所有的存储类型和详细信息
 
 
⚫点击创建，填写基本信息
 
 
图3.11填写存储类型信息
 
 
⚫点击下一步，选择存储系统
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_21.jpg?Expires=1758447098&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=qXpyJtPGWzcQ5EAa58eg9PVeQ2A%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
11/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_22.jpg?Expires=1758447098&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=g4NJi91asprC%2FBhd1EtqV0MBzcU%3D) 
 
图3-9选择存储类型
 
 
⚫填写存储类型设置，点击创建后完成存储类型创建
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_23.jpg?Expires=1758447098&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=ZfvGgRt4Otls4ofv0dOFuduKUsE%3D) 
 
图3-10存储类型设置
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_24.jpg?Expires=1758447098&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=dR0g91YdGmAuz8p0L71uQVmTJ0g%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
12/ 57
 
 
3.3.2创建存储卷
 
 
存储卷，在异构算力调度平台 TopsStack中一般是指基于 PVC的持久化存储卷，具有单个磁盘的功能，供用户创建的工作负载使用，是将工作负载数据持久化的一种资源对象。
 
 
⚫在已创建好的项目下选择存储 →存储卷，点击创建
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_25.jpg?Expires=1758447098&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=Rx5wvGYcA0D8R00VSLUczsi4OKk%3D) 
 
⚫填写存储卷基本信息
 
 
sde
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_26.jpg?Expires=1758447098&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=w1CPJF7Edow4v0P%2F4m37m4fAiVs%3D) 
 
图3-12填写存储卷信息
 
 
⚫点击下一步，选择已创建好的存储类型和访问模式，填写存储卷容量
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_27.jpg?Expires=1758447098&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=Y5F5gkEp70q4X8EnxxC2LUejXww%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
13/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_28.jpg?Expires=1758447098&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=INuQW91EcRypvhoyJpaIDN4RLMg%3D) 
 
图3-13存储卷设置
 
 
⚫点击创建，即可完成存储卷的创建
 
 
3.4配置中心
 
 
3.4.1创建保密字典
 
 
sh Contio
 
 
保密字典(Secret)可用于存储和管理密码、OAuth令牌和 SSH保密字典等敏感信息。容器组可以通过三种方式使用保密字典：
 
 
⚫作为挂载到容器组中容器化应用上的卷中的文件。
 
 
⚫作为容器组中容器使用的环境变量。
 
 
⚫作为 kubelet为容器组拉取镜像时的镜像仓库凭证。
 
 
⚫在已创建好的项目下选择配置 →保密字典，点击创建
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_29.jpg?Expires=1758447098&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=CzKYGIgAlTRb8a%2FyoTtdfEindu4%3D) 
 
图3-14保密字典
 
 
⚫配置基本信息，然后点击下一步
 
 
⚫设置保密字典，在数据设置选项卡，从类型下拉列表中选择保密字典类型
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
14/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_30.jpg?Expires=1758447098&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=BqnCW%2Fg0odRjrW%2Bl98GHInbvNSQ%3D) 
 
图3-15保密字典数据设置
 
 
⚫默认：对应 Kubernetes的 Opaque保密字典类型，同时也是 Kubernetes的默认保密字典类型。您可以用此类型保密字典创建任意自定义数据。点击添加数据为其添加键值对。
 
 
⚫ TLS信息：对应 Kubernetes的 kubernetes.io/tls保密字典类型，用于存储证书及其相关保密字典。这类数据通常用于 TLS场景，例如提供给应用路由(Ingress)资源用于终结 TLS链接。使用此类型的保密字典时，您必须为其指定凭证和私钥，分别对应 YAML文件中的 tls.crt和 tls.key字段。
 
 
⚫镜像服务信息：对应 Kubernetes的 kubernetes.io/dockerconfigjson保密字典类型，用于存储访问 Docker镜像仓库所需的凭证。有关更多信息，请参阅镜像仓库。
 
 
⚫用户名和密码：对应 Kubernetes的 kubernetes.io/basic-auth保密字典类型，用于存储基本身份认证所需的凭证。使用此类型的保密字典时，您必须为其指定用户名和密码，分别对应 YAML文件中的 username和 password字段。
 
 
⚫点击添加数据，填写键值对信息，例如 Key：MYSQL_ROOT_PASSWORD，Value：123456
 
 
⚫点击创建，即可完成保密字典创建
 
 
3.4.2创建配置字典
 
 
Kubernetes配置字典（ConfigMap）以键值对的形式存储配置数据。配置字典资源可用于向容器组中注入配置数据。配置字典对象中存储的数据可以被 ConfigMap类型的卷引用，并由容器组中运行的容器化应用使用。配置字典通常用于以下场景：
 
 
⚫设置环境变量的值。
 
 
⚫设置容器中的命令参数。
 
 
⚫在卷中创建配置文件。
 
 
⚫在已创建好的项目下选择配置 →配置字典，点击创建
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_31.jpg?Expires=1758447098&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=XwaS1mqhgH40owpTI11WqE2Xj%2Bw%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
15/ 57
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_32.jpg?Expires=1758447099&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=bMyU8J6taVhg4WOLRcGlk8SyO1Q%3D) 
 
图3-16配置字典
 
 
⚫设置配置字典的名称，点击下一步
 
 
XO
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_33.jpg?Expires=1758447099&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=T4uxK7LDx4VU%2BqFhQGHgG7j3Zek%3D) 
 
图3-17创建配置字典
 
 
⚫点击添加数据，输入键值对
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_34.jpg?Expires=1758447099&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=ua%2FBKL3C8QS6jLQabtxXcTrPjFc%3D) 
 
16/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_35.jpg?Expires=1758447099&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=jfkNK9gadPSMWVvIq9qbJRc%2Bhaw%3D) 
 
图3-18配置字典数据设置
 
 
⚫点击对话框右下角的 √以保存配置。您可以再次点击添加数据继续配置更多键值对
 
 
⚫点击创建以生成配置字典
 
 
3.4.3服务账户
 
 
服务账户为 Pod中运行的进程提供了标识。当用户访问集群时，API服务器将用户认证为特定的用户帐户。当这些进程与 API服务器联系时，Pod里容器的进程将被验证为特定的服务帐户。
 
 
⚫在已创建好的项目下选择配置 →服务账户，点击创建
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_36.jpg?Expires=1758447099&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=jPiSZ5foTQWKYOEKA2HnBL9%2F%2ByM%3D) 
 
图3-19服务账户
 
 
⚫在显示的创建服务账户对话框中，您可以设置以下参数：
 
 
⚫名称：（必填项）：服务帐户的唯一标识符。
 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_37.jpg?Expires=1758447099&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=zqckTI8lzzWSauQWF52gXDrF%2BXc%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
17/ 57
 
 
⚫别名：服务帐户的别名，以帮助你更好地识别服务帐户。
 
 
⚫简介：服务帐户简介。
 
 
⚫项目角色：从服务帐户的下拉列表中选择一个项目角色。在一个项目中，不同的项目角色有不同的权限。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_38.jpg?Expires=1758447099&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=bwsUQI3b1Vs0x6Qv1tIcftkSwZk%3D) 
 
图3-20创建服务账户
 
 
⚫完成参数设置后，单击创建。
 
 
3.5应用
 
 
应用通常是一个独立完整的业务功能，比如一个 bookinfo的书城网站就是一个应用，一个应用由多个服务组件组成，对于微服务而言每个组件都可以独立于其他组件创建、启动、运行和治理的。一个应用组件中又可以有一个或多个组件版本，自制应用允许用户选择已有服务或者新建服务组件来构建应用。
 
 
3.5.1应用管理
 
 
⚫在已创建的项目下选择应用负载 →应用，点击创建
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_39.jpg?Expires=1758447099&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=FH2bDJyPe9y8mo0nK7WsDwvKfzc%3D) 
 
图3-21应用管理
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_40.jpg?Expires=1758447099&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=7VT%2FFSzv4JU4v5wgaYqrHZX2Pt0%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
18/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_41.jpg?Expires=1758447100&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=4Fgsjcy%2FzNH5g303dUSMiA7ta%2F0%3D) 
 
图3-22创建自制应用
 
 
⚫点击创建服务 →无状态服务，填写基本信息
 
 
AC
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_42.jpg?Expires=1758447100&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=uiM2yWfGgroguxgZZdRMJ%2BCq2Ko%3D) 
 
图3-23创建无状态服务
 
 
⚫点击添加容器，填写镜像名称，使用资源，应用端口信息
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_43.jpg?Expires=1758447100&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=MSZtDgTejQl7ceHsLdZaVgFQnHk%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
19/ 57
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_44.jpg?Expires=1758447100&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=4ZFcmriZHCwCAHAsq8o80X%2BIKcE%3D) 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_45.jpg?Expires=1758447100&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=OgxN1JSX4kQcC%2BVEMj0XRF8dgKM%3D) 
 
图3-24容器组设置
 
 
⚫点击下一步，可以设置挂载存储卷
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_46.jpg?Expires=1758447100&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=v7mDAHT4tqe7zmmQj1Th0hdenrQ%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
20/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_47.jpg?Expires=1758447100&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=hBkzHzYAIkrEZzTUZBsGTmxcfPo%3D) 
 
图3-25存储卷设置
 
 
⚫点击创建，即可完成应用的创建
 
 
3.6工作负载
 
 
3.6.1创建部署
 
 
⚫在已创建的项目下选择应用负载 →工作负载 →部署，进入工作负载列表页面，选中部署
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_48.jpg?Expires=1758447100&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=DlZNsNhI8m5r51wbqyplbO068LA%3D) 
 
图3-26创建工作负载
 
 
⚫点击创建，填写基本信息
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_49.jpg?Expires=1758447100&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=EMLJTjFHGzoCC91OACjrAn4w6g0%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
21/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_50.jpg?Expires=1758447100&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=4bhZ6EtM1jTtoWXl71tsCSdITLA%3D) 
 
图3-27填写工作负载信息
 
 
Ehflame-Tech Coni⚫点击添加容器，填写镜像名称，使用资源，应用端口信息
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_51.jpg?Expires=1758447101&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=ZhW%2F6cPHpz5C4rnrSryGN8%2BiXmE%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
22/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_52.jpg?Expires=1758447101&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=MuR172v9sxoZWJNVk6QyKJYg%2FBw%3D) 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_53.jpg?Expires=1758447101&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=5%2F8XlaQJnopN9BSE5SxpmuusJk8%3D) 
 
图3-28创建负载容器组设置
 
 
⚫点击创建，完成部署的创建。
 
 
3.6.2有状态副本集
 
 
有状态副本集(StatefulSet)，是为了解决有状态服务的问题，在运行过程中会保存数据或状态，例如 Mysql，它需要存储产生的新数据。而 Deployments是为无状态服务而设计。
 
 
⚫在已创建的项目下选择工作负载 →有状态副本集，进入列表页
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_54.jpg?Expires=1758447101&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=j4ykmm45OEk6YQHhyqSclNWI0bk%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
23/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_55.jpg?Expires=1758447101&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=b%2B0stL0abwn%2BVx28Ax7GbTUkxwg%3D) 
 
图3-29有状态副本集
 
 
⚫点击创建，填写基本信息
 
 
0
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_56.jpg?Expires=1758447101&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=217dinT21vXowXyNLhKtzPmgm28%3D) 
 
图3-30填写有状态副本集信息
 
 
⚫点击添加容器，填写镜像名称，使用资源，应用端口信息
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_57.jpg?Expires=1758447101&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=ASO%2FYj738l4%2BC4BnBUy2L2Uj8UQ%3D) 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_58.jpg?Expires=1758447101&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=VJYFNhpsj3qrZ4Quu1I%2BVc4D2IU%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
24/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_59.jpg?Expires=1758447101&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=b2QNVMrE0KgeywNiYElosUT0YDo%3D) 
 
图3-31有状态副本集容器组设置
 
 
3.6.3守护进程集
 
 
守护进程集管理多组容器组副本，确保所有（或某些）节点运行一个容器组的副本。集群添加节点时，守护进程集会根据需要自动将容器组添加到新节点。
 
 
如果您想在所有节点或者没有用户干预的特定节点上部署持续运行的后台任务，守护进程集会非常有用。例如：
 
 
⚫在每个节点上运行日志收集守护进程，例如 Fluentd和 Logstash等。
 
 
⚫在每个节点上运行节点监控守护进程，例如 Prometheus Node Exporter和 collectd等。
 
 
⚫在每个节点上运行集群存储守护进程和系统程序，例如 kube-dns和 kube-proxy等。
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_60.jpg?Expires=1758447101&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=13SeLzwHYwDq25Ie6oWi8ZXikOU%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
25/ 57
 
 
⚫在已创建的项目下选择工作负载 →守护进程集，进入列表页
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_61.jpg?Expires=1758447101&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=1MydkWPxizKMvr77QaHRk8hRtGA%3D) 
 
⚫点击创建，填写基本信息
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_62.jpg?Expires=1758447102&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=MRlA3igsuEH1BvOfqIcGmU%2BxOAU%3D) 
 
图3-33填写守护进程集信息
 
 
⚫点击添加容器，填写镜像名称，使用资源，应用端口信息
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_63.jpg?Expires=1758447102&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=xvWn0UHT4L7i92BcPuZ2f1zlC0k%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
26/ 57
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_64.jpg?Expires=1758447102&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=BSj4wHehwXPAfVxI7Rm4RUXc%2Bqo%3D) 
 
下步
 
 
图3-34守护进程集容器组设置
 
 
3.7任务
 
 
3.7.1任务
 
 
任务会创建一个或者多个容器组，并确保指定数量的容器组成功结束。随着容器组成功结束，任务跟踪记录成功结束的容器组数量。当达到指定的成功结束数量时，任务（即 Job）完成。删除任务的操作会清除其创建的全部容器组。
 
 
在简单的使用场景中，您可以创建一个任务对象，以便可靠地运行一个容器组直到结束。当第一个容器组故障或者被删除（例如因为节点硬件故障或者节点重启）时，任务对象会启动一个新的容器组。您也可以使用一个任务并行运行多个容器组。
 
 
⚫在已创建好的项目下选择应用负载 →任务 →任务，点击创建
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_65.jpg?Expires=1758447102&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=4C4jWfgqbUCLjBeaQeodLDXzkRc%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
27/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_66.jpg?Expires=1758447102&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=ZGgIl6TxcPbrshLtbvd33%2FBJkOs%3D) 
 
图3-35创建任务
 
 
⚫填写基本信息，并点击下一步
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_67.jpg?Expires=1758447102&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=TjtQuziKUqSSqDhxyNMQP19Y53U%3D) 
 
图3-36填写任务信息
 
 
⚫填写策略设置，并点击下一步。
 
 
最大重试次数：指定将该任务视为失败之前的重试次数。默认值为 6。
 
 
容器组完成数量：指定该任务应该运行至成功结束的容器组的期望数量。
 
 
并行容器组数量：指定该任务在任何给定时间应该运行的最大期望容器组数量。
 
 
最大运行时间：指定该任务在系统尝试终止任务前处于运行状态的持续时间（相对于 stratTime），单位为秒；该值必须是正整数。
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_68.jpg?Expires=1758447102&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=SGY577EmvkBp5LEYc5lMefPEbn0%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
28/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_69.jpg?Expires=1758447102&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=Vo%2FyQhhiyi%2BZSdlkoO7ODATPmbk%3D) 
 
图3-37填写任务策略设置
 
 
Q
 
 
Enflame-Tech Cont
 
 
⚫点击添加容器，填写镜像名称，使用资源，启动命令，点击右下角的 √，然后选择下一步继续。
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_70.jpg?Expires=1758447102&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=BQqMERbHRi9s5bv%2FNlZjcsGxl%2Bk%3D) 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_71.jpg?Expires=1758447102&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=EyQMg2giTChqz%2BawQ%2BPDsEf7JjM%3D) 
 
29/ 57
 
 
创建任务
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_72.jpg?Expires=1758447102&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=Aab%2FB7TwYQ5e8RzSIO%2F9zM%2Fn4EE%3D) 
 
下一步
 
 
图3-38任务容器组设置
 
 
端口设置
 
 
设置用于访问容器的端口。
 
 
协议?  HTTP                  名称 http-                 容器端口                  盲
 
 
添加端口
 
 
Q优先使用本地镜像                                                                             v
 
 
口健康检查
 
 
添加探针以定时检查容器健康状态。
 
 
启动命令
 
 
目定义容器启动时运行的命令。默认情况下,容器启动时将运行镜像默认命令。
 
 
命令
 
 
参数
 
 
容器启动命令的参数。如有多个参数请使用半角运号()分隔.
 
 
图3-39填写任务启动命令
 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_73.jpg?Expires=1758447102&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=UHyK%2Bm8xvx6yJ1CHS%2F9MZ98DvXs%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
30/ 57
 
 
3.7.2定时任务
 
 
定时任务(CronJob)对于创建周期性和重复性任务非常有用，例如运行备份或发送电子邮件。定时任务还可以在特定时间或间隔执行单个任务，例如在集群可能处于空闲状态时执行任
 
 
⚫在已创建好的项目下选择应用负载 →任务 →定时任务，点击创建
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_74.jpg?Expires=1758447103&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=hRkWa1I9rB3ldCqmiTFo934ceGk%3D) 
 
图3-40创建定时任务
 
 
⚫填写基本信息，点击下一步
 
 
定时计划：按照给定的时间计划运行任务。语法参照 CRON
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_75.jpg?Expires=1758447103&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=ZZgI8Kc63I6c1LJazIAu42Z9COY%3D) 
 
图3-41填写定时任务信息
 
 
⚫填写策略设置，点击下一步
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_76.jpg?Expires=1758447103&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=qoPBMtvT%2Bl1Tq7ewF8tRGWd0jVU%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
31/ 57
 
 
⚫点击添加容器，填写镜像名称，使用资源，启动命令，点击右下角的√，然后选择下一步继续。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_77.jpg?Expires=1758447103&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=LptRraO0m5j07GxfeahJHbQowrs%3D) 
 
×
 
 
Enflan
 
 
图3-42定时任务容器组设置
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_78.jpg?Expires=1758447103&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=%2BjiVFEe%2BEyY0EP%2BSd4RFcBXkgjI%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
32/ 57
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_79.jpg?Expires=1758447103&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=wGTFtADC%2BiiUWtSwNEiUW7ObiUk%3D) 
 
图3-43填写定时任务启动命令
 
 
3.8网络与服务
 
 
ide
 
 
3.8.1服务管理
 
 
一个 Kubernetes的服务(Service)是一种抽象，它定义了一类 Pod的逻辑集合和一个用于访问它们的策略-有的时候被称之为微服务，而在这个集合中的 Pod的 IP地址以及数量等都会发生动态变化，这个服务的客户端并不需要知道这些变化，也不需要自己来记录这个集合的 Pod信息，这一切都是由抽象层 Service来完成。创建无状态服务：
 
 
⚫在已创建的项目下选择应用负载 →服务，点击创建
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_80.jpg?Expires=1758447103&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=9rTW4nKCw7xx%2BemWr5mjU05KNvc%3D) 
 
图3-44创建服务
 
 
⚫点击无状态服务
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_81.jpg?Expires=1758447103&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=ngxtvEb9k5UZ0fw7qRUAO45wdYw%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
33/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_82.jpg?Expires=1758447103&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=RmcvN%2FdzN%2FT114df27y1TfKNFkQ%3D) 
 
图3-45选择无状态服务
 
 
⚫填写基本信息，完成后点击下一步
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_83.jpg?Expires=1758447103&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=G97FcUstcnNVttYqhZIML2Hq03k%3D) 
 
描述可包含任意字符,最长256个字符。
 
 
取消      下一步
 
 
图3-46填写无状态服务信息
 
 
⚫点击添加容器，填写镜像名称，使用资源，应用端口信息
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_84.jpg?Expires=1758447103&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=aslyqG%2F7cOoivRwhGlDkBOb7WR0%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
34/ 57
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_85.jpg?Expires=1758447104&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=bv3M827kitRV4hK%2FusP1UsKVSYY%3D) 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_86.jpg?Expires=1758447104&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=znled%2F1g%2BWbnZoxEmCRefr9GV08%3D) 
 
图3-47无状态服务容器组设置
 
 
⚫点击下一步，可以设置挂载存储卷
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_87.jpg?Expires=1758447104&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=berTAKZp44jgfq9fQDziZrtUsS0%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
35/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_88.jpg?Expires=1758447104&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=56js184bjwFGHSMSlUpi1FtXP0E%3D) 
 
图3-48无状态服务存储卷设置
 
 
⚫点击创建，即可完成应用的创建
 
 
3.8.2灰度发布
 
 
当您在 TopsStack中升级应用至新版本时，灰度发布可以确保平稳过渡。采用的具体策略可能不同，但最终目标相同，即提前识别潜在问题，避免影响在生产环境中运行的应用。这样不仅可以将版本升级的风险降到最低，还能测试应用新构建版本的性能。下面的灰度发布策略以 bookinfo为例，bookinfo是系统自带的示例应用。
 
 
⚫蓝绿部署：蓝绿部署会创建一个相同的备用环境，在该环境中运行新的应用版本，从而为发布新版本提供一个高效的方式，不会出现宕机或者服务中断。通过这种方法，TopsStack将所有流量路由至其中一个版本，即在任意给定时间只有一个环境接收流量。如果新构建版本出现任何问题，您可以立刻回滚至先前版本。
 
 
⚫在已创建的项目下选择灰度发布，在发布模式选项卡下，点击蓝绿部署右侧的创建
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_89.jpg?Expires=1758447104&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=TYSeUm58g9PrKQUmJ8xqWMqXBZQ%3D) 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_90.jpg?Expires=1758447104&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=bUQ8mD1N8nxMi523fTtn3%2BSIxds%3D) 
 
图3-49创建灰度发布
 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
36/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_91.jpg?Expires=1758447104&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=%2B%2FVeCUBdGgqYQFyscreE0xaBtVs%3D) 
 
⚫在新版本设置选项卡，添加另一个版本 v2，然后点击下一步
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_92.jpg?Expires=1758447104&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=L8ODgabfXIckxhEqtejxq5vZTgE%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
37/ 57
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_93.jpg?Expires=1758447104&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=K97CIP6fRViH29QkQcf92zlHtjM%3D) 
 
金丝雀发布：金丝雀部署缓慢地向一小部分用户推送变更，从而将版本升级的风险降到最低。具体来讲，您可以在高度响应的仪表板上进行定义，选择将新的应用版本暴露给一部分生产流量。另外，您执行金丝雀部署后，TopsStack会监控请求，为您提供实时流量的可视化视图。在整个过程中，您可以分析新的应用版本的行为，选择逐渐增加向它发送的流量比例。待您对构建版本有把握后，便可以把所有流量路由至该构建版本。
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_94.jpg?Expires=1758447104&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=YXu35z%2F3Olt1DGyCX5PLEfXlq80%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
38/ 57
 
 
⚫在已创建的项目下选择灰度发布，在发布模式选项卡下，点击金丝雀发布右侧的创建。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_95.jpg?Expires=1758447104&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=nspDapdySU%2F6Jb1YWgbQxBV%2BJ2s%3D) 
 
图3-54创建金丝雀发布任务
 
 
⚫填写基本信息，然后点击下一步。
 
 
创建金丝雀发布任务
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_96.jpg?Expires=1758447105&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=Y8FJjtaMOUu5gXgCGpSkAeVXnJM%3D) 
 
名称*
 
 
canary-upgrade
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_97.jpg?Expires=1758447105&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=%2FT0JBQCoD6XvROG%2BJDrefL%2FuSgY%3D) 
 
名称只能包含小写字母、数字和连字符(-),必须以小写字母或数字开头和结尾,最长253个字符。
 
 
<              取消      下一步
 
 
图3-55填写金丝雀发布任务信息
 
 
⚫在服务设置选项卡，选择 reviews并点击下一步。
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_98.jpg?Expires=1758447105&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=gSKeXGWXOSinJebEXuHRCI5lYUw%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
39/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_99.jpg?Expires=1758447105&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=WDwMygEEEwYSA4GAWN%2BYES08LVc%3D) 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_100.jpg?Expires=1758447105&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=zd%2B3wRSn6woYWdECODTLot0CrPw%3D) 
 
取消       上一步      下一步
 
 
图3-57金丝雀新版本设置
 
 
⚫选择指定流量分配，并拖动中间的滑块来更改向这两个版本分别发送的流量比例。操作完成后，点击创建。
 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
40/ 57
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_101.jpg?Expires=1758447105&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=9CnLsl%2FHYOMDmnHLgerihjhkxmw%3D) 
 
图3-58金丝雀策略设置
 
 
⚫流量镜像：流量镜像复制实时生产流量并发送至镜像服务。默认情况下， TopsStack会镜像所有流量，您也可以指定一个值来手动定义镜像流量的百分比。
 
 
⚫在已创建的项目下选择灰度发布，在发布模式选项卡下，点击流量镜像右侧的创建。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_102.jpg?Expires=1758447105&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=G7JAwdanpUh%2FbiBtaKCHoGmYSFc%3D) 
 
图3-59创建流量镜像任务
 
 
⚫填写基本信息，然后点击下一步。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_103.jpg?Expires=1758447105&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=eDkeINcPWpCtq2lVgezvEOHo5gQ%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
41/ 57
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_104.jpg?Expires=1758447105&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=Wn9l6PcU8wkApdTtgqMy63IOL8g%3D) 
 
名称只能包含小写字母、数字和连字符(-),必须以小写字母或数字开头和结尾,最长253个字符。
 
 
取消      下一步
 
 
图3-60填写流量镜像任务信息
 
 
⚫在服务设置选项卡，选择 reviews并点击下一步。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_105.jpg?Expires=1758447105&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=q9crDE4LNPRr3oz3PPsmWZQGX4Y%3D) 
 
图3-61流量镜像任务服务设置
 
 
⚫在新版本设置选项卡，添加另一个版本 v2，然后点击下一步。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_106.jpg?Expires=1758447105&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=O%2FzP7YmHwr5qv49S6Gidxkey6Aw%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
42/ 57
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_107.jpg?Expires=1758447105&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=OQ2Nn5iWDF9Za8SGFomXvct6%2FIk%3D) 
 
图3-62流量镜像任务新版本设置
 
 
⚫在策略设置选项卡，点击创建。
 
 
3.8.3应用路由
 
 
应用路由(Ingress)是用来聚合集群内服务的方式，对应的是 Kubernetes的 Ingress资源，后端使用了 Nginx Controller来处理具体规则。Ingress可以给 service提供集群外部访问的 URL、负载均衡、SSL termination、HTTP路由等。您可以使用应用路由和单个 IP地址来聚合和暴露多个服务。下面的例子中，假设你已经创建了一个无状态的 nginx服务。
 
 
⚫在已创建的项目下选择应用负载 →应用路由，点击创建。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_108.jpg?Expires=1758447106&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=ny%2BSeii8De7Q0%2BCLmY%2FVVVXgze4%3D) 
 
图3-63创建应用路由
 
 
⚫填写基本信息，完成后点击下一步。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_109.jpg?Expires=1758447106&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=QhB41Bon8Vx0M%2B8tOqiBxJslSHI%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
43/ 57
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_110.jpg?Expires=1758447106&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=QrjiUCZ3eG78ku0i%2FrSm%2BhauF1M%3D) 
 
取消     下一步
 
 
图3-64填写应用路由信息
 
 
⚫在路由规则选项卡中，点击添加路由规则，选择一种模式来配置路由规则，点击√，然后点击下一步。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_111.jpg?Expires=1758447106&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=cSmYvjw0mC8jdH%2FK867Vl7DV%2FQc%3D) 
 
模式
 
 
自动生成           指定域名
 
 
系统将自动按照<服务名称>.<项目名称>.<网关地址>.nlp.lo格式生成域名,该域名将由nlp.lo自动解析为网关IP地址,此模式仅支持HTTP协议。
 
 
路径·
 
 
1                                nginx                             8080                       v
 
 
添加
 
 
<
 
 
√
 
 
取消      上一步      下一步
 
 
图3-65填写路由规则
 
 
⚫在高级设置选项卡，点击创建。
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_112.jpg?Expires=1758447106&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=Z4xMLNtYeUGI7RYwrimthPzekqY%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
44/ 57
 
 
3.9监控告警
 
 
3.9.1通知管理
 
 
下面讲解如何配置企业微信通知并添加相应 ID来接收告警策略的通知。本手册假设你已经申请了企业微信，并配置了部门，添加了成员。
 
 
⚫点击工作台->通知管理。
 
 
⚫前往通知管理下的通知配置，选择企业微信。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_113.jpg?Expires=1758447106&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=wXSuwVQJ8KsQ9d4Fq8oSMHGqDc0%3D) 
 
图3-66企业微信通知配置
 
 
⚫在服务器设置下的企业 ID、应用 AgentId以及应用 Secret中分别输入您的企业ID、应用 AgentId以及应用 Secret。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_114.jpg?Expires=1758447106&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=vTtTRkUXKDeDEDqTzwD5uw0ggrU%3D) 
 
图3-67企业微信服务设置
 
 
⚫在接收设置中，输入用户 ID后点击添加。您可以添加多个 ID。
 
 
接收设造
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_115.jpg?Expires=1758447106&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=FDGY%2B3l4SK0yDppoxfvl52%2F62T0%3D) 
 
图3-68企业微信接收设置
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_116.jpg?Expires=1758447106&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=MWLOkSrPGz%2F04S9ZvCQ9Ybht7bA%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
45/ 57
 
 
⚫勾选通知条件左侧的复选框即可设置通知条件。您可以点击添加来添加多个通知条件，或点击通知条件右侧的删除图标来删除通知条件。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_117.jpg?Expires=1758447106&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=udiRm0xeOxw9xWTMrCR7d9lB6U8%3D) 
 
图3-69企业微信通知条件
 
 
⚫配置完成后，您可以点击右下角的发送测试信息进行验证。
 
 
⚫在右上角，打开未启用开关来接收企业微信通知，或者关闭已启用开关来停用企业微信通知。
 
 
3.9.2告警策略
 
 
TopsStack支持针对节点和工作负载的告警策略。本手册演示如何为项目中的工作负载创建告警策略。
 
 
⚫在已创建的项目下选择监控告警 →告警策略，进入列表页。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_118.jpg?Expires=1758447106&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=t0Qc%2Fcv8AVduNeJwMJStFHKgbj4%3D) 
 
图3-70创建告警策略
 
 
⚫点击创建，填写基本信息，然后点击下一步。
 
 
名称：使用简明名称作为其唯一标识符。
 
 
别名：帮助您更好地识别告警策略。
 
 
描述信息：对该告警策略的简要介绍。
 
 
阈值时间（分钟）：告警规则中设置的情形持续时间达到该阈值后，告警策略将变为触发中状态。
 
 
告警级别：提供的值包括一般告警、重要告警和危险告警，代表告警的严重程度
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_119.jpg?Expires=1758447107&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=Zsv2aKYu7IoNcwptChtiE0t6YOc%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
46/ 57
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_120.jpg?Expires=1758447107&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=OpmwI42JO78yM4yF4ZBgeFHQOhk%3D) 
 
图3-71填写告警策略信息
 
 
⚫在规则设置选项卡，您可以使用规则模板或创建自定义规则。若想使用模板，请填写以下字段。完成后，点击下一步
 
 
资源类型：选择想要监控的资源类型，例如部署、有状态副本集或守护进程集。
 
 
监控目标：取决于您所选择的资源类型，目标可能有所不同。如果项目中没有工作负载，则无法看到任何监控目标。
 
 
告警规则：为告警策略定义规则。这些规则基于 Prometheus表达式，满足条件时将会触发告警。您可以对 CPU、内存等对象进行监控。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_121.jpg?Expires=1758447107&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=5yWHd7ktLQXSd3wo50F%2B9FjxYdU%3D) 
 
图3-72告警策略规则设置
 
 
⚫在消息设置选项卡，输入想要在包含在通知中的告警标题和消息，然后点击创建
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_122.jpg?Expires=1758447107&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=jXkIjcz7esDbmoV3WbR7h59cUHQ%3D) 
 
47/ 57
 
 
⚫告警策略刚创建后将显示为未触发状态；一旦满足规则表达式中的条件，则会首先达到待触发状态；满足告警条件的时间达到阈值时间后，将变为触发中状态
 
 
3.9.3告警消息
 
 
告警消息中记录着按照告警规则触发的告警的详细信息。本手册演示如何查看工作负载级别的告警消息。
 
 
⚫在已创建的项目下选择监控告警 →告警消息，进入列表页
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_123.jpg?Expires=1758447107&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=OCS7FzKeXablYzdlFAV6FICjiAc%3D) 
 
图3-73告警消息列表
 
 
⚫在告警消息页面，可以看到列表中的全部告警消息。第一列显示您在告警通知中定义的标题和消息。如需查看某一告警消息的详情，点击该告警策略的名称，然后在显示的页面中点击告警历史选项卡
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_124.jpg?Expires=1758447107&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=z2Aw9IJnM7REYdqSoizMRiioXJQ%3D) 
 
图3-74查看告警消息详情
 
 
⚫在告警历史选项卡，您可以看到告警级别、监控目标以及告警激活时间
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_125.jpg?Expires=1758447107&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=sAZT9GP6NXP8zm1bFmWcoBat66U%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_126.jpg?Expires=1758447107&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=xXSIjbybIbTFwoeklhY0Qc0LskM%3D) 
 
图3-75查看告警历史
 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
48/ 57
 
 
3.10集群管理
 
 
3.10.1添加集群
 
 
在使用 TopsStack的中央控制平面管理多个集群之前，您需要创建一个主集群。主集群实际上是一个启用了多集群功能的 TopsStack集群，您可以使用它提供的控制平面统一管理。成员集群是没有中央控制平面的普通 TopsStack集群。也就是说，拥有必要权限的租户（通常是集群管理员）能够通过主集群访问控制平面，管理所有成员集群，例如查看和编辑成员集群上面的资源。反过来，如果您单独访问任意成员集群的 Web控制台，您将无法查看其他集群的任何资源。
 
 
只能有一个主集群存在，而多个成员集群可以同时存在。在多集群架构中，主集群和成员集群之间的网络可以直接连接，或者通过代理连接。成员集群之间的网络可以设置在完全隔离的环境中。
 
 
下面介绍直连连接的方式添加集群。
 
 
如果主集群的任何节点都能访问的 kube-apiserver地址，您可以采用直接连接。当成员集群的 kube-apiserver地址可以暴露给外网，或者主集群和成员集群在同一私有网络或子网中时，此方法均适用。
 
 
要通过直接连接使用多集群功能，您必须拥有至少两个集群，分别用作主集群和成员集群。您可以在安装 TopsStack之前或者之后将一个集群指定为主集群或成员集群。
 
 
⚫以 admin身份登录 TopsStack控制台，转到集群管理页面点击添加集群。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_127.jpg?Expires=1758447107&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=33Wus2Yuc5h0bnBbzkCQWQ0u7gY%3D) 
 
图3-76集群管理
 
 
⚫在导入集群页面，输入要导入的集群的基本信息。您也可以点击右上角的编辑模式以 YAML格式查看并编辑基本信息。编辑完成后，点击下一步。
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
49/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_128.jpg?Expires=1758447107&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=olaW1ZBdQu4br96c7jMWu4JzKSE%3D) 
 
图3-77填写集群信息
 
 
⚫在连接方式，选择直接连接 Kubernetes集群，复制 kubeconfig内容并粘贴至文本框。您也可以点击右上角的编辑模式以 YAML格式编辑的 kubeconfig。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_129.jpg?Expires=1758447107&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=Tf3%2FRiynNqBOemmsuW65ZZKIW44%3D) 
 
图3-78填写集群kubeconfig
 
 
⚫点击创建，然后等待集群初始化完成。
 
 
3.10.2集群节点
 
 
Kubernetes将容器放入容器组（Pod）中并在节点上运行，从而运行工作负载。取决于具体的集群环境，节点可以是虚拟机，也可以是物理机。每个节点都包含运行容器组所需的服务，这些服务由控制平面管理。
 
 
本手册介绍集群管理员可查看的集群节点信息和可执行的操作。节点状态：
 
 
⚫点击工作台->集群管理
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_130.jpg?Expires=1758447107&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=jZEEe%2Bp8NEuUYeT%2FFTRd%2Ftp2tyY%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
50/ 57
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_131.jpg?Expires=1758447107&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=OyJggNZ9DQG4yto9VWMfHsgjdIs%3D) 
 
图3-79平台管理
 
 
⚫选择 host集群
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_132.jpg?Expires=1758447108&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=7jn6GxaGgT%2FCO0j11GDLCkb8InE%3D) 
 
图3-80选择集群
 
 
⚫在左侧导航栏中选择节点下的集群节点，查看节点的状态详情
 
 
名称：节点的名称和子网 IP地址。
 
 
状态：节点的当前状态，标识节点是否可用。
 
 
角色：节点的角色，标识节点是工作节点还是主节点。
 
 
CPU用量：节点的实时 CPU用量。
 
 
内存用量：节点的实时内存用量。
 
 
容器组：节点的实时容器组用量。
 
 
已分配 CPU：该指标根据节点上容器组的总 CPU请求数计算得出。
 
 
已分配内存：该指标根据节点上容器组的总内存请求计算得出。
 
 
已分配 GCU：该指标根据节点上容器组的总 GCU请求数计算得出。
 
 
已分配 vGCU：该指标根据节点上容器组的总 vGCU请求数计算得出。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_133.jpg?Expires=1758447108&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=W%2F9FMq%2F5J20vhpNiO9q7mitxIao%3D) 
 
图3-81集群节点列表
 
 
⚫点击其中一个节点，查看运行状态
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_134.jpg?Expires=1758447108&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=d8WBnes9HHnu%2FLZrsX6LBFirBGw%3D) 
 
51/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_135.jpg?Expires=1758447108&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=7sp55KwnIPzmQwRa%2Bh9yuZPQj3g%3D) 
 
图3-82查看节点运行状态
 
 
⚫切换到监控，查看节点的监控数据
 
 
0
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_136.jpg?Expires=1758447108&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=N9n4kpzHceuM%2FdsKebG%2FNHCQRoc%3D) 
 
图3-83查看节点监控
 
 
⚫节点管理：在集群节点页面，您可以执行以下操作：停止调度/启用调度：点击集群节点右侧的操作图标，然后点击停止调度或启用调度停止或启用调度节点。您可以在节点重启或维护期间将节点标记为不可调度。Kubernetes调度器不会将新容器组调度到标记为不可调度的节点。但这不会影响节点上现有工作负载。
 
 
集群节点品
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_137.jpg?Expires=1758447108&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=htcROb%2BXrvW5vmJCaLrvm7gniFw%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_138.jpg?Expires=1758447108&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=d1TDBuZANG6cvrfExxRCxiqqXY4%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
52/ 57
 
 
图3-84停止调度
 
 
⚫应用驱逐/负载恢复：点击集群节点右侧的操作图标，然后点击应用驱逐或负载恢复驱逐节点或者恢复节点，节点驱逐会将节点上运行的所有负载杀掉，驱逐后也不可调度到节点。升级芯片的驱动前需要驱逐节点，确保芯片没有被使用。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_139.jpg?Expires=1758447108&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=%2FKhbZjLzP%2BQ40iFCimHHXhgy%2B%2Fw%3D) 
 
图3-85应用驱逐
 
 
⚫编辑污点：污点允许节点排斥一些容器组。勾选目标节点前的复选框，在上方弹出的按钮中点击编辑污点。在弹出的编辑污点对话框，您可以添加或删除污点。
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_140.jpg?Expires=1758447108&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=ETU94BK%2F0iRsz%2FCBhVJaLhtr1uM%3D) 
 
图3-86编辑污点
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_141.jpg?Expires=1758447108&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=gATX1FosFQqQeIyiFFrCYOK5GIA%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
53/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_142.jpg?Expires=1758447108&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=IAsz%2BIM5WYfEBusbvkDkJwjioyg%3D) 
 
图3-87添加污点信息
 
 
⚫编辑标签：您可以利用节点标签将容器组分配给特定节点。首先标记节点（例如，用 node-role.kubernetes.io/gpu-node标记 GPU节点），然后在创建工作负载时在高级设置中添加此标签，从而使容器组在 GPU节点上运行。要添加节点标签，请点击更多操作>编辑标签。
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_143.jpg?Expires=1758447108&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=rm%2F0%2BRjTtoK8hlguIHTd8AHJfic%3D) 
 
EntN
 
 
图3-88编辑节点标签
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_144.jpg?Expires=1758447109&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=Y11yX8aI7dTCGwMf8uQ%2BYdavJ7k%3D) 
 
图3-90选择网关模式
 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
54/ 57
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_145.jpg?Expires=1758447109&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=FxUhXB8%2FrpmzuUv993Ffk3ON78s%3D) 
 
图3-89修改节点标签
 
 
⚫查看节点运行状态、容器组、元数据、监控和事件。
 
 
3.10.3网关设置
 
 
TopsStack提供集群级别的网关，使所有项目共用一个全局网关。本文档介绍如何在TopsStack设置集群网关。
 
 
创建集群网关
 
 
⚫以 admin身份登录 web控制台，点击左上角的平台管理并选择集群管理
 
 
⚫点击导航面板中集群设置下的网关设置，选择集群网关选项卡，并点击开启网关
 
 
⚫在显示的对话框中，选择网关的访问模式
 
 
开启网关
 
 
在创建应用路由之前,需要先开启外网访问入口,即网关。这一步是创建对应的应用路由控制器,负责将请求转发到对应的后端服务
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_146.jpg?Expires=1758447109&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=yXv1gfaLD34DHM0yWaDrypIizpI%3D) 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_147.jpg?Expires=1758447109&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=MLSv%2BYq6u2Wo%2F6HJntEMrDa2LL4%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
55/ 57
 
 
⚫点击确定创建集群网关
 
 
⚫在这个页面中会展示创建的集群网关和该网关的基本信息
 
 
⚫点击管理，从下拉菜单中选择一项操作:
 
 
⚫查看详情：转至集群网关详情页面。
 
 
⚫编辑：编辑集群网关配置。
 
 
⚫关闭：关闭集群网关。
 
 
集群网关详情
 
 
⚫在集群网关选项卡下，点击集群网关右侧的管理，选择查看详情以打开其详情页面
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_148.jpg?Expires=1758447109&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=b5CKjGqcufvOhUSD3Elb2v3rKwI%3D) 
 
图3-91查看网关详情
 
 
⚫在详情页面，点击编辑以配置集群网关，或点击关闭以关闭网关
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_149.jpg?Expires=1758447109&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=y4ofAw4h81yRuZ8EgwIHBEnkFdM%3D) 
 
图3-92查看网关监控
 
 
⚫点击监控选项卡，查看集群网关的监控指标
 
 
⚫点击配置选项选项卡以查看集群网关的配置选项
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_150.jpg?Expires=1758447109&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=cXaIy6b0fCbEJaob3LEwPbi%2BwKg%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
56/ 57
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_151.jpg?Expires=1758447109&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=eUXEAyhIUsCFM3Z6pSjanJPWso4%3D) 
 
图3-93查看网关配置选项
 
 
⚫点击网关日志选项卡以查看集群网关日志
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_152.jpg?Expires=1758447109&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=WRrwEJoNZEN8F8FlxyLIhAlGry8%3D) 
 
图3-94查看网关日志
 
 
⚫点击资源状态选项卡，以查看集群网关的负载状态
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_153.jpg?Expires=1758447109&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=L4FtNgDtmJF7DmN1rqLXLmLbtYI%3D) 
 
图3-95查看网关资源状态
 
 
⚫点击元数据选项卡，以查看集群网关的注解
 
 
异构算力调度平台 TopsStack用户使用手册
 
 
![](http://darwin-controller-pro-01.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_154.jpg?Expires=1758447109&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=VX2R%2Bb%2FwaoI0I0dLjWbPeNNfWEU%3D) 
 
版权所有©2022上海燧原科技有限公司保留所有权利
 
 
57/ 57
 
 
![](http://darwin-controller-pro.oss-cn-hangzhou.aliyuncs.com/docs/1419011741985099776/%E3%80%90%E5%8E%9F%E6%96%87%E3%80%91%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E8%B0%83%E5%BA%A6%E5%B9%B3%E5%8F%B0TopsStack%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C_155.jpg?Expires=1758447109&OSSAccessKeyId=LTAI5tBVMtznbk7xyCa56gof&Signature=ZDOUiTvRGLhQbIi9aVtphuMvWFo%3D) 
 
图3-96查看网关元数据
 
 
