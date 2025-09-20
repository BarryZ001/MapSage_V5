
## 版本申明

| 版本 | 修改内容               | 修改时间  |
| ---- | ---------------------- | --------- |
| v1.0 | 初始版本，新增使用方法 | 8/12/2022 |
|      |                        |           |
|      |                        |           |



## 简介

go-eflib 是基于 Enflame Management Library (EFML) 的golang开发库，经过对efml的二次封装提供GCU设备管理golang API，从而给开发者用户提供了enflame GCU 的设备管理golang接口。


## 使用配置

### go.mod配置

将 `go-eflib` 添加到 golang项目的 `go.mod `文件里, 例如:

```
module modulename

go 1.15

require (
	go-eflib v1.0.0
)

```



### vendor配置

先将`go-eflib` 整个目录复制到golang项目的vendor目录下，例如：

```
# ls vendor/
github.com  go-eflib  golang.org  google.golang.org  gopkg.in  k8s.io  modules.txt

```

同时将 `go-eflib` 添加到 `vendor/modules.txt` 里, 例如:

```
# go-eflib v1.0.0
## explicit
go-eflib
go-eflib/efml

```



### import导入

编写代码时需要将go-eflib 导入文件内，例如:

```
package metrics

import (
	"strconv"

	"go-eflib"
	"go-eflib/efml"
)

```

如果添加了 `go-eflib/efml`  但是 文件里没用上，会出现类似以下的error信息：

```
imported and not used: "go-eflib/efml"
```

那么就在import里删除 `go-eflib/efml`这一行。



### 代码示例

参考 目录下的 `samples/metrics.go`。



## 常见问题

### go-eflib 里的libefml.so 版本

默认采用主机上的libefml.so，因此需要保证主机上已安装libefml.so。如果是ARM平台 存在 libefml-arm.so。
建议执行 `ln -sf /usr/lib/libefml-arm.so /usr/lib/libefml.so` 保证 /usr/lib/libefml.so 库名称可兼容。

### 是否提供API介绍说明

推荐直接看代码以及编程samples示例。







