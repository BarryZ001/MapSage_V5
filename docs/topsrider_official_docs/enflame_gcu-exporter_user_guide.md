
## 版本申明

| 版本 | 修改内容                         | 修改时间   |
| ---- | -------------------------------- | ---------- |
| v1.0 | 初始化，指标说明                 | 5/24/2022  |
| v1.1 | 更新内存总量指标名称             | 5/25/2022  |
| v1.2 | 更新内存指标标签与端口说明       | 5/30/2022  |
| v1.3 | 新增告警指标说明                 | 6/20/2022  |
| v1.4 | 更新告警处理策略说明以及指标内容 | 6/22/2022  |
| v1.5 | 填加 部署章节                    | 7/04/2022 |
| v.1.6 | 更新常见问题以及排版 | 7/13/2022 |
| v.1.7 | 新增power consumption 以及 capability | 1/11/2023 |
| | | |


## 简介

Gcu-exporter是一个企业级的Prometheus Exporter，其用于采集Enflame GCU的运行指标，然后将这些指标通过 Prometheus展示到Grafana或其他可视化界面以便用户获取设备的运行指标与告警信息。 Gcu-exporter 依赖于EFML（Enflame Management Library）获取Enflame GCU的运行指标信息，除了支持Enflame的训练加速卡之外同时也支持推理加速卡。



## 端口

- gcu-exporter 的默认端口为 “9400”
- 通过 `web.listen-address"` 可以修改端口，例如 "gcu-exporter --web.listen-address=:9402"



## 指标说明

### enflame_gcu_usage

#### 指标说明

- enflame_gcu_usage：GCU整卡利用率

#### 标签说明

- busid ：bus总线号
- host ：主机名称
- minor_number：设备号，卡号
- name：加速卡设备型号
- pod_name：加速卡所在的k8s pod 名称
- pod_namespace: 加速卡pod所在k8s命名空间
- slot：设备槽号
- uuid : 加速卡uuid

#### 示例

 ```
enflame_gcu_usage{busid="0000:40:00.0",host="sse-lab-inspur-043",minor_number="2",name="Enflame T10",pod_name="pod-gcu-example",pod_namespace="enflame",slot="PCIE3_GPU",uuid="GCU-U53000080105"} 80
 ```

- enflame_gcu_usage 值 80，表明GCU当前利用率 80%
- busid：Bus总线号为 0000:40:00.0
- host ：主机名称sse-lab-inspur-043
- minor_number：设备号即卡号等于 2
- name：加速卡设备型号为Enflame T10
- pod_name：加速卡所在的k8s pod 名称 为 pod-gcu-example
- pod_namespace: 加速卡pod所在k8s命名空间为enflame
- slot：设备卡槽为PCIE3_GPU
- uuid : 加速卡uuid 为 GCU-U53000080105



### enflame_gcu_memory_total_bytes

#### 指标说明

- enflame_gcu_memory_total_bytes：GCU 内存总量，单位Byte

#### 标签说明

- busid ：bus总线号
- host ：主机名称
- minor_number：设备号，卡号
- name：加速卡设备型号
- pod_name：加速卡所在的k8s pod 名称
- pod_namespace: 加速卡pod所在k8s命名空间
- slot：设备槽号
- uuid : 加速卡uuid

#### 示例

 ```
enflame_gcu_memory_total_bytes{busid="0000:40:00.0",host="sse-lab-inspur-043",minor_number="2",name="Enflame T10",pod_name="pod-gcu-example",pod_namespace="enflame",slot="PCIE3_GPU",uuid="GCU-U53000080105"} 1.7179869184e+10
 ```

- enflame_gcu_memory_total_bytes值 1.7179869184e+10，表明GCU当前内存总量为1.7179869184e+10 bytes
- busid ：bus总线号为0000:40:00.0
- host ：主机名称sse-lab-inspur-043
- minor_number：设备号即卡号等于 2
- name：加速卡设备型号为Enflame T10
- pod_name：加速卡所在的k8s pod 名称 为 pod-gcu-example
- pod_namespace: 加速卡pod所在k8s命名空间为enflame
- slot：设备卡槽为PCIE3_GPU
- uuid : 加速卡uuid 为 GCU-U53000080105



### enflame_gcu_memory_used_bytes

#### 指标说明

- enflame_gcu_memory_used_bytes：GCU 当前内存使用量，单位Byte

#### 标签说明

- busid ：bus总线号
- host ：主机名称
- minor_number：设备号，卡号
- name：加速卡设备型号
- pod_name：加速卡所在的k8s pod 名称
- pod_namespace: 加速卡pod所在k8s命名空间
- slot：设备槽号
- uuid : 加速卡uuid

#### 示例

 ```
enflame_gcu_memory_used_bytes{busid="0000:40:00.0",host="sse-lab-inspur-043",minor_number="2",name="Enflame T10",pod_name="pod-gcu-example",pod_namespace="enflame",slot="PCIE3_GPU",uuid="GCU-U53000080105"} 1073741824
 ```

- enflame_gcu_memory_used_bytes值 1073741824，表明GCU当前内存使用量为1073741824 bytes
- busid ：bus总线号为0000:40:00.0
- host ：主机名称sse-lab-inspur-043
- minor_number：设备号即卡号等于 2
- name：加速卡设备型号为Enflame T10
- pod_name：加速卡所在的k8s pod 名称 为 pod-gcu-example
- pod_namespace: 加速卡pod所在k8s命名空间为enflame
- slot：设备卡槽为PCIE3_GPU
- uuid : 加速卡uuid 为 GCU-U53000080105

### enflame_gcu_memory_usage

#### 指标说明

- enflame_gcu_memory_usage：GCU 内存利用率

#### 标签说明

- busid ：bus总线号
- host ：主机名称
- minor_number：设备号，卡号
- name：加速卡设备型号
- pod_name：加速卡所在的k8s pod 名称
- pod_namespace: 加速卡pod所在k8s命名空间
- slot：设备槽号
- uuid : 加速卡uuid

#### 示例

 ```
enflame_gcu_memory_usage{busid="0000:40:00.0",host="sse-lab-inspur-043",minor_number="2",name="Enflame T10",pod_name="pod-gcu-example",pod_namespace="enflame",slot="PCIE3_GPU",uuid="GCU-U53000080105"} 0.00982666015625
 ```

- enflame_gcu_memory_usage值 0.00982666015625，表明GCU当前内存总量为0.98%,例如当内存使用量为161MB，内存总量为16GB时，利用率计算公式为161/16*1024=0.00982666015625，约等于0.98%

- busid ：bus总线号为0000:40:00.0

- host ：主机名称sse-lab-inspur-043

- minor_number：设备号即卡号等于 2

- name：加速卡设备型号为Enflame T10

- pod_name：加速卡所在的k8s pod 名称 为 pod-gcu-example

- pod_namespace: 加速卡pod所在k8s命名空间为enflame

- slot：设备卡槽为PCIE3_GPU

- uuid : 加速卡uuid 为 GCU-U53000080105

  

### enflame_gcu_count

#### 指标说明

- enflame_gcu_count：主机内GCU 卡数，默认为8

#### 标签说明

- host ：主机名称

#### 示例

 ```
enflame_gcu_count{host="sse-lab-inspur-043"} 8
 ```

- enflame_gcu_count值8，表明当前主机内加速卡数量为8

- host ：主机名称sse-lab-inspur-043

  

### enflame_gcu_clock

#### 指标说明

- enflame_gcu_clock：GCU 加速卡的时钟频率

#### 标签说明

- busid ：bus总线号
- host ：主机名称
- minor_number：设备号，卡号
- name：加速卡设备型号
- power_mode: 加速卡电源工作模式，分为"Sleep"模式与"Active"模式
- pod_name：加速卡所在的k8s pod 名称
- pod_namespace: 加速卡pod所在k8s命名空间
- slot：设备槽号
- uuid : 加速卡uuid

#### 示例

 ```
enflame_gcu_clock{busid="0000:40:00.0",host="sse-lab-inspur-043",minor_number="2",name="Enflame T10",pod_ame="pod-gcu-example",pod_namespace="enflame",power_mode="Active",slot="PCIE3_GPU",uuid="GCU-U53000080105"} 1150
 ```

- enflame_gcu_clock值 1150，表明GCU当前时钟频率为1150 MHz
- busid ：bus总线号为0000:40:00.0
- host ：主机名称sse-lab-inspur-043
- minor_number：设备号即卡号等于 2
- name：加速卡设备型号为Enflame T10
- power_mode: 加速卡当前电源工作模式为"Active"
- pod_name：加速卡所在的k8s pod 名称 为 pod-gcu-example
- pod_namespace: 加速卡pod所在k8s命名空间为enflame
- slot：设备卡槽为PCIE3_GPU
- uuid : 加速卡uuid 为 GCU-U53000080105



### enflame_gcu_temperatures

#### 指标说明

- enflame_gcu_temperatures：GCU 加速卡的温度

#### 标签说明

- busid ：bus总线号
- host ：主机名称
- minor_number：设备号，卡号
- name：加速卡设备型号
- pod_name：加速卡所在的k8s pod 名称
- pod_namespace: 加速卡pod所在k8s命名空间
- slot：设备槽号
- uuid : 加速卡uuid

#### 示例

 ```
enflame_gcu_temperatures{busid="0000:40:00.0",host="sse-lab-inspur-043",minor_number="2",name="Enflame T10",pod_name="pod-gcu-example",pod_namespace="enflame",slot="PCIE3_GPU",uuid="GCU-U53000080105"}  34.5
 ```

- enflame_gcu_temperatures值 34.5，表明GCU加速卡当前温度为34.5度
- busid ：bus总线号为0000:40:00.0
- host ：主机名称sse-lab-inspur-043
- minor_number：设备号即卡号等于 2
- name：加速卡设备型号为Enflame T10
- pod_name：加速卡所在的k8s pod 名称 为 pod-gcu-example
- pod_namespace: 加速卡pod所在k8s命名空间为enflame
- slot：设备卡槽为PCIE3_GPU
- uuid : 加速卡uuid 为 GCU-U53000080105

### enflame_gcu_power_usage

#### 指标说明

- enflame_gcu_power_usage：GCU 加速卡的电源利用率

#### 标签说明

- busid ：bus总线号
- host ：主机名称
- minor_number：设备号，卡号
- name：加速卡设备型号
- pod_name：加速卡所在的k8s pod 名称
- pod_namespace: 加速卡pod所在k8s命名空间
- slot：设备槽号
- uuid : 加速卡uuid

#### 示例

 ```
enflame_gcu_power_usage{busid="0000:40:00.0",host="sse-lab-inspur-043",minor_number="2",name="Enflame T10",pod_name="pod-gcu-example",pod_namespace="enflame",slot="PCIE3_GPU",uuid="GCU-U53000080105"} 22.22222328186035
 ```

- enflame_gcu_power_usage值22.22222328186035 ，表明GCU加速卡当前电源利用率为22.22%
- busid ：bus总线号为0000:40:00.0
- host ：主机名称sse-lab-inspur-043
- minor_number：设备号即卡号等于 2
- name：加速卡设备型号为Enflame T10
- pod_name：加速卡所在的k8s pod 名称 为 pod-gcu-example
- pod_namespace: 加速卡pod所在k8s命名空间为enflame
- slot：设备卡槽为PCIE3_GPU
- uuid : 加速卡uuid 为 GCU-U53000080105

### enflame_gcu_power_consumption

#### 指标说明

- enflame_gcu_power_consumption：GCU 加速卡的当前电源消耗量

#### 标签说明

- busid ：bus总线号
- host ：主机名称
- minor_number：设备号，卡号
- name：加速卡设备型号
- pod_name：加速卡所在的k8s pod 名称
- pod_namespace: 加速卡pod所在k8s命名空间
- slot：设备槽号
- uuid : 加速卡uuid

#### 示例

 ```
enflame_gcu_power_consumption{busid="0000:40:00.0",host="sse-lab-inspur-043",minor_number="2",name="Enflame T10",pod_name="pod-gcu-example",pod_namespace="enflame",slot="PCIE3_GPU",uuid="GCU-U53000080105"} 56.0
 ```

- enflame_gcu_power_consumption值56.0，表明GCU加速卡当前电源消耗量为56.0W
- busid ：bus总线号为0000:40:00.0
- host ：主机名称sse-lab-inspur-043
- minor_number：设备号即卡号等于 2
- name：加速卡设备型号为Enflame T10
- pod_name：加速卡所在的k8s pod 名称 为 pod-gcu-example
- pod_namespace: 加速卡pod所在k8s命名空间为enflame
- slot：设备卡槽为PCIE3_GPU
- uuid : 加速卡uuid 为 GCU-U53000080105

### enflame_gcu_power_capability

#### 指标说明

- enflame_gcu_power_capability：GCU 加速卡的电源总功耗量

#### 标签说明

- busid ：bus总线号
- host ：主机名称
- minor_number：设备号，卡号
- name：加速卡设备型号
- pod_name：加速卡所在的k8s pod 名称
- pod_namespace: 加速卡pod所在k8s命名空间
- slot：设备槽号
- uuid : 加速卡uuid

#### 示例

 ```
enflame_gcu_power_capability{busid="0000:40:00.0",host="sse-lab-inspur-043",minor_number="2",name="Enflame T10",pod_name="pod-gcu-example",pod_namespace="enflame",slot="PCIE3_GPU",uuid="GCU-U53000080105"} 225
 ```

- enflame_gcu_power_capability值225，表明GCU加速卡当前电源总功耗为225W
- busid ：bus总线号为0000:40:00.0
- host ：主机名称sse-lab-inspur-043
- minor_number：设备号即卡号等于 2
- name：加速卡设备型号为Enflame T10
- pod_name：加速卡所在的k8s pod 名称 为 pod-gcu-example
- pod_namespace: 加速卡pod所在k8s命名空间为enflame
- slot：设备卡槽为PCIE3_GPU
- uuid : 加速卡uuid 为 GCU-U53000080105


### enflame_gcu_health

#### 指标说明

- enflame_gcu_health：GCU 加速卡健康状态， 值为2时表示healthy状态，值为1时表示unhealthy状态，值为0时表示unknown状态

#### 标签说明

- busid ：bus总线号
- host ：主机名称
- minor_number：设备号，卡号
- name：加速卡设备型号
- pod_name：加速卡所在的k8s pod 名称
- pod_namespace: 加速卡pod所在k8s命名空间
- slot：设备槽号
- uuid : 加速卡uuid
- healthmsg: GCU加速卡状态信息

#### 示例

 ```
enflame_gcu_health{busid="0000:40:00.0",healthmsg="Healthy",host="sse-lab-inspur-043",minor_number="2",name="Enflame T10",pod_name="pod-gcu-example",pod_namespace="enflame",slot="PCIE3_GPU",uuid="GCU-U53000080105"} 2
 ```

- enflame_gcu_health值2，表明当前GCU加速卡的处于Healthy状态
- busid ：bus总线号为0000:40:00.0
- host ：主机名称sse-lab-inspur-043
- minor_number：设备号即卡号等于 2
- name：加速卡设备型号为Enflame T10
- pod_name：加速卡所在的k8s pod 名称 为 pod-gcu-example
- pod_namespace: 加速卡pod所在k8s命名空间为enflame
- slot：设备卡槽为PCIE3_GPU
- uuid : 加速卡uuid 为 GCU-U53000080105
- healthmsg: 值为Healthy代表GCU加速卡处于健康状态

### enflame_gcu_ecc_double_bit_error_total_count

#### 指标说明

- enflame_gcu_ecc_double_bit_error_total_count：GCU 加速卡ECC DBE 累计总数，DBE 会触发加速卡内存 row remapping RAS机制进行修复，因此这个指标不建议作为告警参考。

#### 标签说明

- busid ：bus总线号
- host ：主机名称
- metrics：指标
- minor_number：设备号，卡号
- name：加速卡设备型号
- slot：设备槽号
- uuid : 加速卡uuid

#### 示例

 ```
enflame_gcu_ecc_double_bit_error_total_count{busid="0000:40:00.0",host="sse-lab-inspur-043",metrics="ecc_double_bit_error_total_count",minor_number="2",name="Enflame T10",slot="PCIE3_GPU",uuid="GCU-U53000080105"} 0
 ```

- enflame_gcu_ecc_double_bit_error_total_count值0，表明当前GCU加速卡的ECC DBE总数为0
- busid ：bus总线号为0000:40:00.0
- host ：主机名称sse-lab-inspur-043
- metrics：指标为ecc_double_bit_error_total_count
- minor_number：设备号即卡号等于 2
- name：加速卡设备型号为Enflame T10
- slot：设备卡槽为PCIE3_GPU
- uuid : 加速卡uuid 为 GCU-U53000080105

### enflame_gcu_ecc_single_bit_error_total_count

#### 指标说明

- enflame_gcu_ecc_single_bit_error_total_count：GCU 加速卡ECC SBE 累计总数

#### 标签说明

- busid ：bus总线号
- host ：主机名称
- metrics：指标
- minor_number：设备号，卡号
- name：加速卡设备型号
- slot：设备槽号
- uuid : 加速卡uuid

#### 示例

 ```
enflame_gcu_ecc_single_bit_error_total_count{busid="0000:40:00.0",host="sse-lab-inspur-043",metrics="ecc_single_bit_error_total_count",minor_number="2",name="Enflame T10",slot="PCIE3_GPU",uuid="GCU-U53000080105"} 0
 ```

- enflame_gcu_ecc_single_bit_error_total_count值0，表明当前GCU加速卡的ECC SBE 总数为0
- busid ：bus总线号为0000:40:00.0
- host ：主机名称sse-lab-inspur-043
- metrics：指标为ecc_single_bit_error_total_count
- minor_number：设备号即卡号等于 2
- name：加速卡设备型号为Enflame T10
- slot：设备卡槽为PCIE3_GPU
- uuid : 加速卡uuid 为 GCU-U53000080105

### enflame_gcu_pcie_link_width

#### 指标说明

- enflame_gcu_pcie_link_width：GCU 加速卡的PCIe link width

#### 标签说明

- busid ：bus总线号
- host ：主机名称
- metrics：指标
- minor_number：设备号，卡号
- name：加速卡设备型号
- pod_name：加速卡所在的k8s pod 名称
- pod_namespace: 加速卡pod所在k8s命名空间
- slot：设备槽号
- uuid : 加速卡uuid

#### 示例

 ```
enflame_gcu_pcie_link_width{busid="0000:40:00.0",host="sse-lab-inspur-043",metrics="pcie_link_width_x",minor_number="2",name="Enflame T10",pod_name="pod-gcu-example",pod_namespace="enflame",slot="PCIE3_GPU",uuid="GCU-U53000080105"} 16
 ```

- enflame_gcu_pcie_link_width值16，表明当前GCU加速卡的pcie link width 为 16
- busid ：bus总线号为0000:40:00.0
- host ：主机名称sse-lab-inspur-043
- metrics：指标为pcie_link_width_x
- minor_number：设备号即卡号等于 2
- name：加速卡设备型号为Enflame T10
- pod_name：加速卡所在的k8s pod 名称 为 pod-gcu-example
- pod_namespace: 加速卡pod所在k8s命名空间为enflame
- slot：设备卡槽为PCIE3_GPU
- uuid : 加速卡uuid 为 GCU-U53000080105

### enflame_gcu_pcie_max_link_width

#### 指标说明

- enflame_gcu_pcie_max_link_width：GCU 加速卡的PCIe max link width

#### 标签说明

- busid ：bus总线号
- host ：主机名称
- metrics：指标
- minor_number：设备号，卡号
- name：加速卡设备型号
- pod_name：加速卡所在的k8s pod 名称
- pod_namespace: 加速卡pod所在k8s命名空间
- slot：设备槽号
- uuid : 加速卡uuid

#### 示例

 ```
enflame_gcu_pcie_max_link_width{busid="0000:40:00.0",host="sse-lab-inspur-043",metrics="pcie_max_link_width_x",minor_number="2",name="Enflame T10",pod_name="pod-gcu-example",pod_namespace="enflame",slot="PCIE3_GPU",uuid="GCU-U53000080105"} 16
 ```

- enflame_gcu_pcie_max_link_width值16，表明当前GCU加速卡的pcie max link width 为 16
- busid ：bus总线号为0000:40:00.0
- metrics：指标为pcie_max_link_width_x
- host ：主机名称sse-lab-inspur-043
- minor_number：设备号即卡号等于 2
- name：加速卡设备型号为Enflame T10
- pod_name：加速卡所在的k8s pod 名称 为 pod-gcu-example
- pod_namespace: 加速卡pod所在k8s命名空间为enflame
- slot：设备卡槽为PCIE3_GPU
- uuid : 加速卡uuid 为 GCU-U53000080105


### enflame_gcu_cluster_usage

#### 指标说明

- enflame_gcu_cluster_usage：GCU 片上Cluster 利用率

#### 标签说明

- busid ：bus总线号
- host ：主机名称
- cluster： GCU 片上cluser号
- metrics: 指标名称
- minor_number：设备号，当前 GCU cluster所属GCU卡号
- name：加速卡设备型号
- pod_name：加速卡所在的k8s pod 名称
- pod_namespace: 加速卡pod所在k8s命名空间
- slot：设备槽号
- uuid : 加速卡uuid

#### 示例

 ```
enflame_gcu_cluster_usage{busid="0000:40:00.0",cluster="0",host="sse-lab-inspur-043",metrics="cluster_usage",minor_number="2",name="Enflame T10",pod_name="pod-gcu-example",pod_namespace="enflame",slot="PCIE3_GPU",uuid="GCU-U53000080105"} 80
 ```

- enflame_gcu_cluster_usage 值 80，表明GCU Cluster当前利用率 80%
- busid：Bus总线号为 0000:40:00.0
- host ：主机名称sse-lab-inspur-043
- cluster： GCU 片上cluser号 为 0
- metrics： 指标名称 cluster_usage
- minor_number：设备号即卡号等于 2
- name：加速卡设备型号为Enflame T10
- pod_name：加速卡所在的k8s pod 名称 为 pod-gcu-example
- pod_namespace: 加速卡pod所在k8s命名空间为enflame
- slot：设备卡槽为PCIE3_GPU
- uuid : 加速卡uuid 为 GCU-U53000080105


### enflame_gcu_pg_usage(仅限i系列推理卡）

#### 指标说明

- enflame_gcu_pg_usage：GCU 片上PG 利用率

#### 标签说明

- busid ：bus总线号
- host ：主机名称
- pg： GCU 片上pg号
- metrics: 指标名称
- minor_number：设备号，当前 GCU pg所属GCU卡号
- name：加速卡设备型号
- pod_name：加速卡所在的k8s pod 名称
- pod_namespace: 加速卡pod所在k8s命名空间
- slot：设备槽号
- uuid : 加速卡uuid

#### 示例

 ```
enflame_gcu_pg_usage{busid="0000:40:00.0",pg="0",host="sse-lab-inspur-043",metrics="pg_usage",minor_number="2",name="Enflame I20",pod_name="pod-gcu-example",pod_namespace="enflame",slot="PCIE3_GPU",uuid="GCU-U53000080105"} 80
 ```

- enflame_gcu_pg_usage 值 80，表明GCU PG当前利用率 80%
- busid：Bus总线号为 0000:40:00.0
- host ：主机名称sse-lab-inspur-043
- pg： GCU 片上cluser号 为 0
- metrics： 指标名称 pg_usage
- minor_number：设备号即卡号等于 2
- name：加速卡设备型号为Enflame I20
- pod_name：加速卡所在的k8s pod 名称 为 pod-gcu-example
- pod_namespace: 加速卡pod所在k8s命名空间为enflame
- slot：设备卡槽为PCIE3_GPU
- uuid : 加速卡uuid 为 GCU-U53000080105


## 告警与处理

#### enflame_gcu_health == 1，致命告警

- 告警说明：enflame_gcu_health == 1表示 gcu处于非健康状态，enflame_gcu_health 值为2时表示healthy状态，值为1时表示unhealthy状态，值为0时表示unknown状态
- 处理策略：当前触发unhealthy，说明加速卡触发RMA故障，建议走RMA流程，当前unhealthy 只依赖于RMA 标识，其他可自动修复（比如hot reset 修复）的告警指标不作为unhealthy状态，因此unhealthy 不大会触发误报。

#### enflame_gcu_temperatures > 95，严重告警

- 告警说明：enflame_gcu_temperatures > 95， 表示当前gcu卡子的温度超过95°，温度过高，这会触发自动降频行为
- 处理策略：检查设备的降温情况

#### enflame_gcu_count < 正常值， 严重告警

- 告警说明：enflame_gcu_count < 正常值，比如enflame_gcu_count < 8，服务器里的 gcu可能掉卡，卡子个数要根据实际的服务器配置情况调整
- 处理策略：采用其他功能更强的工具，比如efsmi 检查服务器内的加速卡个数，再根据具体情况进行处理

#### enflame_gcu_pcie_link_width <  enflame_gcu_pcie_max_link_width，严重告警

- 告警说明：enflame_gcu_pcie_link_width <  enflame_gcu_pcie_max_link_width，服务器里的 gcu pcie link width 小于最大值，比如pcie max link width 为 x16， 而 pcie link width 值却为 x8，这表明加速卡还能用，但是影响性能

- 处理策略：先检查BIOS配置里的是否正常，也看看能否用工具修复，如果修复不了，可能触发硬件故障，建议深入检查

  

## 部署

### 裸机部署

首先确定机器上已经安装efsmi，然后再直接执行 `./gcu-exporter`，步骤如下：

```
# ./gcu-exporter
Starting HTTP server on :9400

```

在gcu-exporter服务拉起后，在浏览器上输入 本机 ${IP}:9400 即可看到相应指标。

注：

`gcu-exporter` 获取 slot信息依赖于`dmidecode`，如果主机系统里没安装`dmidecode`，需要先装上，例如：

```
# apt-get update && apt-get install -y dmidecode
```



### Docker部署

首先在安装包目录下执行 `docker-image-build.sh`构建 `gcu-exporter:latest`镜像，步骤如下：

```
gcu-exporter_x.x.x# ls
docker  docker-image-build.sh  gcu-exporter  libefml  LICENSE.md  README.md  yaml
#./docker-image-build.sh

Sending build context to Docker daemon  45.14MB
Step 1/5 : FROM ubuntu:18.04
 ---> 35b3f4f76a24
Step 2/5 : RUN apt-get update && apt-get install -y dmidecode
 ---> Using cache
 ---> ea0682fd7490
Step 3/5 : COPY gcu-exporter /usr/bin/
 ---> bb6436b218c5
Step 4/5 : EXPOSE 9400
 ---> Running in e18ca253bf61
Removing intermediate container e18ca253bf61
 ---> 2772b415bec9
Step 5/5 : ENTRYPOINT ["/usr/bin/gcu-exporter"]
 ---> Running in f2218ec5be65
Removing intermediate container f2218ec5be65
 ---> 0c26afa1171f
Successfully built 0c26afa1171f
Successfully tagged artifact.enflame.cn/enflame_docker_images/enflame/gcu-exporter:latest
```

在构建好镜像后，再直接执行 `docker run` 拉起`gcu-exporter`服务，例如：

```
# docker run  --privileged -d --name=gcu-exporter -v /usr/local/efsmi:/usr/local/efsmi -v /usr/lib/libefml.so:/usr/lib/libefml.so -v /var/lock:/var/lock -v /etc/localtime:/etc/localtime -v /sys:/sys -p 9400:9400 --network host artifact.enflame.cn/enflame_docker_images/enflame/gcu-exporter:latest

如果是 ARM 平台, 若 /usr/lib/libefml.so 不存在，但 /usr/lib/libefml-arm.so 存在，则需执行以下步骤：
# docker run  --privileged -d --name=gcu-exporter -v /usr/local/efsmi:/usr/local/efsmi -v /usr/lib/libefml-arm.so:/usr/lib/libefml.so -v /var/lock:/var/lock -v /etc/localtime:/etc/localtime -v /sys:/sys -p 9400:9400 --network host artifact.enflame.cn/enflame_docker_images/enflame/gcu-exporter:latest
```

如此，在gcu-exporter服务拉起后，在浏览器上输入 本机 ${IP}:9400 即可看到相应指标。



注：

1) `gcu-exporter` 获取 slot信息依赖于`dmidecode`，如果构建的镜像里没安装`dmidecode`，需要先装上，例如在`docker/Dockerfile.ubuntu`里添加：

```
RUN apt-get update && apt-get install -y dmidecode
```
2) `docker-image-build.sh` 里的`gcu-exporter`镜像路径以及名称需要根据实际情况修改或定制


### K8S部署

假设k8s已经部署好，那么需要先在安装包目录下执行 `docker-image-build.sh`构建 `gcu-exporter:latest`镜像，步骤如下：

```
gcu-exporter_x.x.x# ls
docker  docker-image-build.sh  gcu-exporter  libefml  LICENSE.md  README.md  yaml
#./docker-image-build.sh
Sending build context to Docker daemon  45.14MB
Step 1/5 : FROM ubuntu:18.04
 ---> 35b3f4f76a24
Step 2/5 : RUN apt-get update && apt-get install -y dmidecode
 ---> Using cache
 ---> ea0682fd7490
Step 3/5 : COPY gcu-exporter /usr/bin/
 ---> bb6436b218c5
Step 4/5 : EXPOSE 9400
 ---> Running in e18ca253bf61
Removing intermediate container e18ca253bf61
 ---> 2772b415bec9
Step 5/5 : ENTRYPOINT ["/usr/bin/gcu-exporter"]
 ---> Running in f2218ec5be65
Removing intermediate container f2218ec5be65
 ---> 0c26afa1171f
Successfully built 0c26afa1171f
Successfully tagged artifact.enflame.cn/enflame_docker_images/enflame/gcu-exporter:latest
```

然后再直接执行 `kubectl apply` 拉起`gcu-exporter`服务，例如：
```
#kubectl apply -f yaml/gcu-exporter.yaml

```
在gcu-exporter服务拉起后，在浏览器上输入 本机 ${IP}:9400 即可看到相应指标。



其中`gcu-exporter` 获取 slot信息依赖于`dmidecode`，如果构建的镜像里没安装`dmidecode`，需要先装上，例如在`docker/Dockerfile.ubuntu`里添加：

```
RUN apt-get update && apt-get install -y dmidecode
```



## 常见问题

### gcu-exporter 与 enflame-exporter的差异

- `gcu-exporter` 对标的企业级版本，  更规范、更完善的指标 ；
- `enflame-exporter` 对标的开源版本，满足不同的用户需求；

### gcu-exporter 是基于什么库实现的

`gcu-exporter` 基于 `go-eflib` 实现，然后`go-eflib` 调用的 `libefml.so`，而`libefml.so` 是从`efsmi `安装包里头获取的。
如果是ARM平台，则是 `libefml-arm.so`，如果不存在 `/usr/lib/libefml.so`，需要保持名称兼容：`ln -sf /usr/lib/libefml-arm.so /usr/lib/libefml.so`。


### 出现 Oops! log，要求重启gcu-exporter服务

出现类似以下的oops log，这是 gcu-exporter里头调用`libefml.so`读取到的GCU个数与从enflame driver里读取到的GCU个数不匹配（比如测试场景 ：在OS里手工 remove 掉一个卡：`echo 1 > /sys/bus/pci/devices/0000\:3e\:00.0/remove`）。

```
Oops! Check device count error:xxxx, the exporter needs to be restarted
```

为了避免访问不存在的gcu 设备造成未知的异常问题，这里 打印出一个 oops信息，在5.3 k8s部署场景下，gcu-exporter 是一个daemon pod会被k8s自己重新拉起。而在5.1裸机部署以及5.2 docker部署场景下，需要手工再拉起 gcu-exporter服务，当然 docker 也可以采用`--restart=always`参数来达成自动拉起gcu-exporter服务。

### gcu-exporter 遇到libefml初始化失败问题，如何解决
该问题一般是gcu-exporter内置的libefml与当前宿主机上的kmd版本不兼容造成的，可以通过挂载宿主机的libefml解决,具体配置见gcu-exporter-with-host-libefml.yaml.当前gcu-exporter默认采用主机的Libefml.so， 如果采用容器挂载 gcu-exporter的方式，需要 将Libefml.so 挂载进容器内。

### 关于版本号

版本号遵循的是 `MAJOR.MINOR.BUILT` 的格式：
1）MAJOR 版本发生改变，表示接口发生变化；
2）MINOR 版本发生变化，表示新增了特性或功能；
3）BUILT 版本发生变化，表示安装包编译日期发生变化或进行了BUG修复。






