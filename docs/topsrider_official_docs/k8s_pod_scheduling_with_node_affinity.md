
## 版本申明

| 版本   | 修改内容 | 修改时间      |
|------|------|-----------|
| v1.0 | 初始化  | 6/10/2022 |
|      |      |           |


## 利用节点亲和性选择特定标签的节点

### 为pod配置节点亲和性
如题，将pod调度到具有特定标签的节点，有node selector和Node Affinity两种方式可以做到，考虑到node selector过于简单粗暴，并且会逐步被废弃，这里使用节点亲和性（node affinity）来实现
我们需要在pod对象下的规约（spec）下添加亲和性描述，下面是一个描述pod 对象的yaml文件，可以看到我们我们在pod的spec下增加了亲和性描述

```yaml

apiVersion: v1
kind: Pod
metadata:
name: pod-gcu-example
namespace: enflame
spec:
#  restartPolicy: OnFailure
affinity:    #放在spec下，用于描述该对象所具备的亲和性属性
nodeAffinity:   #用于进一步描述该亲和性特性为节点亲和性
#DuringSchedulingIgnoredDuringExecution意思是仅仅在调度时生效，如果pod已经调度到节点上，则该亲和性规则不生效
preferredDuringSchedulingIgnoredDuringExecution:  #节点亲和性有两种模式，软限制preferred和硬限制required，软限制适用于部分未打标签的场景
nodeSelectorTerms: #开始描述节点选择模式，如果一个nodeAffinity下有多个nodeSelectorTerm，只需要满足其中一个就可以
  - matchExpressions:    #描述匹配表达式，如果一个nodeSelectorTerms下有多个matchExpressions，需要满足所有matchExpressions才生效
  - key: enflame.com/gcu_type   #下面四行为匹配表达式的详细内容，支持In， Not In，Exist，Not Exist，Gt， Lt六种操作符
operator: In
values:
  - t20i
hostNetwork: true
containers:
  - name: pod-gcu-example
    image: ubuntu:18.04
    imagePullPolicy: IfNotPresent
    command: [ "sleep" ]
    args: [ "100000" ]
    securityContext:
      #      capabilities:
      #        add: [ "IPC_LOCK" ]
      #    privileged: true
      resources:
        limits:
          enflame.com/gcu: 8
      #        rdma/hca: 1
      volumeMounts:
        - name: sshpath
          mountPath: /root/.ssh
          readOnly: false
        - name: homepath
          mountPath: /home
          readOnly: false
volumes:
  - name: sshpath
    hostPath:
    path: "/root/.ssh"
    type: Directory
  - name: homepath
    hostPath:
    path: "/home"
    type: Directory


```

### 为deployment配置节点亲和性
考虑到我们一般使用controller对象托管pod，这里给出一个deployment对象的yaml描述文件,类同于上面的pod描述文件，我们需要在pod template的spec里添加节点亲和性描述

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gcu-deployment
  labels:
    app: gcu
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gcu
  template:
    metadata:
      labels:
        app: gcu
    spec:
      affinity: #放在spec下，用于描述该对象所具备的亲和性属性
        nodeAffinity: #用于进一步描述该亲和性特性为节点亲和性
          requiredDuringSchedulingIgnoredDuringExecution: #节点亲和性的两种模式之一，这里指仅仅在调度时生效，如果pod应调度到节点上，则该亲和性规则不生效
            nodeSelectorTerms: #开始描述节点选择模式，如果一个nodeAffinity下有多个nodeSelectorTerm，只需要满足其中一个就可以
              - matchExpressions: #描述匹配表达式，如果一个nodeSelectorTerms下有多个matchExpressions，需要满足所有matchExpressions才生效
                  - key: enflame.com/gcu_type   #下面四行为匹配表达式的详细内容，支持In， Not In，Exist，Not Exist，Gt， Lt六种操作符
                    operator: In
                    values:
                      - t20i
      hostNetwork: true
      containers:
        - name: pod-gcu-example
          image: ubuntu:18.04
          imagePullPolicy: IfNotPresent
          command: [ "sleep" ]
          args: [ "100000" ]
          securityContext:
          resources:
            limits:
              enflame.com/gcu: 8
          volumeMounts:
            - name: sshpath
              mountPath: /root/.ssh
              readOnly: false
            - name: homepath
              mountPath: /home
              readOnly: false
      volumes:
        - name: sshpath
          hostPath:
          path: "/root/.ssh"
          type: Directory
        - name: homepath
          hostPath:
          path: "/home"
          type: Directory




```

我们利用一个两节点单master集群进行实验（k8s版本v1.19.9）：

假设t10i的卡所在node，我们打上标签enflame.com/gcu_type=t10i

假设t20i的卡所在node，我们打上标签enflame.com/gcu_type=t20i

当values为t20i时

我们通过kubectl get po -A -o wide可以看到，pod准确落在node2

当values为t10i时

我们通过kubectl get po -A -o wide可以看到，pod准确落在node1
## 如何给节点打上标签

### 命令行方式

```shell
kubectl label nodes k8s-node1 enflame.com/gcu_type=t10i #为node加上标签
kubectl label --overwrite nodes k8s-node1 enflame.com/gcu_type=t10i #改写node标签
kubectl label nodes --all enflame.com/gcu_type=t10i #为所有node加上标签
kubectl label nodes k8s-node1 enflame.com/gcu_type=t10i- #为node去除标签

```

### yaml文件方式

```yaml
kind: Node
apiVersion: v1
metadata:
  name: k8s-node1
  labels:
    enflame.com/gcu_type: t10i
```

执行kubectl apply -f node.yaml即可进行标签的添加和覆盖


