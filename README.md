# mindmpc
The **real** MPC-AI framework based on Mindspore
- [mindmpc](#mindmpc)
  - [框架概述](#框架概述)
  - [模块介绍](#模块介绍)
    - [Player](#player)
    - [Tensor](#tensor)
    - [变量池](#变量池)
    - [Placeholder](#placeholder)
    - [share_base 组件](#share_base-组件)
    - [可变的MPC协议](#可变的mpc协议)
    - [nn](#nn)
    - [ops](#ops)

## 框架概述
mindmpc是基于Mindspore开发的隐私机器学习计算库

## 模块介绍
### Player
[player](./player/README.md)是用户进行身份辨识，加入到计算网络的基础，用户需要声明自己的用户身份以加入到计算网络中
### Tensor
[Tensor](./common/README.md#Tensor)基于Mindspore的基本数据结构Tensor（张量），在此之上构建编码后的编码张量（IntTensor）最后组成私有的张量类型（PrivateTensor）

### 变量池
所有用户在本地维持一个变量池，相关内容见[var_pool](./common/README.md#var_pool)
### Placeholder
与变量池对应的，用户在代码层面使用的变量属于Placeholder类型，每个Placeholder的变量在变量池中都将会有对应的变量与其匹配
### share_base 组件
[share_base 组件](./common/README.md#wrap_function)用于构建share_base的MPC协议，其实现了几个用户间的交互的几个基本功能
### 可变的MPC协议
[可变的MPC协议](./protocol/README.md) 规定了符合框架的MPC协议需要实现的接口，具体编写需求见[可变的MPC协议](./protocol/README.md)

### nn
[nn](./nn/README.md)是供用户使用的核心模块，其包括了卷积池化等机器学习的api
### ops
[ops](./ops/README.md)是供用户使用的核心模块，其包括了加减乘除等基本计算api