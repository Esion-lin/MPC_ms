# mindmpc
The **real** MPC-AI framework based on Mindspore
- [mindmpc](#mindmpc)
  - [TODO](#todo)
  - [模块介绍](#模块介绍)
    - [Tensor](#tensor)
    - [协议API](#协议api)
## TODO
- 网络层 
  - pipe Error 问题解决
  - ssl网络构建（身份认证的网络）
- 协议相关
  - truncation
- Mindspore相关
  - 无法使用的算子的临时替换
- 权重load和save
- OT/TEE
- grad
## 模块介绍
### Tensor
[Tensor](./common/COMMON.md)基于Mindspore的基本数据结构Tensor（张量），在此之上构建编码后的编码张量（IntTensor）最后组成私有的张量类型（PrivateTensor）

### 协议API
[协议API](./protocol/README.md) 规定了符合框架的MPC协议需要实现的接口，具体编写需求见[协议API](./protocol/README.md)