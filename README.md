# mindmpc
The **real** MPC-AI framework based on Mindspore
- [mindmpc](#mindmpc)
  - [TODO](#todo)
  - [Tensor](#tensor)
    - [**IntTensor**](#inttensor)
    - [**PrivateTensor**](#privatetensor)
      - [**convert_public**](#convert_public)
      - [**share**](#share)
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

## Tensor
### **IntTensor**

经过编码的Tensor
>**class** IntTensor(internal:Boolean, tensor：Union(Tensor,list))
初始化一个编码后的Tensor

**Parameter**
<br> *internal:* 是否需要经过编码，True：已经经过编码所以不需要再次编码。
<br> *tensor:* 输入的数据可以是Tensor类型也可以是list类型


----

### **PrivateTensor**

私有的Tensor
>**class** PrivateTensor(protocol, shared:Boolean, tensor:Union(Tensor,list))

初始化一个编码后的Tensor

**Parameter**
<br> *protocol:* 私有Tensor分成份额需要调用的协议
<br> *shared:* 是否需要进行share
<br> *tensor:* 数据，可以为Tensor，list，还可以是IntTensor

* * *

#### **convert_public**
得到私有Tensor的密文

>**method** convert_public()

**Returns**
<br> IntTensor, 密文数据

----

#### **share**

返回通过协议作用后得到的需要分发给别的秘密份额。

>**method** share()

**Returns**
<br> *list:* 得到用户长度的列表，对应第i位为第i位用户的份额

----

## 协议API
见 [协议API](./protocol/README.md)