# mindmpc
The **real** MPC-AI framework based on Mindspore
- [mindmpc](#mindmpc)
  - [协议API](#协议api)
    - [**dispatch**](#dispatch)
    - [**check_open**](#check_open)
    - [**composite**](#composite)

## 协议API
### **dispatch**

用于生成份额
>method dispatch(vlaue:IntTensor)

使用value来生成分享份额

**Parameter**
<br> *value* 输入的数据，需要是经过编码的数据

**Returns**
<br> Tuple， `(x_1, [x_2, x_3])` x<sub>1</sub> 为当前用户自己持有的share， x<sub>2</sub>和x<sub>3</sub>为将要发送给其他用户的share

### **check_open**

用于检查份额是否可以打开
>method check_open(share0, share1):

**Parameter**
<br> *share0* 自己的share
<br> *share1:[list]* 其他用户share的列表 


**Returns**
<br> Boolean， 返回是否符合打开要求

### **composite**

用于份额重构成
>method composite(share0, share1)

与`dispatch`对应的生成明文

**Parameter**
<br> *share0* 自己的share
<br> *share1:[list]* 其他用户share的列表 

**Returns**
<br> IntTensor， 输出明文的编码Tensor

