# mindmpc
The **real** MPC-AI framework based on Mindspore
- [mindmpc](#mindmpc)
  - [Tensor](#tensor)
    - [**IntTensor**](#inttensor)
    - [**PrivateTensor**](#privatetensor)
      - [**convert_public**](#convert_public)
      - [**share**](#share)
  - [协议API](#协议api)
    - [**dispatch**](#dispatch)
    - [**check_open**](#check_open)
    - [**composite**](#composite)
    - [**Add**](#add)
    - [**Add_cons**](#add_cons)
    - [**Mul**](#mul)
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
### **dispatch**

用于生成份额
>**method** dispatch(vlaue:IntTensor)

使用value来生成分享份额

**Parameter**
<br> *value:* 输入的数据，需要是经过编码的数据

**Returns**
<br> *Tuple:*， `(x_1, [x_2, x_3])` x<sub>1</sub> 为当前用户自己持有的share， x<sub>2</sub>和x<sub>3</sub>为将要发送给其他用户的share

-----

### **check_open**

用于检查份额是否可以打开
>method check_open(share0, share1):

**Parameter**
<br> *share0:* 自己的share
<br> *share1:[list]:* 其他用户share的列表 


**Returns**
<br> Boolean， 返回是否符合打开要求

-----

### **composite**

用于份额重构成
>**method** composite(share0, share1)

与`dispatch`对应的生成明文

**Parameter**
<br> *share0:* 自己的share
<br> *share1:[list]:* 其他用户share的列表 

**Returns**
<br> IntTensor， 输出明文的编码Tensor

-----
### **Add**

计算[x+y]
>**method** Add(x:Placeholder,y:Placeholder,z:Placeholder = None):

返回z以确保调用的连贯

**Parameter**
<br> *x:* 一个需要确保已经被填入私有Tensor的占位符
<br> *y:* 一个需要确保已经被填入私有Tensor的占位符
<br> *z:* 一个用于接收结果的占位符，不需要有对应的私有Tensor

**Returns**
<br> Placeholder， 返回计算结果的占位符

-----

### **Add_cons**

计算[x] + a,计算份额与常数的加法
>**method** Add_cons(x:Placeholder,y:IntTensor)



**Parameter**
<br> *x:* 一个需要确保已经被填入私有Tensor的占位符
<br> *y:* 需要加上的明文数据

**Returns**
<br> Placeholder， 返回计算结果的占位符

-----

### **Mul**

计算[x] × [y],计算份额的乘法
>**method** Mul(x:Placeholder,y:Placeholder,z:Placeholder, triple = None)


**Parameter**
<br> *x:* 一个需要确保已经被填入私有Tensor的占位符
<br> *y:* 一个需要确保已经被填入私有Tensor的占位符
<br> *z:* 一个用于接收结果的占位符，不需要有对应的私有Tensor
<br> *triple:* 用于计算的Triple 如果有的话需要为占位符的列表（如[a,b,c]若为乘法triples 则其占位符的私有Tensor对应[a*b] = [c]）


**Returns**
<br> Placeholder， 返回计算结果的占位符