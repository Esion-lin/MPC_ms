- [Tensor](#tensor)
  - [**IntTensor**](#inttensor)
  - [**PrivateTensor**](#privatetensor)
    - [**convert_public**](#convert_public)
    - [**share**](#share)
- [wrap_function(PlayerDecorator)](#wrap_functionplayerdecorator)
    - [**from_**](#from_)
    - [**to_**](#to_)
    - [**open_**](#open_)

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


## wrap_function(PlayerDecorator)

#### **from_**

令指定用户运行代码

>**method** from_(func = None, player_name = "")

**Parameter**
<br> *func:* 指定用户运行的函数
<br> *player_name:* 指定的用户名

**Returns**
<br> *func（）:* 返回`func`运行的结果


----

#### **to_**

令指定用户将变量池中的数据分发给目标用户

>**method** to_(func = None, player_name = "", target = "", var_name = "x")

**Parameter**
<br> *func:* 指定用户运行的函数
<br> *player_name:* 指定的用户名，若为空字符串则默认为config列表中的所有用户
<br> *target:* 目标的用户名，若为空字符串则默认为config列表中的所有用户、
<br> *var_name:* 变量名，需要接收的用户记住的变量名

**Returns**
<br> *Placeholder:* 返回分发变量的Placeholder类型结果

**Sample**

```python
@dec.to_(player_name = player_name, var_name = var_name)
def input():
  get_var_pool()[var_name] = ptensor
  return ptensor.share()
```

如上诉代码所示，使player_name将ptensor.share()的结果分发给所有用户，其中ptensor.share()的维度需要与所有用户的数量-1一致

----

#### **open_**

向指定用户派发数据

>**method** to_(func = None, player_name = "", target = "", var_name = "x")

**Parameter**
<br> *func:* 用户收集完数据需要运行的函数，如open等操作
<br> *player_name:* 指定的用户名，若为空字符串则默认为config列表中的所有用户
<br> *target:* 需要向指定用户发送数据的目标用户，若为空字符串则默认为config列表中的所有用户、
<br> *var_name:* 变量名，需要指定用户发送的变量名

**Returns**
<br> *Placeholder:* 返回分发变量的Placeholder类型结果

**Sample**
```python
@dec.open_(player_name = player_name, var_name = var_name)
def open():
  return get_var_pool()[var_name].open()
```
如上诉代码所示，使所有用户将var_name的变量发送给player_name的用户，结束后执行open()函数。

----