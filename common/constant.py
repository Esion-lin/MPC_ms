class _const:
    class ConstError(TypeError):pass
    def __setattr__(self,name,value):
        if name in self.__dict__:
            raise self.ConstError("Can't rebind const (%s)" %name)
        self.__dict__[name]=value

# define the network json action
ACTION = _const()
ACTION.OPEN = 0x01 
ACTION.SHARE = 0x02