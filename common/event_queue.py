#使用event来确保多线程的顺序执行

import threading
class Message:
    def __init__(self):
        self.event = threading.Event()
    def lock(self):
        self.event.clear()
    def unlock(self):
        self.event.set()
    def stand(self):
        self.event.wait()

class MessageOnceQue:
    def __init__(self):
        self.que = {}
    def set_ele(self,name):
        if name not in self.que:
            self.que[name] = Message()
        return self.get_ele(name) 
    def get_ele(self,name):
        return self.que[name]

ins_messs_que = MessageOnceQue()
add_share_que = MessageOnceQue()