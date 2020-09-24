import sys
import time


from p2pnetwork.node import Node

def node_callback(event, main_node, connected_node, data):
    try:
        if event != 'node_request_to_stop': # node_request_to_stop does not have any connected_node, while it is the main_node that is stopping!
            print('Event: {} from main node {}: connected node {}: {}'.format(event, main_node.id, connected_node.id, data))

    except Exception as e:
        print(e)


node_1 = Node("127.0.0.1", 8001, node_callback)
node_2 = Node("127.0.0.1", 8002, node_callback)
node_3 = Node("127.0.0.1", 8003, node_callback)

time.sleep(1)
#node_1.debug = True
#node_2.debug = True
#node_3.debug = True
node_1.start()
node_2.start()
node_3.start()
time.sleep(1)

node_1.connect_with_node('127.0.0.1', 8002)
node_2.connect_with_node('127.0.0.1', 8003)
node_3.connect_with_node('127.0.0.1', 8001)

time.sleep(2)

node_1.send_to_nodes({"esion":100})

time.sleep(5)

node_1.stop()
node_2.stop()
node_3.stop()

print('end')