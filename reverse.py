"""

将froze过的模型文件进行optimize之后 得出的模型节点网络顺序不一定符合拓扑排序
所以将模型加载再保存一遍得出顺序遍历的模型文件

"""


import tensorflow as tf
from tensorflow.python.platform import gfile
import os


pb_file_path = os.getcwd() + '/models/mobilenet_v2_optimize.pb'

sess = tf.Session()
sess.run(tf.global_variables_initializer())

with gfile.FastGFile(pb_file_path,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def,name='')
    input_graph_def = sess.graph.as_graph_def()

# output_nodes = ["MobilenetV2/Predictions/Reshape_1"]
output_nodes = ["MobilenetV2/Predictions/Softmax"]
output_graph_def = tf.graph_util.convert_variables_to_constants(sess,input_graph_def,output_nodes)

with open("models/self_reverse_frozen.pb","wb") as f:
    f.write(output_graph_def.SerializeToString())
