import numpy as np
import tensorflow as tf
import os
from tensorflow.python.platform import gfile
import cv2

pb_file_path = os.getcwd() + '/gpu_frozen_8dim.pb'

sess = tf.Session()
sess.run(tf.global_variables_initializer())

with gfile.FastGFile(pb_file_path,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def,name='')


image_array_2  = cv2.imread(os.getcwd()+"/MNIST_data/raw/1.jpg",0)
image_array_2  = image_array_2.reshape(1,784)

input_x = sess.graph.get_tensor_by_name("Placeholder:0")
softmax = sess.graph.get_tensor_by_name("Softmax:0")

out_2 = sess.run(softmax,{input_x:image_array_2})
prediction_label_2 = np.argmax(out_2)

print("==================================")
print("predict label:",prediction_label_2)

print("==================================")


### Reshape  Input
# Reshape = sess.graph.get_tensor_by_name("Reshape:0")
# Reshape_out_2 = sess.run(Reshape,{input_x:image_array_2})

# print("==================================")
# blob_784_2 = Reshape_out_2.reshape(784)

# for i in range(784):
#     print(i," : ",blob_784_2[i]);


### Conv2D convolution
Conv2D = sess.graph.get_tensor_by_name("Relu:0") # for relu is merge the conv2D layer
Conv2D_out = sess.run(Conv2D,{input_x:image_array_2})
Conv2D_transfer_out = [] # 1*28*28*32 -> 1*32*28*28
n = Conv2D_out.shape[0] #1
h = Conv2D_out.shape[1] #28 
w = Conv2D_out.shape[2] #28
c = Conv2D_out.shape[3] #32
Conv2D_out = Conv2D_out.reshape(Conv2D_out.size)
for i in range(n):
    for j in range(c):
        for p in range(h):
            for q in range(w):
                Conv2D_transfer_out.append(Conv2D_out[i*h*w*c+p*w*c+q*c+j])

file = open("./tensor_info/Conv2D.txt","w+")
for i in range (Conv2D_out.size):
    # file.writelines(i ," : " , Conv2D_out[i])
    stringData = str(i) + " : " + str(Conv2D_transfer_out[i]) +"\n";
    file.write(stringData)
file.close()


#### MaxPool pooling
MaxPool = sess.graph.get_tensor_by_name("MaxPool:0") # for relu is merge the conv2D layer
MaxPool_out = sess.run(MaxPool,{input_x:image_array_2})
MaxPool_transfer_out = [] # 1*28*28*32 -> 1*32*28*28
n = MaxPool_out.shape[0] #1
h = MaxPool_out.shape[1] #28 
w = MaxPool_out.shape[2] #28
c = MaxPool_out.shape[3] #32
MaxPool_out = MaxPool_out.reshape(MaxPool_out.size)
for i in range(n):
    for j in range(c):
        for p in range(h):
            for q in range(w):
                MaxPool_transfer_out.append(MaxPool_out[i*h*w*c+p*w*c+q*c+j])

file = open("./tensor_info/MaxPool.txt","w+")
for i in range (MaxPool_out.size):
    stringData = str(i) + " : " + str(MaxPool_transfer_out[i]) +"\n";
    file.write(stringData)
file.close()

#### Conv2D_1 convolution
Conv2D_1 = sess.graph.get_tensor_by_name("Relu_1:0") # for relu_1 is merge the conv2D_1 layer
Conv2D_1_out = sess.run(Conv2D_1,{input_x:image_array_2})
Conv2D_1_transfer_out = [] # 1*28*28*32 -> 1*32*28*28
n = Conv2D_1_out.shape[0] #1
h = Conv2D_1_out.shape[1] #14
w = Conv2D_1_out.shape[2] #14
c = Conv2D_1_out.shape[3] #50

Conv2D_1_out = Conv2D_1_out.reshape(Conv2D_1_out.size)

for i in range(n):
    for j in range(c):
        for p in range(h):
            for q in range(w):
                Conv2D_1_transfer_out.append(Conv2D_1_out[i*h*w*c+p*w*c+q*c+j])

file = open("./tensor_info/Conv2D_1.txt","w+")
for i in range (Conv2D_1_out.size):
    stringData = str(i) + " : " + str(Conv2D_1_transfer_out[i]) +"\n";
    file.write(stringData)
file.close()


#### MaxPool pooling
MaxPool_1 = sess.graph.get_tensor_by_name("MaxPool_1:0") # for relu is merge the conv2D layer
MaxPool_1_out = sess.run(MaxPool_1,{input_x:image_array_2})
MaxPool_1_transfer_out = [] # 1*28*28*32 -> 1*32*28*28
n = MaxPool_1_out.shape[0] #1
h = MaxPool_1_out.shape[1] #28 
w = MaxPool_1_out.shape[2] #28
c = MaxPool_1_out.shape[3] #32
MaxPool_1_out = MaxPool_1_out.reshape(MaxPool_1_out.size)
for i in range(n):
    for j in range(c):
        for p in range(h):
            for q in range(w):
                MaxPool_1_transfer_out.append(MaxPool_1_out[i*h*w*c+p*w*c+q*c+j])

file = open("./tensor_info/MaxPool_1.txt","w+")
for i in range (MaxPool_1_out.size):
    stringData = str(i) + " : " + str(MaxPool_1_transfer_out[i]) +"\n";
    file.write(stringData)
file.close()

### Reshape_1 reshape
Reshape_1 = sess.graph.get_tensor_by_name("Reshape_1:0")
Reshape_1_out = sess.run(Reshape_1,{input_x:image_array_2})
Reshape_1_transfer_out = [] 
n = Reshape_1_out.shape[0] #1
c = Reshape_1_out.shape[1] #3136
Reshape_1_out = Reshape_1_out.reshape(Reshape_1_out.size)
for i  in range(n):
    for j in range(c):
        Reshape_1_transfer_out.append(Reshape_1_out[i*c+j])
file = open("./tensor_info/Reshape_1.txt","w+")
for i in range (Reshape_1_out.size):
    stringData = str(i) + " : " + str(Reshape_1_transfer_out[i]) +"\n";
    file.write(stringData)
file.close()

### Matmul innerproct
MatMul = sess.graph.get_tensor_by_name("Relu_2:0")
MatMul_out = sess.run(MatMul,{input_x:image_array_2})
MatMul_transfer_out = []
n = MatMul_out.shape[0] # 1
c = MatMul_out.shape[1] #1024
MatMul_out = MatMul_out.reshape(MatMul_out.size)
# for i in range(n):
#     for j in range(c):
#         MatMul_transfer_out.append(MatMul_out[i*c+j])
for i in range(MatMul_out.size):
    MatMul_transfer_out.append(MatMul_out[i])
file = open("./tensor_info/MatMul.txt","w+")
for i in range (MatMul_out.size):
    stringData = str(i) + " : " + str(MatMul_transfer_out[i]) +"\n";
    file.write(stringData)
file.close()

### Matmul_1 innerproct
MatMul_1 = sess.graph.get_tensor_by_name("add_3:0")
MatMul_1_out = sess.run(MatMul_1,{input_x:image_array_2})
MatMul_1_transfer_out = []
n = MatMul_1_out.shape[0]
c = MatMul_1_out.shape[1]
MatMul_1_out = MatMul_1_out.reshape(MatMul_1_out.size)
for i in range(n):
    for j in range(c):
        MatMul_1_transfer_out.append(MatMul_1_out[i*c+j])
file = open("./tensor_info/MatMul_1.txt","w+")
for i in range (MatMul_1_out.size):
    stringData = str(i) + " : " + str(MatMul_1_transfer_out[i]) +"\n";
    file.write(stringData)
file.close()

# Softmax = sess.graph.get_tensor_by_name("Softmax:0")
# Softmax_out = sess.run(Softmax,{input_x:image_array_2})
# Softmax_transfer_out = []
# n = Softmax_out.shape[0]
# Softmax_out = Softmax_out.reshape(Softmax_out.size)
# for i in range (n):
#     Softmax_transfer_out.append(Softmax_out[i])
# file = open("./tensor_info/Softmax.txt","w+")
# for i in range (Softmax_out.size) :
#     stringData = str(i) + " : " + str(Softmax_transfer_out[i]) +"\n";
#     file.write(stringData)
# file.close()









