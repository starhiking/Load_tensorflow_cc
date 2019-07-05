blob_file = open("./blob_info/MatMul_1.bin.txt")
tensor_file = open("./tensor_info/MatMul_1.txt")
write_file_name = "diff.txt"

blob_lines = blob_file.readlines()
tensor_lines = tensor_file.readlines()

if( blob_lines.__len__() != tensor_lines.__len__()):
    print("ERROR : lines is difference! blob_lines: %d , tensor_lines: %d"%(blob_lines.__len__(),tensor_lines.__len__()))
    exit()

length = blob_lines.__len__()
for i in range(length):
    # parse blob data 
    blob_index = blob_lines[i].find(':') + 1
    blob_data = float(blob_lines[i][blob_index:])
    # parse tensor data 
    tensor_index = tensor_lines[i].find(':') + 1
    tensor_data = float(tensor_lines[i][tensor_index:])
    diff_data = abs(tensor_data-blob_data)
    if(diff_data>0.001):
#     print(i," : ",blob_data," ",tensor_data)
        print("%d : %f %f"%(i,blob_data,tensor_data))
