#include <stdio.h>
#include <limits.h>

#include <iostream>

#include <fstream>
#include <set>
#include <limits>
#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>

#include "graph.pb.h"

using namespace std;
using namespace tensorflow;

static bool read_proto_from_binary(const char* filepath, google::protobuf::Message* message)
{
    std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open())
    {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);

    codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

    bool success = message->ParseFromCodedStream(&codedstr);

    fs.close();

    return success;
}

int main()
{
    tensorflow::GraphDef tf_graph;
    const char* tensorflowpb =  "/home/starhiking/code/tensorflow/models/reverse_frozen.pb";
    if(!read_proto_from_binary(tensorflowpb,&tf_graph))
    {
        fprintf(stderr,"read tensorflow frozen model failed.\n");
        return -1;
    }
    cout<<"read graph successfully"<<endl<<"==============================="<<endl;
    vector <string> node_names;
    vector <string> node_ops;
    int op_nums = 0;
    cout<<"node size : "<<tf_graph.node_size()<<endl;
    for(int i = 0;i<tf_graph.node_size();i++)
    {       
        NodeDef node = tf_graph.node(i);
        cout<<node.name()<<"  "<<node.op()<<"   "<<"  "<<node.input_size()<<"  ";
        for(int j = 0;j<node.input_size();j++)
        {
            cout<<node.input(j)<<"   ";
        }
        cout<<endl<<endl;
        node_ops.push_back(node.op());
        node_names.push_back(node.name());
        for(int j = 0;j<node.input_size();j++)
        {
            if(count(node_names.begin(),node_names.end(),node.input(j))==0)
            {
                cout<<"WARNING : "<<node.name()<< " input is not previous. with "<<node.op()<<endl;
                cout<<endl<<endl;
            }
        }
        if(false&&node.op() == "Squeeze")
        {
            node.PrintDebugString();
        }
    }
    

    cout<<"================================"<<endl;
    cout<<"node op : ";
    sort(node_ops.begin(),node_ops.end());
    auto iter = unique(node_ops.begin(),node_ops.end());
    node_ops.erase(iter,node_ops.end());
    for(size_t i =0;i<node_ops.size();i++)
    {
        cout<<node_ops[i]<<"  ";
    }
    getchar();


    cout<<"================================"<<endl;
    return 0;

}
