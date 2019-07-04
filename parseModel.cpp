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
    const char* tensorflowpb =  "/home/starhiking/code/onnx_saveloader/tensorflow/models/mobilenet_v2_self_frozen.pb";
    if(!read_proto_from_binary(tensorflowpb,&tf_graph))
    {
        fprintf(stderr,"read tensorflow frozen model failed.\n");
        return -1;
    }
    cout<<"read graph successfully"<<endl<<"==============================="<<endl;

    for(int i = 0;i<tf_graph.node_size();i++)
    {
        NodeDef node = tf_graph.node(i);
        cout<<node.name()<<"  "<<node.op()<<"   "<<"  "<<node.input_size()<<"  ";
        for(int j = 0;j<node.input_size();j++)
        {
            cout<<node.input(j)<<"   ";
        }
        cout<<endl<<endl;

    }
    
    cout<<"================================"<<endl;

    for(int i =0;i<tf_graph.node_size();i++)
    {
        NodeDef node = tf_graph.node(i);
        
        if(node.op()=="Identity")
        {
            node.PrintDebugString();
            // node.PrintDebugString();
            // cout<<node.attr_size()<<endl;
            google::protobuf::Map<string, AttrValue >::const_iterator it;
            it = node.attr().begin();
            while(it!=node.attr().end())    
            {
                // cout<< it->first <<endl;
                if(it->first=="value")
                {
                    cout<<node.name()<<"  "<<it->second.tensor().tensor_shape().DebugString()<<endl;

                }

                it++;
            }
      
            // break;
        }
    }

    cout<<"================================"<<endl;    
    return 0;

}
