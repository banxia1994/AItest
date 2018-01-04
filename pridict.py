#coding:utf-8
import os
import sys
sys.path.insert(0,'../caffe-face-caffe-face/python')
import numpy as np
import caffe
import json

def store (data):
    with open ('test2.json','a') as js:
        json.dump(data,js,ensure_ascii=False)
bot_data_root = './'
dir = '../data/ai_challenger_scene_test_a_20170922/test/'
# 设置网络结构
net_file = bot_data_root + './deploy_google.prototxt'
# 添加训练之后的网络权重参�?
caffe_model = bot_data_root + './snapshot/google_place365_iter_35000.caffemodel'
# 均值文�?
mean_value = 127.5
mean_file = bot_data_root + '/myVGG16/mean.npy'
# 设置使用gpu
caffe.set_mode_gpu()

# 构造一个Net
net = caffe.Net(net_file, caffe_model, caffe.TEST)
# 得到data的形状，这里的图片是默认matplotlib底层加载�?
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# matplotlib加载的image是像素[0-1],图片的数据格式[weight,high,channels]，RGB
# caffe加载的图片需要的是[0-255]像素，数据格式[channels,weight,high],BGR，那么就需要转�?
# channel 放到前面
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([mean_value,mean_value,mean_value]))
# 图片像素放大到[0-255]
#transformer.set_raw_scale('data', 255)
# RGB-->BGR 转换
transformer.set_channel_swap('data', (2, 1, 0))
#设置输入的图片shape�?张，3通道，长宽都�?24
net.blobs['data'].reshape(1, 3, 256, 256)
file_list = os.listdir(dir)
list = []
for filename in file_list:
    data = {}
    dirt = dir + filename
    # 加载图片
    im = caffe.io.load_image(dirt)*255

    # 用上面的transformer.preprocess来处理刚刚加载图�?
    net.blobs['data'].data[...] = transformer.preprocess('data', im)*0.0078125

    #输出每层网络的name和shape
    # for layer_name, blob in net.blobs.iteritems():
    #     print layer_name + '\t' + str(blob.data.shape)

    # 网络开始向前传播啦
    output = net.forward()

    # 找出最大的那个概率
    output_prob = output['prob'][0]
    print '预测的类别是:', output_prob.argmax()

    # 找出最可能的前俩名的类别和概率
    top_inds = output_prob.argsort()[::-1][:3]
    id = [top_inds[0],top_inds[1],top_inds[2]]
    data["image_id"] = filename
    data["label_id"] = id
    list.append(data)
    print "ben zhang tu pian :" ,filename
    print "预测最可能的前两名的编�? ",top_inds
    print "对应类别的概率是: ", output_prob[top_inds[0]], output_prob[top_inds[1]]
store(list)