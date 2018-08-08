# AItest
togithub
‘’‘ python
def get_inference_results(val_file, net_ptt, caffe_model, data_dir=""):
    """Get results from the validation file
    Args:
        val_file: the validation file, a line is formated as: image_path label
        net_ptt: the prototxt for evaluation and deployment
        caffe_model: the trained network weights for evaluation
        data_dir: where the test images lie on, abs_image_path=$data_dir/image_path
    Return:
        The lists contains the image paths, ground truth and predictions
    """
    caffe.set_mode_gpu()
    caffe.set_device(0)
    #net = caffe.Net(net_ptt, caffe_model, caffe.TEST)
    net = caffe.Net(net_ptt, caffe_model, caffe.TEST)
    print("The input data shape: " + str(net.blobs['data'].data.shape))

    # Set the transformer for preprocessing data
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255)
    transformer.set_mean('data', np.array([104, 117, 123]))
    transformer.set_channel_swap('data', (2,1,0))

    img_paths, y_test, predictions = [], [], []

    # Generate the evaluation results from the given file
    for line in open(val_file, 'r'):
        line = line.strip().split(' ')
        img, label = line[0], line[1]

        img_path = os.path.join(data_dir, img)
        img_paths.append(img_path)
        if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
            print('Lost image: %s' % img_path)
            continue
        try:
            im = caffe.io.load_image(img_path)
            net.blobs['data'].data[...] = transformer.preprocess('data', im)
            out = net.forward()
        except Exception as e:
            print(e)
            continue
        y_test.append(int(label))
        predictions.append(np.argsort(out['loss'])[0])
    return img_paths, y_test, predictions
    ’‘’
