
#############模型搭建#############
train_file_path = './random_train.txt'
data_root_path='./cat_12_train'
test_file_path = './test.txt'
#训练集数据处理
def train_mapper(sample):
    img, label = sample

    # 读取图像数据
    img = paddle.dataset.image.load_image(img)
    # 将图像尺寸缩放到统一大小
    img = paddle.dataset.image.simple_transform(im=img,
                                                resize_size=224,
                                                crop_size=224,
                                                is_train=True,
                                                is_color=True)
    # 归一化
    img = img.astype('float32') / 255.0

    return img, label
def train_r(train_list):
    def reader():
        with open(train_list, 'r') as f:
            for line in f:
                img_path, lab = line.strip().split('\t')
                yield img_path, int(lab)

    return paddle.reader.xmap_readers(train_mapper,
                                      reader,
                                      cpu_count(),
                                      1024)

# 测试集的读取器
def test_mapper(sample):
    img, label = sample

    # 读取图像数据
    img = paddle.dataset.image.load_image(img)
    # 将图像尺寸缩放到统一大小
    img = paddle.dataset.image.simple_transform(im=img,
                                                resize_size=224,
                                                crop_size=224,
                                                is_train=False,
                                                is_color=True)
    # 归一化
    img = img.astype('float32') / 255.0

    return img, label


# 数据准备(数据读取器)
def test_r(test_list):
    def reader():
        with open(test_list, 'r') as f:
            for line in f:
                img_path, lab = line.strip().split('\t')
                yield img_path, int(lab)

    return paddle.reader.xmap_readers(test_mapper,
                                      reader,
                                      cpu_count(),
                                      1024)


BATCH_SIZE = 32
#训练集原始数据读取
train_reader = train_r(train_file_path)
random_train_reader = paddle.reader.shuffle(train_reader, 1024)
batch_train_reader = paddle.batch(random_train_reader,
                                  BATCH_SIZE)

#测试集数据读取
test_reader = test_r(test_file_path)
batch_test_reader = paddle.batch(test_reader,
                                 BATCH_SIZE)
# 搭建模型
def vgg(image):
    def conv_block(ipt, num_filter, group):
        return fluid.nets.img_conv_group(input=ipt,  # 输入数据
                                         conv_num_filter=[num_filter] * group,  # 卷积核数量
                                         pool_size=2,  # 池化区域
                                         conv_padding=1,  # 卷积填充
                                         conv_filter_size=3,  # 卷积核尺寸
                                         conv_act='relu',  # 卷基层的激活函数
                                         conv_with_batchnorm=True,  # 卷基层是否做批量归一化
                                         pool_stride=2,  # 池化步长
                                         pool_type='max')  # 池化类型

    conv1 = conv_block(image, 64, 2)
    conv2 = conv_block(conv1, 128, 2)
    conv3 = conv_block(conv2, 256, 3)
    conv4 = conv_block(conv3, 512, 3)
    conv5 = conv_block(conv4, 512, 3)

    #dropout
    drop=fluid.layers.dropout(x=conv5,dropout_prob=0.5,)
    # 全连接
    fc1 = fluid.layers.fc(drop, 512, act='relu')

    #BN(批量归一化）
    BN=fluid.layers.batch_norm(fc1,act='relu')
    #dropout
    drop=fluid.layers.dropout(x=BN,dropout_prob=0.5)

    fc2 = fluid.layers.fc(drop, 512, act='relu')
    pred_y = fluid.layers.fc(fc2, 5, act='softmax')

    return pred_y


# 占位符
image = fluid.layers.data(name='image',
                          shape=[3, 224, 224],
                          dtype='float32')
label = fluid.layers.data(name='label',
                          shape=[1],
                          dtype='int64')

pred_y = vgg(image=image)

# 损失函数
cost = fluid.layers.cross_entropy(input=pred_y,  # 预测值
                                  label=label)  # 真实值
avg_cost = fluid.layers.mean(cost)

# 精度
accuracy = fluid.layers.accuracy(input=pred_y,
                                 label=label)

#克隆program进行测试
test_program=fluid.default_main_program().clone(for_test=True)

# 梯度下降优化器
fluid.optimizer.Adam(0.00001).minimize(avg_cost)

# 开始训练
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place=place)
exe.run(fluid.default_startup_program())  # 初始化

# 参数喂入器
feeder = fluid.DataFeeder(feed_list=[image, label],
                          place=place)

costs = []
accs = []
iters = []
for pass_id in range(50):
    train_costs = []
    train_accs = []
    for data in batch_train_reader():
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, accuracy])
        train_costs.append(train_cost[0])
        train_accs.append(train_acc[0])
    #求每一轮训练的平均的损失值和准确率
    train_avg_cost = sum(train_costs) / len(train_costs)
    train_avg_acc = sum(train_accs) / len(train_accs)
    print('轮数:{},cost:{},acc:{}'.format(pass_id,
                                        train_avg_cost,
                                        train_avg_acc))
    costs.append(train_avg_cost)
    accs.append(train_avg_acc)
    iters.append(pass_id)

    #测试（验证）
    test_accs=[]
    for data in batch_test_reader():
        test_acc=exe.run(program=test_program,feed=feeder.feed(data),fetch_list=[accuracy])
        test_accs.append(test_acc[0][0])
        test_avg_acc=sum(accs)/len(train_accs)
        print('测试集:{},acc:{}'.format(pass_id,test_avg_acc))


#训练过程可视化
plt.plot(iters,costs,color='orangered',label='cost')
plt.plot(iters,accs,color='dodgerblue',label='acc')

plt.savefig('train.png')

#模型的保存

model_save_path = './model/fruits/'

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

fluid.io.save_inference_model(model_save_path,
                              feeded_var_names=['image'],
                              target_vars=[pred_y],
                              executor=exe)
print('模型保存成功:',model_save_path)
