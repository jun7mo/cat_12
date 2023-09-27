import os
import paddle
from multiprocessing import cpu_count
import paddle.fluid as fluid
import matplotlib.pyplot as plt
import random
import cv2
###########数据预处理################
train_file_path = './train_list.txt'
data_root_path='./cat_12_train'
test_file_path = './test.txt'

cv2.destroyAllWindows()
# 读取文件中的内容，并写入列表FileNameList
def ReadFileDatas(original_filename):
      FileNameList = []
      file = open(original_filename, 'r+', encoding='utf-8')
      for line in file:
            FileNameList.append(line)  # 写入文件内容到列表中去
      print('数据集总量：', len(FileNameList))
      file.close()
      return FileNameList

# listInfo为 ReadFileDatas 的列表
def WriteDatasToFile(listInfo, new_filename):
      f = open(new_filename, mode='w', encoding='utf-8')
      for idx in range(len(listInfo)):
            str = listInfo[idx]  # 列表指针
            f.write(str)
      f.close()
      print('写入 %s 文件成功.' % new_filename)

listFileInfo = ReadFileDatas('./train_list.txt')            # 读取文件
random.shuffle(listFileInfo)                       # 打乱顺序
WriteDatasToFile(listFileInfo,'./random_train.txt')       # 保存新的文件
random_train='./random_train.txt'
with open(test_file_path, 'w') as f:
    pass
with open(random_train, 'r') as f:
    i = 0
    for line in f:
        if i % 10 == 0:
            with open(test_file_path,'a') as f:
                f.write(line)
        i += 1

print('训练集测试集划分完成!!!')


import os
import paddle
from multiprocessing import cpu_count
import paddle.fluid as fluid
import matplotlib.pyplot as plt
import cv2
#############模型搭建#############
train_file_path = './train_list.txt'
data_root_path = './cat_12_train'
test_file_path = './test.txt'


def train_mapper(sample):  # sample为元组
    img_path, label = sample
    # 读取图像数据
    # try:
    img = paddle.dataset.image.load_image(img_path)
    # 对图片进行预处理，进行缩放，裁剪成相同大小
    # im:图像路径     resize_size:缩放大小  is_color: 是否为彩色  crop_size: 裁剪大小  is_train:训练模式（随机裁剪）   #VGG的resize_size和crop_size为224
    img = paddle.dataset.image.simple_transform(im=img, resize_size=224, crop_size=224, is_color=True, is_train=True)
    # 归一化处理（把每个像素值缩放到0~1之间）模型可能不会收敛
    img = img.astype('float32') / 255.0
    # 返回图像和标签
    return img, label
    # except AttributeError:
    #     print("图像读取失败，请检查图像路径或名称")

def train_r(train_list, buffer_size=1024):
    def reader():
        with open(train_list, 'r') as f:
            for line in f.readlines():  # 遍历每一行数据   strip()去掉两侧空白符
                img_path, img_label = line.strip().replace('\n', '').split('\t')  # 去掉换行符并把图像路径和标签拆分成列表
                # print('划分成功,路径为{},类别为{}'.format(img_path,img_label))
                yield img_path, int(img_label)  # yield特殊的迭代器，节省内存
        # mapper:处理函数
        # reader: 原始读取函数，将数据传递给train_mapper
        # process_num: 线程数量    cpu_count() 计算cpu数量
        # buffer_size: 读取缓冲区大小

    return paddle.reader.xmap_readers(
        mapper=train_mapper, reader=reader,
        buffer_size=buffer_size, process_num=cpu_count())



def test_mapper(sample):  # sample为元组
    img_path, label = sample
    img = paddle.dataset.image.load_image(img_path)
    img = paddle.dataset.image.simple_transform(im=img, resize_size=224, crop_size=224, is_color=True, is_train=False)
    img = img.astype('float32') / 255.0
    return img, label


def test_r(test_list, buffer_size=1024):
    def reader():
        with open(test_list, 'r') as f:
            for line in f.readlines():  # 遍历每一行数据   strip()去掉两侧空白符
                img_path, img_label = line.strip().replace('\n', '').split('\t')  # 去掉换行符并把图像路径和标签拆分成列表
                yield img_path, int(img_label)  # yield特殊的迭代器，节省内存

    return paddle.reader.xmap_readers(
        mapper=test_mapper, reader=reader,
        buffer_size=buffer_size, process_num=cpu_count())


# 定义数据读取器
BATCH_SIZE = 12
# 训练集
train_reader = train_r(train_file_path)  # 原始数据

random_train_reader = paddle.reader.shuffle(train_reader, buf_size=500)  # 随机数据读取器
batch_train_reader = paddle.batch(random_train_reader, batch_size=BATCH_SIZE)  # 批量数据读取器
# 测试集
test_reader = train_r(test_file_path)  # 原始数据
batch_test_reader = paddle.batch(test_reader, batch_size=BATCH_SIZE)  # 批量数据读取器


# 搭建VGG模型

def VGG_fuc(image):
    def conv_block(input, num_filters, group):
        return fluid.nets.img_conv_group(input=input,  # 输入数据
                                         conv_num_filter=[num_filters] * group,  # 卷积核数量
                                         pool_size=2,  # 池化区域
                                         conv_padding=1,  # 卷积填充
                                         conv_filter_size=3,  # 卷积核尺寸
                                         conv_act='relu',  # 激活函数
                                         conv_with_batchnorm=True,  # 卷积层是否做批量归一化
                                         pool_stride=2,  # 池化步长
                                         pool_type='max'  # 最大池化
                                         )

    conv1 = conv_block(input=image, num_filters=64, group=2)
    conv2 = conv_block(input=conv1, num_filters=128, group=2)
    conv3 = conv_block(input=conv2, num_filters=256, group=3)
    conv4 = conv_block(input=conv3, num_filters=512, group=3)
    conv5 = conv_block(input=conv4, num_filters=512, group=3)

    # 加上两个dropout并减少神经元数量防止过拟合
    # dropout
    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)

    # 全连接层fc
    fc1 = fluid.layers.fc(input=drop,
                          size=512,  # 神经元数量
                          act='relu')
    # BN批量归一化
    bn = fluid.layers.batch_norm(fc1, act='relu')
    # dropout
    drop = fluid.layers.dropout(x=bn, dropout_prob=0.5)

    fc2 = fluid.layers.fc(input=drop, size=512, act='relu')
    pred_y = fluid.layers.fc(input=fc2, size=5, act='softmax')

    return pred_y


# 占位符
image = fluid.layers.data(name='image', shape=[3, 224, 224], dtype='float32')  # VGG尺寸224*224*3
label = fluid.layers.data(name='label', shape=[1], dtype='int64')  # 特征只有1个维度

# 模型预测值
predict_y = VGG_fuc(image=image)
# 损失函数 交叉熵（此处为分类较适合）、均方误差
# input：预测值   lable：标签值/真实值
cost = fluid.layers.cross_entropy(input=predict_y, label=label)
avg_cost = fluid.layers.mean(cost)  # 平均值
# 正确率/精度
accuracy = fluid.layers.accuracy(input=predict_y, label=label)

# 预先克隆一个program，执行测试，便于测试时不会执行梯下降
test_program = fluid.default_main_program().clone(for_test=True)

# 梯度下降优化器
# Adam()自适应梯度下降优化器
optimizer = fluid.optimizer.Adam(learning_rate=0.0001)
optimizer.minimize(avg_cost)  # 指定优化的目标函数，求得损失函数的最小值

# 开始训练
# 定义一个使用GPU的执行器
place = fluid.CPUPlace()
# place = fluid.CUDAPlace(0)  # 0设备编号
exe = fluid.Executor(place=place)
exe.run(fluid.default_startup_program())  # 参数初始化
# 参数喂数器
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
costs = []  # 为可视化做准备
accs = []
iters = []
for pass_id in range(50):  # 外层控制轮次
    train_costs = []
    train_accs = []
    for data in batch_train_reader():
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, accuracy])
        train_costs.append(train_cost[0])
        train_accs.append(train_acc[0])
    # 求每一轮训练的平均的损失值和准确率
    train_avg_cost = sum(train_costs) / len(train_costs)
    train_avg_acc = sum(train_accs) / len(train_accs)
    print('轮数：{}    平均损失值：{:.3f}    平均准确率：{:.3f}'.format(pass_id + 1, train_avg_cost, train_avg_acc))
    costs.append(train_avg_cost)
    accs.append(train_avg_acc)
    iters.append(pass_id)

    # 测试（验证）
    test_accs = []
    for data in batch_test_reader():
        test_acc = exe.run(program=test_program,
                           feed=feeder.feed(data),
                           fetch_list=[accuracy])
        test_accs.append(test_acc[0][0])

    test_avg_acc = sum(test_accs) / len(test_accs)
    print('测试集轮数:{}    测试精度:{:.3f}'.format(pass_id + 1, test_avg_acc))

# 训练过程可视化
plt.plot(iters, costs, color='r', label='cost')
plt.plot(iters, accs, color='g', label='acc')

plt.savefig('train.png')  # 保存图片
plt.show()

# 模型保存
model_save_path = './model/fruits/'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
fluid.io.save_inference_model(dirname=model_save_path,
                              feeded_var_names=['image'],
                              target_vars=[predict_y],
                              executor=exe)
print('模型保存成功······')
