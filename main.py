import numpy as np
import matplotlib.pyplot as plt
import os
import time
import scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse
import cv2
import utls
import datetime
import math

# 输入参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--is_training", default=1, help="training or testing")
parser.add_argument("--data_dir", default="", help="path to dataset")
parser.add_argument("--task", default="ckpt", help="path to the folder containing the model")   # checkpoint文件夹
parser.add_argument("--continue_training", action="store_true", help="continue training or not")
parser.add_argument("--save_model_freq", default=1, type=int, help="frequency to save the model")
ARGS = parser.parse_args()

task = ARGS.task
is_training = (ARGS.is_training == 1)
continue_training = ARGS.continue_training
train_data_root = [ARGS.data_dir]
channel = 256 # number of feature channels to build the model

if not os.path.exists(task):
    os.makedirs(task)

# os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')  
if is_training:
    os.environ['CUDA_VISIBLE_DEVICES']=str(0)
else:
    os.environ['CUDA_VISIBLE_DEVICES']=str(0)

# build VGG19 to load pre-trained parameters
def build_net(ntype, nin, nwb = None, name = None):
    if ntype == 'conv':
        return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME', name=name) + nwb[1])
    elif ntype == 'pool':
        return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def get_weight_bias(vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    weights = tf.constant(weights)
    bias = vgg_layers[i][0][0][2][0][1]
    bias = tf.constant(np.reshape(bias, (bias.size)))
    return weights, bias

vgg_path = scipy.io.loadmat('./VGG_Model/imagenet-vgg-verydeep-19.mat')
print('[i] Loaded pre-trained vgg19 parameters')

def build_vgg19(input, reuse = False):
    with tf.variable_scope("vgg19"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        net = {}   # ~~dictionary
        vgg_layers = vgg_path['layers'][0]
        net['input'] = input - np.array([123.6800, 116.7790, 103.9390]).reshape((1, 1, 1, 3))
        net['conv1_1'] = build_net('conv', net['input'], get_weight_bias(vgg_layers, 0), name='vgg_conv1_1')
        net['conv1_2'] = build_net('conv', net['conv1_1'], get_weight_bias(vgg_layers, 2), name='vgg_conv1_2')
        net['pool1'] = build_net('pool', net['conv1_2'])
        net['conv2_1'] = build_net('conv', net['pool1'], get_weight_bias(vgg_layers, 5), name='vgg_conv2_1')
        net['conv2_2'] = build_net('conv', net['conv2_1'], get_weight_bias(vgg_layers, 7), name='vgg_conv2_2')
        net['pool2'] = build_net('pool', net['conv2_2'])
        net['conv3_1'] = build_net('conv', net['pool2'], get_weight_bias(vgg_layers, 10), name='vgg_conv3_1')
        net['conv3_2'] = build_net('conv', net['conv3_1'], get_weight_bias(vgg_layers, 12), name='vgg_conv3_2')
        net['conv3_3'] = build_net('conv', net['conv3_2'], get_weight_bias(vgg_layers, 14), name='vgg_conv3_3')
        net['conv3_4'] = build_net('conv', net['conv3_3'], get_weight_bias(vgg_layers, 16), name='vgg_conv3_4')
        net['pool3'] = build_net('pool', net['conv3_4'])
        net['conv4_1'] = build_net('conv', net['pool3'], get_weight_bias(vgg_layers, 19), name='vgg_conv4_1')
        net['conv4_2'] = build_net('conv', net['conv4_1'], get_weight_bias(vgg_layers, 21), name='vgg_conv4_2')
        net['conv4_3'] = build_net('conv', net['conv4_2'], get_weight_bias(vgg_layers, 23), name='vgg_conv4_3')
        net['conv4_4'] = build_net('conv', net['conv4_3'], get_weight_bias(vgg_layers, 25), name='vgg_conv4_4')
        net['pool4'] = build_net('pool', net['conv4_4'])
        net['conv5_1'] = build_net('conv', net['pool4'], get_weight_bias(vgg_layers, 28), name='vgg_conv5_1')
        net['conv5_2'] = build_net('conv', net['conv5_1'], get_weight_bias(vgg_layers, 30), name='vgg_conv5_2')
        return net

# build our highlight remove model
def lrelu(x):
    return tf.maximum(x*0.2, x)

def nm(x):
    w0 = tf.Variable(1.0, name='w0')
    w1 = tf.Variable(0.0, name='w1')
    return w0*x+w1*slim.batch_norm(x)

def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2], shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

def build(input, org):
    vgg19_features = build_vgg19(org[:, :, :, 0:3]*255.0)
    input = tf.concat([org, input], axis=3)
    for layer_id in range(1, 6):
        vgg19_f = vgg19_features['conv%d_2'%layer_id]
        input = tf.concat([tf.image.resize_bilinear(vgg19_f, (tf.shape(input)[1], tf.shape(input)[2]))/255.0, input], axis=3)  
    net = slim.conv2d(input, channel, [1, 1], padding='SAME', rate=1, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv0')
    net = slim.conv2d(net, channel, [3, 3], padding='SAME', rate=1, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv1')
    net = slim.conv2d(net, channel, [3, 3], padding='SAME', rate=2, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv2')
    net = slim.conv2d(net, channel, [3, 3], padding='SAME', rate=4, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv3')
    net = slim.conv2d(net, channel, [3, 3], padding='SAME', rate=8, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv4')
    net = slim.conv2d(net, channel, [3, 3], padding='SAME', rate=16, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv5')
    net = slim.conv2d(net, channel, [3, 3], padding='SAME', rate=32, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv6')
    net = slim.conv2d(net, channel, [3, 3], padding='SAME', rate=64, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv7')
    net = slim.conv2d(net, channel, [3, 3], padding='SAME', rate=128, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv8')  
    net = slim.conv2d(net, channel, [3, 3], padding='SAME', rate=256, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv9')
    net = slim.conv2d(net, 3, [1, 1], padding='SAME', rate=1, activation_fn=None, scope='g_conv_last')   # output 3 channels
    return net

def compute_l1_loss(input, output):
    return tf.reduce_mean(tf.abs(input-output))

def compute_loss(i_pred, i_true):
    MAE_loss = compute_l1_loss(i_true, i_pred)
    SSIM = utls.tf_ssim(tf.expand_dims(i_pred[:, :, :, 0], -1), tf.expand_dims(i_true[:, :, :, 0], -1)) + utls.tf_ssim(
        tf.expand_dims(i_pred[:, :, :, 1], -1), tf.expand_dims(i_true[:, :, :, 1], -1)) + utls.tf_ssim(
        tf.expand_dims(i_pred[:, :, :, 2], -1), tf.expand_dims(i_true[:, :, :, 2], -1))

    vgg_pred = build_vgg19(i_pred*255.0, reuse=True)
    vgg_real = build_vgg19(i_true * 255.0, reuse=False)
    p5 = compute_l1_loss(vgg_real['conv3_4'], vgg_pred['conv3_4'])
    VGG_loss = p5/255.0

    return MAE_loss, SSIM, VGG_loss

def compute_region_loss(i_pred, i_true, i_org):
    standard = 0.75  # 大于0.75的部分视为重点关注的高光部分
    gray1 = 0.39 * i_org[:, :, :, 0] + 0.5 * i_org[:, :, :, 1] + 0.11 * i_org[:, :, :, 2]  # 3维
    standard = tf.expand_dims(tf.expand_dims(standard, -1), -1)
    mask = tf.to_float(gray1 >= standard)
    mask1 = tf.expand_dims(mask, -1)
    mask = tf.concat([mask1, mask1, mask1], -1)

    high_fake_clean = tf.multiply(mask, i_pred[:, :, :, :])
    high_clean = tf.multiply(mask, i_true[:, :, :, :])
    Region_loss = compute_l1_loss(high_clean, high_fake_clean)

    return Region_loss

# Session #
sess = tf.Session()
input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
org = tf.placeholder(tf.float32, shape=[None, None, None, 3])
target = tf.placeholder(tf.float32, shape=[None, None, None, 3])
lr = tf.placeholder(tf.float32)
img_out = build(input, org)
MAE_loss, SSIM, VGG_loss = compute_loss(img_out, target)
SSIM_loss = (3-SSIM)/3
Region_loss = compute_region_loss(img_out, target, org)
loss = MAE_loss*5.0 + VGG_loss + SSIM_loss + Region_loss*40
train_vars = tf.trainable_variables()


train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss) # optimizer
saver = tf.train.Saver(max_to_keep=5)  # 保存最大的checkpoint文件数
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(task)
print("[i] contain checkpoint: ", ckpt)

if ckpt:
    saver_restore = tf.train.Saver([var for var in tf.trainable_variables()])
    print('loaded' +ckpt.model_checkpoint_path)
    saver_restore.restore(sess, ckpt.model_checkpoint_path)   

#输入数据准备
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_img_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def data_augmentation(traindata, traindata_label):   # 数据增强处理
    rotation_or_transp_rand = np.random.randint(0, 4)

    if rotation_or_transp_rand == 1:  
        traindata = cv2.flip(traindata, -1)
        traindata_label = cv2.flip(traindata_label, -1)
    elif rotation_or_transp_rand == 2: 
        traindata = cv2.flip(traindata, 1)
        traindata_label = cv2.flip(traindata_label, 1)
    elif rotation_or_transp_rand == 3:
        traindata = cv2.flip(traindata, 0)
        traindata_label = cv2.flip(traindata_label, 0)

    return traindata, traindata_label

def psf_generator(img):   # 得到Ipsf
    h, w = img.shape[0], img.shape[1]
    Imin = np.ones((h, w))
    Ipsf = np.ones((h, w, 3))

    for i in range(h):
        for j in range(w):
            Imin[i, j] = min(img[i, j, :])
    for i in range(3):
        Ipsf[:, :, i] = img[:, :, i] - Imin

    return Ipsf

maxepoch = 10000
if is_training:
    def prepare_data(train_path):
        input_names = []
        output_names = []
        for dirname in train_path:
            train_b = dirname + "blended/"   # 包含高光图像路径
            train_d = dirname + "diffuse/"   #　去高光图像位置
            for root, dirs, files in sorted(os.walk(train_b)):
                for file in files:
                    if is_img_file(file):
                        path_input = os.path.join(train_b, file)
                        input_names.append(path_input)
            for root, dirs, files in sorted(os.walk(train_d)):
                for file in files:
                    if is_img_file(file):
                        path_output = os.path.join(train_d, file)
                        output_names.append(path_output)
        return input_names, output_names

    def prepare_data_val(val_path):
        input_names = []
        truth_names = []
        for dirname in val_path:
            val_b = dirname + "blended/"   # 包含高光图像路径
            val_d = dirname + "diffuse/"   #　去高光图像位置
            for _, _, files in sorted(os.walk(val_b)):
                for file in files:
                    if is_img_file(file):
                        path_input = os.path.join(val_b, file)
                        input_names.append(path_input)
            for _, _, files in sorted(os.walk(val_d)):
                for file in files:
                    if is_img_file(file):
                        path_output = os.path.join(val_d, file)
                        truth_names.append(path_output)
        return input_names, truth_names

    input_blended_images, output_diffuse_images = prepare_data(train_data_root)
    print("[i] Total %d training images, first path of image is %s." % (len(output_diffuse_images), input_blended_images[0]))

    num_train = len(output_diffuse_images)

    txt_name = '%s/training_record.txt' % task
    f1 = open(txt_name, 'a')
    now = datetime.datetime.now()
    f1.write('\n' + str(now) + '\n\n')
    f1.close()

    last_epoch = 1 
    learning_rate = 1e-6 # 初始lr
    cnt = 0
    val_cnt = last_epoch
    loss0 = 10
    loss1 =10

    for epoch in range(last_epoch, maxepoch):
        all_l = np.zeros(num_train, dtype=float)
        mae_l = np.zeros(num_train, dtype=float)
        ssim_l = np.zeros(num_train, dtype=float)
        vgg_l = np.zeros(num_train, dtype=float)
        region_l = np.zeros(num_train, dtype=float)

        if os.path.isdir("%s/%04d"%(task, epoch)):
            continue
        count = 0
        for id in np.random.permutation(num_train):   # 对图像随机排序
            st = time.time()
            inputimg = cv2.imread(input_blended_images[id], -1)   # 参数-1：读取完整原图
            outputimg = cv2.imread(output_diffuse_images[id], -1)
            neww = np.random.randint(256, 300) 
            newh = round((neww / inputimg.shape[1]) * inputimg.shape[0]) 
            inputimg = cv2.resize(np.float32(inputimg), (neww, newh), cv2.INTER_CUBIC) / 255.0  
            outputimg = cv2.resize(np.float32(outputimg), (neww, newh), cv2.INTER_CUBIC) / 255.0
            inputimg, outputimg = data_augmentation(inputimg, outputimg)
            input_org = np.expand_dims(inputimg, axis=0)
            psfimage = psf_generator(inputimg)
            input_psf = np.expand_dims(psfimage, axis=0)  #扩充一个通道
            target_image = np.expand_dims(outputimg, axis=0)

            # alternate training
            fetch_list = [train_op, img_out, loss, MAE_loss, SSIM_loss, VGG_loss, Region_loss]   # 希望查看的变量
            _, pred_image, current, MAE, SSIM, VGG_L, Region_L = sess.run(fetch_list, feed_dict={input:input_psf, target:target_image, org:input_org, lr:learning_rate})  # 更新训练

            all_l[id] = current
            mae_l[id], ssim_l[id], vgg_l[id], region_l[id] = MAE, SSIM, VGG_L, Region_L
            print(
                "iter: %d %d || all_loss: %.5f || mae_loss: %.5f || ssim_loss: %.5f || vgg_loss: %.5f || region_loss: %.5f || time: %.2f" % (
                epoch, count, np.mean(all_l[np.where(all_l)]), np.mean(mae_l[np.where(mae_l)]),
                np.mean(ssim_l[np.where(ssim_l)]), np.mean(vgg_l[np.where(vgg_l)]),
                np.mean(region_l[np.where(region_l)]), time.time() - st))
            if math.isnan(current):
                print("loss is Nan, stop training!")
                break
            count += 1

        # change learning rate
        if np.mean(all_l[np.where(all_l)]) < loss0:
            loss0 = np.mean(all_l[np.where(all_l)])
            cnt = 0
        else:
            cnt += 1
        print("minloss")
        print(loss0)
        print(cnt)

        if cnt == 5:
            print("No improvement, stop training!")
            f1 = open(txt_name, 'a')
            f1.write("No improvement, stop training! min loss is %.5f" % loss0 + '\n')
            f1.close()
            break

        if cnt >= 2 and learning_rate > 1e-6:
            learning_rate = learning_rate*0.5
            print("learning rate change as: %.10f" % learning_rate)
            f1 = open(txt_name, 'a')
            f1.write("learning rate change as: %.10f" % learning_rate + '\n')
            f1.close()

        # save model and images every epoch
        train_sess = sess
        if epoch % ARGS.save_model_freq == 0:
            os.makedirs("%s/%04d" % (task, epoch))
            fileid = os.path.splitext(os.path.basename(input_blended_images[id]))[0] 
            if not os.path.isdir("%s/%04d/%s" % (task, epoch, fileid)):
                os.makedirs("%s/%04d/%s" % (task, epoch, fileid))
            pred_image = np.minimum(np.maximum(pred_image, 0.0), 1.0) * 255.0
            print("shape of outputs: ", pred_image.shape)
            f1 = open(txt_name, 'a')
            f1.write(
                '.' + 'epoch: %d || all_loss: %.5f || mae_loss: %.5f || ssim_loss: %.5f || vgg_loss: %.5f || region_loss: %.5f ' % (
                epoch, np.mean(all_l[np.where(all_l)]), np.mean(mae_l[np.where(mae_l)]),
                np.mean(ssim_l[np.where(ssim_l)]), np.mean(vgg_l[np.where(vgg_l)]),
                np.mean(region_l[np.where(region_l)])) + '\n')
            f1.close()
            cv2.imwrite("%s/%04d/%s/input.png" % (task, epoch, fileid), np.uint8(np.squeeze(inputimg * 255.0)))
            cv2.imwrite("%s/%04d/%s/target.png" % (task, epoch, fileid), np.uint8(np.squeeze(outputimg * 255.0)))
            cv2.imwrite("%s/%04d/%s/output.png" % (task, epoch, fileid), np.uint8(np.squeeze(pred_image)))

        #validation
        val_path = ["Dataset/validation/"]  # 验证集位置
        val_names, truth_names = prepare_data_val(val_path)
        num_val = len(val_names)
        val_losses = np.zeros(num_val, dtype=float)
        for i in range(num_val):
            if not os.path.isfile(val_names[i]):
                continue
            img = cv2.imread(val_names[i])
            ground_truth = cv2.imread(truth_names[i])
            input_org = np.expand_dims(np.float32(img/255.0), axis=0)
            Ipsf = psf_generator(img)
            input_psf = np.expand_dims(np.float32(Ipsf/255.0), axis=0)
            truth_image = np.expand_dims(np.float32(ground_truth/255.0), axis=0)
            val_loss = sess.run(loss, feed_dict={input:input_psf, target:truth_image, org:input_org})
            val_losses[i] = val_loss
        mean_val_loss = np.mean(val_losses)
        if mean_val_loss < loss1:
            saver.save(train_sess, "%s/model.ckpt" % task)
            saver.save(train_sess, "%s/%04d/model.ckpt" % (task, epoch))
            loss1 = mean_val_loss
            val_cnt = epoch
        print("val loss: %.5f" %mean_val_loss)
        f1 = open(txt_name, 'a')
        f1.write("val loss: %.5f" %mean_val_loss + '\n')
        f1.write("min val loss :epoch %d : %.5f" %(val_cnt, loss1) + '\n')
        f1.close()

# Test
else:
    def prepare_data_test(test_path):
        input_names = []
        for dirname in test_path:
            for _, _, fnames in sorted(os.walk(dirname)):  
                for fname in fnames:
                    if is_img_file(fname):
                        input_names.append(os.path.join(dirname, fname))
        return input_names

    test_path = ["Dataset/photo/"]  # Replace with your own test image path  # 要测试的文件夹
    subtask = "complete"  # if you want to save different tests separately, replace it with the name of the test set  # 输出图片位置
    val_names = prepare_data_test(test_path)

    for val_path in val_names:
        testid = os.path.splitext(os.path.basename(val_path))[0]
        if not os.path.isfile(val_path):
            continue
        img = cv2.imread(val_path)
        input_org = np.expand_dims(np.float32(img / 255.0), axis=0)
        Ipsf = psf_generator(img)
        input_psf = np.expand_dims(np.float32(Ipsf / 255.0), axis=0)
        st = time.time()
        output_image = sess.run(img_out, feed_dict={input:input_psf, org:input_org})
        print("Test time %.3f for image %s" % (time.time()-st, val_path))
        output_image = np.minimum(np.maximum(output_image, 0.0), 1.0)*255.0
        print("shape of outputs: ", output_image.shape)
        output_image = np.squeeze(output_image)
        if not os.path.isdir("test_results/%s" % (subtask)):
            os.makedirs("test_results/%s" % (subtask))
        #cv2.imwrite("./test_results/%s/%s/input.png" % (subtask, testid), img)
        cv2.imwrite("test_results/%s/%s.png" % (subtask, testid), output_image)

        
















