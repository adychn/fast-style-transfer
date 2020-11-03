from __future__ import print_function
import functools
import pdb, time
import tensorflow as tf
import tensorflow.compat.v1 as v1
import numpy as np
import os
# the following imports are from our own python file
import transform
import vgg
from utils import get_img

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
DEVICES = 'CUDA_VISIBLE_DEVICES'

# np arr, np arr
def optimize(content_targets, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1000,
             batch_size=4, save_path='checkpoints/fast_style_transfer.ckpt', slow=False,
             learning_rate=1e-3, debug=False):
    if slow:
        batch_size = 1

    # content_target is a list of files, 4-D size, so this is about the batch size here.
    # If using only one content image, then mod here is 0.
    mod = len(content_targets) % batch_size
    if mod > 0:
        print("Train set has been trimmed slightly...")
        content_targets = content_targets[:-mod]

    # training image get to be 256 x 256 because of get_img resize,
    # it then get into tensorflow graph from Adam optimizer feed_dict.
    batch_shape = (batch_size, 256, 256, 3)
    style_shape = (1,) + style_target.shape # add 1 in the front for batch size, 4-D.
    print(f"batch_shape of the content image is: {batch_shape}")
    print(f"style_shape of the style image is: {style_shape}")

    ### Graph Construction ###
    # vgg won't be trained, because in vgg.py the weights are loaded through that matlab file.
    # computed vgg style features in gram matrices
    # tf.device('/cpu:0')
    config = v1.ConfigProto()
    config.gpu_options.allow_growth = True

    style_features = {}
    with tf.Graph().as_default(), v1.Session(config=config) as sess:
        style_image = v1.placeholder(tf.float32, shape=style_shape, name='style_image') # 4-D placeholder for feed_dict
        vgg_style_net = vgg.net(vgg_path, vgg.preprocess(style_image)) # extract feature volume
        np_style_target = np.array([style_target]) # a 3-D numpy array for feed_dict's input

        for layer in STYLE_LAYERS:
            # vgg_style_net[layer] is a tf.Tensor returned by tf.nn.relu,
            # eval at that layer, by running forward to that vgg layer or entire network.
            features = vgg_style_net[layer].eval(feed_dict={style_image:np_style_target}) # extract a fVol value
            features = np.reshape(features, (-1, features.shape[3])) # (N*H*W, C)
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    # computed vgg content feature map and both losses
    with tf.Graph().as_default(), v1.Session(config=config) as sess:
        X_content = v1.placeholder(tf.float32, shape=batch_shape, name="X_content") # 4-D
        vgg_content_net = vgg.net(vgg_path, vgg.preprocess(X_content)) # run ground truth image through the pre-trained model

        # noisy prediction image runs through feed forward conv net, then
        # run through vgg to extract feature volume predicitons
        if slow:
            preds = tf.Variable(
                tf.random.normal(X_content.get_shape()) * 0.256
            )
            preds_pre = preds
        else:
            preds = transform.net(X_content/255.0) # run through the style feed forward network. why need to normalize pixel to 0-1?
        net = vgg.net(vgg_path, vgg.preprocess(preds)) # run generated image through the pre-trained model

        # _tensor_size is a reduce function only count from [1:],
        # so it doesn't have batch_size information.
        content_size = _tensor_size(vgg_content_net[CONTENT_LAYER]) * batch_size
        vgg_content_net_size = _tensor_size(vgg_content_net[CONTENT_LAYER])
        vgg_transform_content_net_size = _tensor_size(net[CONTENT_LAYER])
        # print(f"vgg_content_net_size is {vgg_content_net_size}")
        # print(vgg_content_net[CONTENT_LAYER])
        # print(f"vgg_transform_content_net_size is {vgg_transform_content_net_size}")
        # print(net[CONTENT_LAYER])
        assert vgg_content_net_size == vgg_transform_content_net_size

        # define loss functions
        # content loss
        content_l2_loss = 2 * tf.nn.l2_loss(net[CONTENT_LAYER] - vgg_content_net[CONTENT_LAYER])
        content_loss = content_weight * (content_l2_loss / content_size)

        # style loss
        style_l2_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            N, H, W, C = map(lambda i : i, layer.get_shape())
            feats = tf.reshape(layer, (N, H*W, C))        # N, HW, C
            feats_T = tf.transpose(feats, perm=[0, 2, 1]) # N, C, HW
            pred_gram = tf.matmul(feats_T, feats) / (H * W * C)
            true_gram = style_features[style_layer] # numpy array

            style_l2_loss = 2 * tf.nn.l2_loss(pred_gram - true_gram)
            style_l2_losses.append(style_l2_loss / true_gram.size)
        style_loss = style_weight * functools.reduce(tf.add, style_l2_losses) / batch_size

        # total variation denoising regularization loss
        # test if not needed in NN conv case and mirror padding
        # tv_y_size = _tensor_size(preds[:,1:,:,:])
        # tv_x_size = _tensor_size(preds[:,:,1:,:])
        # # N, H, W, C
        # y_tv = 2 * tf.nn.l2_loss(preds[:, 1:, :, :] - preds[:, :batch_shape[1]-1, :, :]) # H, down - up
        # x_tv = 2 * tf.nn.l2_loss(preds[:, :, 1:, :] - preds[:, :, :batch_shape[2]-1, :]) # W, right - left
        # tv_loss = tv_weight * (x_tv/tv_x_size + y_tv/tv_y_size) / batch_size

        # total loss
        # total_loss = content_loss + style_loss + tv_loss
        total_loss = content_loss + style_loss

        # train the feed forward net, and save weights to a checkpoint.
        import random
        uid = random.randint(1, 100)
        print("This random UID is: %s" % uid)

        optimizer = v1.train.AdamOptimizer(learning_rate).minimize(total_loss)
        sess.run(v1.global_variables_initializer())
        for epoch in range(epochs): # epoch loop
            iterations = 0
            num_examples = len(content_targets) # COCO train2014 ~20000 images
            while iterations * batch_size < num_examples: # batch loop
                # start training a batch
                start_time = time.time()

                X_batch = np.zeros(batch_shape, dtype=np.float32)
                start = iterations * batch_size
                end = iterations * batch_size + batch_size
                for i, img_p in enumerate(content_targets[start:end]): # img_p is a coco images
                   X_batch[i] = get_img(img_p, (256,256,3)).astype(np.float32) # resize to 256 x 256

                optimizer.run(feed_dict={X_content:X_batch})

                end_time = time.time()
                # end training a batch

                # update training information
                iterations += 1
                is_print_iter = int(iterations) % print_iterations == 0
                is_last_train = epoch == epochs - 1 and iterations * batch_size >= num_examples

                if slow:
                    is_print_iter = epoch % print_iterations == 0
                if debug:
                    print("UID: %s, batch training time: %s" % (uid, end_time - start_time))
                # monitor the training losses
                if is_print_iter or is_last_train:
                    _style_loss, _content_loss, _total_loss, _preds = \
                        sess.run([style_loss, content_loss, total_loss, preds],
                                  feed_dict={X_content:X_batch})
                    losses = (_style_loss, _content_loss, _total_loss)
                    generated_image = _preds

                    if slow:
                       generated_image = vgg.unprocess(generated_image)
                    else:
                       res = v1.train.Saver().save(sess, save_path)
                    print("yield")
                    yield(generated_image, losses, iterations, epoch)

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d for d in tensor.get_shape()[1:]), 1)
