"""
Code idea from https://github.com/shekkizh/FCN.tensorflow
"""

import tensorflow as tf
import numpy as np
import fcn_model as fcn
import BatchDatasetReader as dataset
import ReadData as read
import scipy.misc as misc
import os, sys

mode = "train"
data_dir = "/data"
#data_dir = "zzzkkkyyy/datasets/data/1"
saver_dir = "/output/saver"
check_dir_read = "/data2"
check_dir = "/output/"
logs_dir = "/output/"

adam_beta = 0.9
adam_init_lr = 1e-2

input_width = 128
input_height = 192
input_channels = 4
batch_size = 16

def save_image(image, save_dir, name):
    misc.imsave(os.path.join(save_dir, name + ".png"), image)

def main(argv = None):
    x = tf.placeholder(tf.float32, [None, input_width, input_height, input_channels])
    y = tf.placeholder(tf.float32, [None, input_width, input_height, 1])
    is_training = tf.placeholder(tf.bool)
    global_step = tf.Variable(0, trainable = False)
    
    y_out = fcn.construct_layer(x, is_training)
    #tf.summary.image("input_image", x, max_outputs = 2)
    #tf.summary.image("ground_truth", tf.cast(y, tf.uint8), max_outputs = 2)
    #tf.summary.image("pred_annotation", tf.cast(y_out, tf.uint8), max_outputs = 2)
    #total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = tf.squeeze(y, squeeze_dims = [3]), name = "entropy")
    
    trainable_var = tf.trainable_variables()
    #mean_loss = -tf.reduce_mean(y * tf.log(tf.clip_by_value(y_out, 1e-10, 1.0)))
    #mean_loss = tf.reduce_mean(tf.square(y_out - y))
    mask_loss = tf.clip_by_value(y, 0, 1)
    
    mean_loss = tf.reduce_mean(tf.square(y_out - y) * mask_loss)
    adam_lr = tf.train.exponential_decay(adam_init_lr, global_step = global_step, decay_steps = 100, decay_rate = 0.9)
    #mean_loss = tf.reduce_mean(tf.square(y_out - y))
    #tf.summary.scalar("entropy", mean_loss)
    optimizer = tf.train.AdamOptimizer(adam_lr, adam_beta)
    grads = optimizer.compute_gradients(mean_loss, var_list = trainable_var)
    train_op = optimizer.apply_gradients(grads)
    
    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()
    print("Setting up image reader...")
    train_records, valid_records = read.read_dataset(data_dir)

    sess = tf.Session()
    
    print("Setting up Saver...")
    saver = tf.train.Saver()
    #summary_writer = tf.summary.FileWriter(saver_dir, sess.graph)
    
    print("Setting up dataset reader...")
    image_options = {"resize": True, "resize_width": input_width, "resize_height": input_height}
    train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)
    
    add_global = global_step.assign_add(1)
    print("Global variables initializing...")
    sess.run(tf.global_variables_initializer())
    """
    ckpt = tf.train.get_checkpoint_state(check_dir_read)
    if ckpt and ckpt.model_checkpoint_path:
        saver = tf.train.import_meta_graph('model.ckpt-100.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    """
    print("Begin running...")
    valid_images, valid_annotations = validation_dataset_reader.next_batch(batch_size)
    max_range = 1001
    valid_anno = np.squeeze(valid_annotations, axis = 3)
    for itr in range(max_range):
        train_images, train_annotations = train_dataset_reader.next_batch(batch_size)
        feed_dict = {x: train_images, y: train_annotations, is_training: True}
        sess.run(train_op, feed_dict = feed_dict)
        if itr % 5 == 0:
            train_loss = sess.run(mean_loss, feed_dict = feed_dict)
            print("Step: {} ---> Train_loss: {}".format(itr, train_loss ** 0.5))
            #summary_writer.add_summary(summary_str, itr)
            #saver_path = saver.save(sess, check_dir + "model.ckpt", itr)
        if itr % 20 == 0:
            valid_loss, pred = sess.run([mean_loss, y_out], feed_dict = {x: valid_images, y: valid_annotations, is_training: False})
            print("Step: {} ---> Validation_loss: {}".format(itr, valid_loss ** 0.5))
            pred = np.squeeze(pred, axis = 3)
            save_image(pred[0], logs_dir, name = "pred_" + str(itr))
            if itr % 100 == 0:
                for it in range(batch_size):
                    #save_image(valid_images[itr][:, :, 1:-1].astype(np.uint8), logs_dir, name = "input_" + str(itr))
                    save_image(valid_anno[it].astype(np.uint8), logs_dir, name = "truth_" + str(it) + "-" + str(itr))
                    save_image(pred[it], logs_dir, name = "prediction_" + str(it) + "-" + str(itr))
                print("Saved %d images." % batch_size)
        if itr != 0 and itr % 1000 == 0:
            saver_path = saver.save(sess, check_dir + "model.ckpt", itr)

    #pred = sess.run(y_out, feed_dict = {x: valid_images, y: valid_annotations, is_training: False})
    #pred = np.squeeze(pred, axis = 3)
    

    
if __name__ == "__main__":
    tf.app.run()    
    
    
