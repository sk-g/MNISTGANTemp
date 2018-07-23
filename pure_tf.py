import tensorflow as tf
import numpy as np
import helper
import os
from glob import glob
from matplotlib import pyplot
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# get data
data_dir = './data'
helper.download_extract('mnist', data_dir)
helper.download_extract('celeba', data_dir)

show_n_images = 25

mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'mnist/*.jpg'))[:show_n_images], 28, 28, 'L')
pyplot.imshow(helper.images_square_grid(mnist_images, 'L'), cmap='gray')

show_n_images = 25

mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'img_align_celeba/*.jpg'))[:show_n_images], 28, 28, 'RGB')
pyplot.imshow(helper.images_square_grid(mnist_images, 'RGB'))


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

import problem_unittests as tests

def model_inputs(image_width, image_height, image_channels, z_dim):
    """
    Create the model inputs
    :param image_width: The input image width
    :param image_height: The input image height
    :param image_channels: The number of image channels
    :param z_dim: The dimension of Z
    :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
    """
    image_placeholder = tf.placeholder(dtype=tf.float32,shape = [None,image_width, image_height, image_channels])
    z_placeholder = tf.placeholder(dtype = tf.float32,shape=[None,z_dim])
    lr_placeholder = tf.placeholder(dtype = tf.float32)
    return image_placeholder, z_placeholder, lr_placeholder
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def discriminator(images, reuse=False):
    """
    Create the discriminator network
    :param images: Tensor of input image(s)
    :param reuse: Boolean if the weights should be reused
    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """
    alpha = 0.1
    keep_prob = 0.9
    # TODO: Implement Function    
    with tf.device("/device:GPU:0") and tf.variable_scope('discriminator', reuse=reuse):
        #with tf.variable_scope("discriminator") as scope:
#         if reuse == True:
#             tf.variable_scope("discriminator").reuse_variables()
        x1 = tf.layers.conv2d(images, 
                              filters = 128,
                              kernel_size = 5, 
                              strides=2, 
                              padding='same', 
                              activation=tf.nn.relu)
#         x1 = tf.maximum(alpha * x1, x1) # leaky relu
        
        x2 = tf.layers.conv2d(x1, 
                              filters = 128,
                              kernel_size = 5,
                              strides=2,
                              padding='same',
                              activation=tf.nn.relu)
        x2 = tf.layers.max_pooling2d(x2,pool_size = 2, strides = 1)
        
        x2 = tf.layers.batch_normalization(x2, training=True)
        
        x2 = tf.maximum(alpha * x2, x2)
        
        x2 = tf.nn.dropout(x2, keep_prob=keep_prob)
        
        x3 = tf.layers.conv2d(x2, 256, 5, strides=2, padding='same', activation=tf.nn.relu)
        
        #x3 = tf.layers.batch_normalization(x3, training=True)
        
        #x3 = tf.maximum(alpha * x3, x3)
        
        #x3 = tf.nn.dropout(x3, keep_prob=keep_prob)
        
        flat = tf.reshape(x3, (-1,  4 * 4 * 256))
        
        logits = tf.layers.dense(flat, 1)
        
        out = tf.sigmoid(logits)
        
    return out, logits

def generator(z, out_channel_dim, is_train=True):
    """
    Create the generator network
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """
    
    alpha = 0.1
    keep_prob = 0.9
    
    with tf.device("/device:GPU:0") and tf.variable_scope('generator', reuse= not is_train):
        nch = 512
        g1 = tf.layers.dense(z,28*28*256,activation = tf.nn.relu)
        g1 = tf.reshape(g1, (-1,28,28,256))
        g1 = tf.layers.batch_normalization(g1, training=is_train)
        g1 = tf.layers.conv2d(g1,filters = nch//4,kernel_size = (3,3),activation = tf.nn.relu,padding = 'same')
        g1 = tf.nn.dropout(g1, keep_prob = keep_prob)
        g1 = tf.layers.conv2d(g1,filters = nch//8,kernel_size = (1,1),activation = tf.nn.relu,padding = 'same')
        g1 = tf.nn.dropout(g1, keep_prob = keep_prob)
        g1 = tf.layers.conv2d(g1,filters = 32,kernel_size = (5,5),activation = tf.nn.relu,padding = 'same')
        g1 = tf.nn.dropout(g1, keep_prob = keep_prob)        
        g1 = tf.layers.conv2d(g1,filters = 1,kernel_size = (3,3),activation = tf.nn.relu,padding = 'same')
        g1 = tf.nn.dropout(g1, keep_prob = keep_prob)
        #g1 = tf.nn.max_pool(g1, padding = 'SAME')
        logits = tf.layers.conv2d_transpose(g1, out_channel_dim, 3, 
                                            strides=1, 
                                            padding='same', 
                                            activation = tf.nn.relu)
        
        out = tf.tanh(logits)
    return out

def model_loss(input_real, input_z, out_channel_dim):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """

    
    smooth = 0.1
    g = generator(input_z, out_channel_dim)
    d_op_real, d_logits_real = discriminator(input_real,reuse = False)
    d_op_fake, d_logits_fake = discriminator(g, reuse = True)
    
    
    disc_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits = d_logits_real,
            labels = tf.ones_like(d_op_real) * (1 - smooth)
        )
    )
    
    
    disc_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits = d_logits_fake,
            labels = tf.zeros_like(d_op_fake) # zero if real
        )
    )
    
    
    gen_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits = d_logits_fake,
            labels = tf.ones_like(d_op_fake) # one if fake
        )
    )
    
    disc_loss = disc_loss_real + disc_loss_fake
    
    return disc_loss, gen_loss

def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    # TODO: Implement Function
    
    g_vars = tf.trainable_variables(scope = 'generator')
    d_vars = tf.trainable_variables(scope = 'discriminator')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        gen_train_opt = tf.train.AdamOptimizer(learning_rate = learning_rate, 
                                               beta1=beta1).minimize(g_loss,
                                                                     var_list = g_vars)
        disc_train_opt = tf.train.AdamOptimizer(learning_rate = learning_rate,
                                                beta1 = beta1).minimize(d_loss,
                                                                      var_list = d_vars)

        return disc_train_opt, gen_train_opt  
def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
    """
    model_inputs <-     return image_placeholder, z_placeholder, lr_placeholder
    model_loss <- d loss, g loss
    model_opt <- disc_train_opt, gen_train_opt  
    
    Train the GAN
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """

    _, image_width, image_height, image_channels = data_shape
    input_real, input_z, lr = model_inputs(image_width, image_height, image_channels, z_dim)
    disc_loss, gen_loss = model_loss(input_real, input_z, image_channels)
    disc_opt, gen_opt = model_opt(disc_loss, gen_loss, lr, beta1)

    
    saver = tf.train.Saver()
    losses = []
    steps = 0
    total_steps = epoch_count * batch_size    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            for batch_images in get_batches(batch_size):
                steps += 1
                batch_images *= 2
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                
                _ = sess.run(
                    disc_opt, 
                    feed_dict={
                        input_real: batch_images, input_z: batch_z, lr: learning_rate 
                    }
                )
                _ = sess.run(
                    gen_opt, 
                    feed_dict={
                        input_real: batch_images, 
                        input_z: batch_z, 
                        lr:learning_rate
                    }
                )
                if steps == 1:
                    print('initial output:')
                    show_generator_output(sess, 16, input_z, image_channels, data_image_mode)
                if steps % 100 == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = disc_loss.eval({input_z: batch_z, input_real: batch_images})
                    train_loss_g = gen_loss.eval({input_z: batch_z})

                    print("Epoch {}/{}...".format(epoch_i+1, epochs),
                            "Generator Loss: {:.4f}".format(train_loss_g),
                            "Discriminator Loss: {:.4f}...".format(train_loss_d))
                    # Save losses to view after training
                    losses.append((train_loss_d, train_loss_g))

                if steps % 100 == 0:
                    show_generator_output(sess, 16, input_z, image_channels, data_image_mode)
                    

        saver.save(sess, 'generator.ckpt')
                
    return losses                    
                    
                
                
def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
    """
    Show example output for the generator
    :param sess: TensorFlow session
    :param n_images: Number of Images to display
    :param input_z: Input Z Tensor
    :param out_channel_dim: The number of channels in the output image
    :param image_mode: The mode to use for images ("RGB" or "L")
    """
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})

    images_grid = helper.images_square_grid(samples, image_mode)
    pyplot.imshow(images_grid, cmap=cmap)
    pyplot.show()    
def main():
    batch_size = 128
    z_dim = 128
    learning_rate = 0.009
    beta1 = 0.99


    celeba_dataset = helper.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))
    with tf.Graph().as_default():
        train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
            celeba_dataset.shape, celeba_dataset.image_mode)
    batch_size = 512
    z_dim = 128
    learning_rate = 0.01
    beta1 = 0.99


    """
    DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
    """
    epochs = 24

    mnist_dataset = helper.Dataset('mnist', glob(os.path.join(data_dir, 'mnist/*.jpg')))
    with tf.Graph().as_default():
        train(epochs, batch_size, z_dim, learning_rate, beta1, mnist_dataset.get_batches,
            mnist_dataset.shape, mnist_dataset.image_mode)
if __name__ == '__main__':
    # testing functions
    tests.test_model_inputs(model_inputs)
    tests.test_discriminator(discriminator, tf)
    tests.test_generator(generator, tf)
    tests.test_model_loss(model_loss)
    tests.test_model_opt(model_opt, tf)
