import tensorflow as tf
import numpy as np 
import math
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data

# load MNIST data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
mean_img = np.mean(mnist.train.images, axis=0)

# Define a session to use across multiple computational graphs
sess = tf.Session()

# Fetch a subset of the data in order to counter the limitation of compute resources
def get_train_data(size):
	train_data = mnist.train.images[:size,:]
	print("train data description: " + str(train_data.shape))
	train_labels = mnist.train.labels[:size,:]
	print("train labels description: " + str(train_labels.shape))	
	return train_data, train_labels

def get_test_data(size):
	test_data = mnist.test.images[:size,:]
	print("test data description: " + str(test_data.shape))
	test_labels = mnist.test.labels[:size,:]
	print("test labels description: " + str(test_data.shape))	
	return test_data, test_labels
	
def get_next_batch(size):
	# imitating the method described in mnist object to load a random data in batch
	# count the number of rows in the training data
	# train_data encapsulates both data [0] and labels [1]
	index = np.arange(0, train_data[0].shape[0])
	np.random.shuffle(index)
	index = index[:size]
	data_shuffled = np.asmatrix([train_data[0][i] for i in index])
	labels_shuffled = np.asmatrix([train_data[1][i] for i in index])
	return data_shuffled, labels_shuffled


def get_noisy_data(x, noise_factor):
	'''
	Add a Gaussian noise to the data
	Input:
	x : input original data
	noise_factor : self explanatory

	Returns:
	x + (noise(x)*noise_factor)
	'''
	full_noise = tf.random_uniform(shape=tf.shape(x), dtype=tf.float32)
	factored_noise = tf.multiply(full_noise, tf.cast(noise_factor, tf.float32))
	#return x*factored_noise + x*(1 - noise_factor)
	return tf.add(x, factored_noise)

def denoising_ae():
	'''
	Input: 
	A list whose:
	First element denotes the number of nodes in the input layer
	Second and subsequent elements denote number of nodes in the hidden layers 

	Output:
	x : placeholder for input data
	y : latent representation at the highest abstraction
	z : recontruction of pure input from the corrupted one
	loss : list of losses across multiple epoches
	'''

	# Input to the autoencoder
	dim_of_layer=[784, 512, 256, 64]
	x = tf.placeholder(tf.float32, shape=[None, dim_of_layer[0]])
	noise_factor = tf.placeholder(tf.float32, [1])
	noisy_input = get_noisy_data(x, noise_factor)

	# Encoder part: Iterate through all the output layers (except the first layer which is input)
	encoder = []
	for layer_index, layer_dim in enumerate(dim_of_layer[1:]):
		input_dim = int(noisy_input.get_shape()[1])
		output_dim = layer_dim
		W = tf.Variable(tf.random_uniform([input_dim, output_dim],-1.0 / math.sqrt(input_dim),1.0 / math.sqrt(input_dim)))

		#W = tf.Variable(tf.random_uniform([input_dim, output_dim]))
		b = tf.Variable(tf.zeros([output_dim]))
		encoder.append([W])
		activation = tf.nn.tanh(tf.matmul(noisy_input, W) + b)
		noisy_input = activation

	# Latent representation at the highest abstraction
	y = noisy_input

	# Decoder part: Iterate through all the output layers in REVERSE (except the first layer which is input)
	encoder.reverse()
	for layer_index, layer_dim in enumerate(dim_of_layer[::-1][1:]):
		input_dim = int(noisy_input.get_shape()[1])
		output_dim = layer_dim
		W = tf.transpose(encoder[layer_index])
		W = tf.reshape(W, [input_dim, output_dim])
		b = tf.Variable(tf.zeros([output_dim]))
		activation = tf.nn.tanh(tf.matmul(noisy_input, W) + b)
		noisy_input = activation
	z = noisy_input

	# RMS loss function
	loss = tf.sqrt(tf.reduce_mean(tf.square(z - x)))

	# Optimizer parameters
	learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	# Initialize all the variables
	sess.run(tf.global_variables_initializer())

	# Fit all training data 
	batch_size = 100
	n_epochs = 2000
	loss_per_epoch = []
	for epoch_i in range(n_epochs):
		for batch_i in range(train_data[0].shape[0] // batch_size):
			batch_input, _ = get_next_batch(batch_size)
			# introduces extra '1' dimension. Hence squeezing it to maintain dim consistency
			train = np.squeeze(np.array([img - mean_img for img in batch_input]))
			sess.run(optimizer, feed_dict={x:train, noise_factor:[1.0]})
		loss_per_epoch.append(sess.run(loss, feed_dict={x:train, noise_factor:[1.0]}))
		print(epoch_i, sess.run(loss, feed_dict={x:train, noise_factor:[1.0]}))


	#return(input, most abstracted latent representation, reconstructed input, noise_factor, loss)
	return {'x': x, 'y': y, 'z': z, 'noise_factor': noise_factor, 'loss': loss_per_epoch}

def reconstruct_mnist():
	# Plot reconstructed input for MNIST samples
	mnist_samples = 10
	testdata, _ = mnist.test.next_batch(mnist_samples)
	testdata_norm = np.array([img - mean_img for img in testdata])
	recon = sess.run(ae['z'], feed_dict={ae['x']:testdata_norm, ae['noise_factor']:[0.0]})
	fig, axis = plt.subplots(2, mnist_samples, figsize=(10, 4))
	for example_i in range(mnist_samples):
		axis[0][example_i].imshow(np.reshape(testdata[example_i, :], (28, 28)))
		axis[1][example_i].imshow(np.reshape([recon[example_i, :] + mean_img], (28, 28)))
	fig.show()
	plt.draw()
	plt.waitforbuttonpress()


if __name__ == '__main__':

	global train_data;
	global test_data;
	train_data = get_train_data(size=5000)
	test_data = get_test_data(size=5000)

	ae = denoising_ae()

	# Get loss curve
	print("printing loss curve:")
	plt.plot(ae['loss'])
	plt.xlabel("number of iterations")
	plt.ylabel("loss")
	plt.show()

	# Get sample recontructions
	reconstruct_mnist()

