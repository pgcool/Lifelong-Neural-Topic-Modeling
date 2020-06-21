import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


def vectors(model, data, session):
	vecs = []
	for _, x, seq_lengths in data:
		vecs.extend(
			session.run([model.h], feed_dict={
				model.x: x,
				model.seq_lengths: seq_lengths
			})[0]
		)
	return np.array(vecs)

def vectors_bidirectional(model, data, session, combination_type):
	vecs = []
	if combination_type == "concat":
		for _, x, x_bw, seq_lengths in data:
			vecs.extend(
				session.run([model.h_comb_concat], feed_dict={
					model.x: x,
					model.x_bw: x_bw,
					model.seq_lengths: seq_lengths
				})[0]
			)
	elif combination_type == "sum":
		for _, x, x_bw, seq_lengths in data:
			vecs.extend(
				session.run([model.h_comb_sum], feed_dict={
					model.x: x,
					model.x_bw: x_bw,
					model.seq_lengths: seq_lengths
				})[0]
			)
	else:
		print("vectors function: Invalid value for combination_type.")
	return np.array(vecs)


def loss(model, data, session):
	loss = []
	for _, x, seq_lengths in data:
		loss.append(
			session.run([model.loss], feed_dict={
				model.x: x,
				model.seq_lengths: seq_lengths
			})[0]
		)
	return sum(loss) / len(loss)


def gradients(opt, loss, vars, step, max_gradient_norm=None, dont_clip=[]):
	gradients = opt.compute_gradients(loss, vars)
	if max_gradient_norm is not None:
		to_clip = [(g, v) for g, v in gradients if v.name not in dont_clip]
		not_clipped = [(g, v) for g, v in gradients if v.name in dont_clip]
		gradients, variables = zip(*to_clip)
		clipped_gradients, _ = clip_ops.clip_by_global_norm(
			gradients,
			max_gradient_norm
		)
		gradients = list(zip(clipped_gradients, variables)) + not_clipped

	# Add histograms for variables, gradients and gradient norms
	for gradient, variable in gradients:
		if isinstance(gradient, ops.IndexedSlices):
			grad_values = gradient.values
		else:
			grad_values = gradient
		if grad_values is None:
			print('warning: missing gradient: {}'.format(variable.name))
		if grad_values is not None:
			tf.summary.histogram(variable.name, variable)
			tf.summary.histogram(variable.name + '/gradients', grad_values)
			tf.summary.histogram(
				variable.name + '/gradient_norm',
				clip_ops.global_norm([grad_values])
			)

	return opt.apply_gradients(gradients, global_step=step)


def linear(input, output_dim, input_bw=None, scope=None, stddev=None, W_initializer=None):
	const = tf.constant_initializer(0.0)

	if W_initializer is None:
		if stddev:
			norm = tf.random_normal_initializer(stddev=stddev)
		else:
			norm = tf.random_normal_initializer(
				stddev=np.sqrt(2.0 / input.get_shape()[1].value)
			)

		with tf.variable_scope(scope or 'linear'):
			w = tf.get_variable(
				'w',
				[input.get_shape()[1], output_dim],
				initializer=norm
			)
	else:
		w = tf.get_variable(
			'w',
			initializer=tf.transpose(W_initializer)
		)

	b = tf.get_variable('b', [output_dim], initializer=const)
	b_bw = tf.get_variable('b_bw', [output_dim], initializer=const)

	input_bw_logits = None
	if input_bw is None:
		input_logits = tf.nn.xw_plus_b(input, w, b)
	else:
		input_logits = tf.nn.xw_plus_b(input, w, b)
		input_bw_logits = tf.nn.xw_plus_b(input_bw, w, b_bw)
	
	return input_logits, input_bw_logits


def linear_TL(input, output_dim, input_old=None, scope=None, stddev=None, U_pretrained=None, bias_sharing=True):
	const = tf.constant_initializer(0.0)

	if U_pretrained is None:
		if stddev:
			norm = tf.random_normal_initializer(stddev=stddev)
		else:
			norm = tf.random_normal_initializer(
				stddev=np.sqrt(2.0 / input.get_shape()[1].value)
			)
		
		U = tf.get_variable(
			'U',
			[input.get_shape()[1], output_dim],
			initializer=norm
		)
	else:
		U = tf.get_variable(
			'U',
			initializer=U_pretrained
		)

	b = tf.get_variable('b', [output_dim], initializer=const)

	if not bias_sharing:
		b_old = tf.get_variable('b_old', [len(input_old), output_dim], initializer=const)

	input_logits = tf.nn.xw_plus_b(input, U, b)
	input_old_logits = []
	if input_old:
		for i, input_old_temp in enumerate(input_old):
			if bias_sharing:
				input_old_temp_logits = tf.nn.xw_plus_b(input_old_temp, U, b)
			else:
				input_old_temp_logits = tf.nn.xw_plus_b(input_old_temp, U, b_old[i])
			input_old_logits.append(input_old_temp_logits)
	
	return input_logits, input_old_logits, U


def linear_reload(input, output_dim, input_bw=None, scope=None, stddev=None, W_initializer=None, 
					V_reload=None, b_reload=None, b_bw_reload=None):
	w = tf.Variable(
		initial_value=V_reload,
		trainable=False
	)
	b = tf.Variable(
		initial_value=b_reload,
		trainable=False
	)

	if input_bw is None:
		b_bw = None
	else:
		b_bw = tf.Variable(
			initial_value=b_bw_reload,
			trainable=False
		)

	input_bw_logits = None
	if input_bw is None:
		input_logits = tf.nn.xw_plus_b(input, w, b)
	else:
		input_logits = tf.nn.xw_plus_b(input, w, b)
		input_bw_logits = tf.nn.xw_plus_b(input_bw, w, b_bw)
	
	return input_logits, input_bw_logits


def linear_TL_reload(input, output_dim, input_old=None, scope=None, stddev=None, W_initializer=None, 
					V_reload=None, b_reload=None, b_old_reload=None):
	w = tf.Variable(
		initial_value=V_reload,
		trainable=False
	)
	b = tf.Variable(
		initial_value=b_reload,
		trainable=False
	)

	if not b_old_reload is None:
		b_old = tf.Variable(
			initial_value=b_old_reload,
			trainable=False
		)

	input_old_logits = None
	if input_old is None:
		input_logits = tf.nn.xw_plus_b(input, w, b)
	else:
		input_logits = tf.nn.xw_plus_b(input, w, b)

		if b_old_reload is None:
			input_old_logits = tf.nn.xw_plus_b(input_old, w, b)
		else:
			input_old_logits = tf.nn.xw_plus_b(input_old, w, b_old)
	
	return input_logits, input_old_logits


def masked_sequence_cross_entropy_loss(
	x,
	seq_lengths,
	logits,
	loss_function=None,
	norm_by_seq_lengths=True,
	name=""
):
	'''
	Compute the cross-entropy loss between all elements in x and logits.
	Masks out the loss for all positions greater than the sequence
	length (as we expect that sequences may be padded).

	Optionally, also either use a different loss function (eg: sampled
	softmax), and/or normalise the loss for each sequence by the
	sequence length.
	'''
	batch_size = tf.shape(x)[0]
	labels = tf.reshape(x, [-1])

	#max_doc_length = tf.reduce_max(seq_lengths)
	max_doc_length = tf.shape(x)[1]
	mask = tf.less(
		tf.range(0, max_doc_length, 1),
		tf.reshape(seq_lengths, [batch_size, 1])
	)
	mask = tf.reshape(mask, [-1])
	mask = tf.to_float(tf.where(
		mask,
		tf.ones_like(labels, dtype=tf.float32),
		tf.zeros_like(labels, dtype=tf.float32)
	))

	if loss_function is None:
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=logits,
			labels=labels
		)
	else:
		loss = loss_function(logits, labels)
	loss *= mask
	loss = tf.reshape(loss, [batch_size, -1])
	loss = tf.reduce_sum(loss, axis=1)
	loss_unnormed = loss
	if norm_by_seq_lengths:
		loss = loss / tf.to_float(seq_lengths)
	return tf.reduce_mean(loss, name="loss_normed_" + name), labels, mask, tf.reduce_mean(loss_unnormed, name="loss_unnormed_" + name)


def masked_sequence_cross_entropy_loss_sal(
	x,
	seq_lengths,
	logits,
	loss_function=None,
	norm_by_seq_lengths=True
):
	'''
	Compute the cross-entropy loss between all elements in x and logits.
	Masks out the loss for all positions greater than the sequence
	length (as we expect that sequences may be padded).

	Optionally, also either use a different loss function (eg: sampled
	softmax), and/or normalise the loss for each sequence by the
	sequence length.
	'''
	batch_size = tf.shape(x)[0]
	labels = tf.reshape(x, [-1])
	
	max_doc_length = tf.shape(x)[1]
	mask = tf.less(
		tf.range(0, max_doc_length, 1),
		tf.reshape(seq_lengths, [batch_size, 1])
	)
	mask = tf.reshape(mask, [-1])
	mask = tf.to_float(tf.where(
		mask,
		tf.ones_like(labels, dtype=tf.float32),
		tf.zeros_like(labels, dtype=tf.float32)
	))

	if loss_function is None:
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=logits,
			labels=labels
		)
	else:
		loss = loss_function(logits, labels)
	loss *= mask
	loss = tf.reshape(loss, [batch_size, -1])
	loss = tf.reduce_sum(loss, axis=1)
	loss_unnormed = loss
	if norm_by_seq_lengths:
		loss = loss / tf.to_float(seq_lengths)
	return tf.identity(loss, name="loss_normed_vec_sal"), labels, mask, tf.identity(loss_unnormed, name="loss_unnormed_vec_sal")


class MLP(object):
	def __init__(self, input, labels, num_classes=10, hidden_sizes=[]):
		self.input = input
		self.labels = labels
		self.num_layers = len(hidden_sizes)
		self.num_classes = num_classes
		self.hidden_sizes = hidden_sizes

		self.U_list = []
		self.d_list = []
		hidden_sizes.append(num_classes)

		for i in range(self.num_layers):
			max_U_init = 1.0 / (hidden_sizes[i] * hidden_sizes[i+1])
			U = tf.get_variable(
				'U_' + str(i),
				[hidden_sizes[i], hidden_sizes[i+1]],
				initializer=tf.random_uniform_initializer(
					maxval=max_U_init
				)
			)
			d = tf.get_variable(
				'd_' + str(i),
				[hidden_sizes[i+1]],
				initializer=tf.constant_initializer(0)
			)
			self.U_list.append(U)
			self.d_list.append(d)

		# Forward pass
		temp = tf.matmul(input, self.U_list[0]) + self.d_list[0]
		for i in range(1, self.num_layers):
			temp = tf.matmul(temp, self.U_list[i]) + self.d_list[i]
		disc_logits = temp

		one_hot_labels = tf.one_hot(labels, depth=num_classes)

		self.pred_labels = tf.argmax(disc_logits, axis=1)

		self.disc_loss = tf.losses.softmax_cross_entropy(
			onehot_labels=one_hot_labels,
			logits=disc_logits,
		)

		self.disc_accuracy = tf.metrics.accuracy(labels, self.pred_labels)
		self.disc_output = disc_logits


class DocNADE_TL(object):
	def __init__(self, x, y, x_old, x_old_doc_ids, seq_lengths, 
				 seq_lengths_old, params, x_old_loss,  W_old_list=None, U_old_list=None, 
				 W_embeddings_matrices_list=None, W_old_indices_list=None, 
				 lambda_embeddings_list=None, W_new_indices_list=None,
				 W_pretrained=None, U_pretrained=None):
		self.x = x
		self.y = y
		self.x_old = x_old
		self.x_old_doc_ids = x_old_doc_ids
		self.seq_lengths = seq_lengths
		self.seq_lengths_old = seq_lengths_old
		self.x_old_loss = x_old_loss

		batch_size = tf.shape(x)[0]
		self.b_s = tf.shape(x)
		self.W_old_list = W_old_list
		self.U_old_list = U_old_list
		self.lambda_embeddings_list = lambda_embeddings_list

		x_old_list = tf.unstack(x_old, axis=0, name='x_old_list')
		self.x_old_list = x_old_list
		seq_lengths_old_list = tf.unstack(seq_lengths_old, axis=0, name='seq_lengths_old_list')
		self.seq_lengths_old_list = seq_lengths_old_list

		# Do an embedding lookup for each word in each sequence
		with tf.device('/cpu:0'):
			if W_pretrained is None:
				max_embed_init = 1.0 / (params.vocab_size * params.hidden_size)
				W = tf.get_variable(
					'embedding',
					[params.vocab_size, params.hidden_size],
					initializer=tf.random_uniform_initializer(
						maxval=max_embed_init
					)
				)
			else:
				W = tf.get_variable(
					'embedding',
					initializer=W_pretrained
				)
			self.embeddings = tf.nn.embedding_lookup(W, x)
			#self.embeddings_old = tf.nn.embedding_lookup(W, x_old)
			self.embeddings_old_list = []
			
			for x_old_temp in x_old_list:
				embeddings_old_temp = tf.nn.embedding_lookup(W, x_old_temp)
				self.embeddings_old_list.append(embeddings_old_temp)
		
			bias = tf.get_variable(
				'bias',
				[params.hidden_size],
				initializer=tf.constant_initializer(0)
			)

			if not params.bias_sharing:
				bias_old = tf.get_variable(
					'bias_old',
					[len(params.sal_gamma_init), params.hidden_size],
					initializer=tf.constant_initializer(0)
				)

			if params.use_embeddings_prior:
				if params.lambda_embeddings == "manual":
					self.embeddings_lambda_list = tf.get_variable(
						'embeddings_lambda_list',
						initializer=lambda_embeddings_list,
						trainable=False
					)
				elif params.lambda_embeddings == "automatic":
					embeddings_lambda_list = tf.get_variable(
						'embeddings_lambda_list_unclipped',
						initializer=lambda_embeddings_list,
						trainable=True
					)
					self.embeddings_lambda_list = tf.clip_by_value(embeddings_lambda_list, 0.0, 1.0, name='embeddings_lambda_list')
				else:
					print("Invalid parameter value for lambda_embeddings: ", params.lambda_embeddings)
					sys.exit()

				self.W_prior_list = []
				self.embeddings_prior_list = []
				for i, W_embeddings in enumerate(W_embeddings_matrices_list):
					W_prior = tf.get_variable(
						'embedding_prior_' + str(i),
						initializer=W_embeddings,
						trainable=False
					)
					embedding_prior = tf.nn.embedding_lookup(W_prior, x)
					embedding_prior_with_lambda = tf.scalar_mul(self.embeddings_lambda_list[i], embedding_prior)
					self.embeddings = tf.add(self.embeddings, embedding_prior_with_lambda)

					self.W_prior_list.append(W_prior)
					self.embeddings_prior_list.append(embedding_prior)


			if params.sal_loss:
				if params.sal_gamma == "manual":
					self.sal_gamma = tf.get_variable(
						'sal_gamma',
						initializer=params.sal_gamma_init,
						trainable=False
					)
				elif params.sal_gamma == "automatic":
					"""
					sal_gamma = tf.get_variable(
						'sal_gamma_unclipped',
						[params.old_dataset_size, 1],
						initializer=tf.constant_initializer(params.sal_gamma_init),
						trainable=True
					)
					self.sal_gamma = tf.clip_by_value(sal_gamma, 0.0, 1.0, name='sal_gamma')
					"""
					print("Invalid parameter value for sal_gamma: ", params.sal_gamma)
					sys.exit()
				else:
					print("Invalid parameter value for sal_gamma: ", params.sal_gamma)
					sys.exit()

			if params.ll_loss:
				if params.ll_lambda == "manual":
					self.ll_lambda_list = tf.get_variable(
						'll_lambda_list',
						initializer=params.ll_lambda_init,
						trainable=False
					)
				elif params.ll_lambda == "automatic":
					"""
					ll_lambda_list = tf.get_variable(
						'll_lambda_list_unclipped',
						initializer=params.ll_lambda_init,
						trainable=True
					)
					self.ll_lambda_list = tf.clip_by_value(ll_lambda_list, 0.0, 1.0, name='ll_lambda_list')
					"""
					print("Invalid parameter value for sal_gamma: ", params.sal_gamma)
					sys.exit()
				else:
					print("Invalid parameter value for sal_gamma: ", params.sal_gamma)
					sys.exit()

				if params.projection:
					self.ll_proj_matrices_W = []
					self.ll_proj_matrices_U = []

					max_embed_init = 1.0 / (params.hidden_size * params.hidden_size)

					for i, ll_temp_init in enumerate(params.ll_lambda_init):
						ll_proj_temp_W = tf.get_variable(
							'll_projection_W_' + str(i),
							[params.hidden_size, params.hidden_size],
							initializer=tf.random_uniform_initializer(
								maxval=max_embed_init
							)
						)
						self.ll_proj_matrices_W.append(ll_proj_temp_W)

						ll_proj_temp_U = tf.get_variable(
							'll_projection_U_' + str(i),
							[params.hidden_size, params.hidden_size],
							initializer=tf.random_uniform_initializer(
								maxval=max_embed_init
							)
						)
						self.ll_proj_matrices_U.append(ll_proj_temp_U)

		# Compute the hidden layer inputs: each gets summed embeddings of
		# previous words
		def sum_embeddings(previous, current):
			return previous + current

		h = tf.scan(sum_embeddings, tf.transpose(self.embeddings, [1, 2, 0]))
		h = tf.transpose(h, [2, 0, 1])

		# add initial zero vector to each sequence, will then generate the
		# first element using just the bias term
		h = tf.concat([
			tf.zeros([batch_size, 1, params.hidden_size], dtype=tf.float32), h
		], axis=1)
		
		self.pre_act = h
		self.pre_act_old_list = []

		# Apply activation
		if params.bias_sharing:
			if params.activation == 'sigmoid':
				h = tf.sigmoid(h + bias)
			elif params.activation == 'tanh':
				h = tf.tanh(h + bias)
			elif params.activation == 'relu':
				h = tf.nn.relu(h + bias)
			else:
				print('Invalid value for activation: %s' % (params.activation))
				exit()
		else:
			if params.activation == 'sigmoid':
				h = tf.sigmoid(h + bias)
			elif params.activation == 'tanh':
				h = tf.tanh(h + bias)
			elif params.activation == 'relu':
				h = tf.nn.relu(h + bias)
			else:
				print('Invalid value for activation: %s' % (params.activation))
				exit()
		
		self.aft_act = h
		self.aft_act_old_list = []

		# Extract final state for each sequence in the batch
		indices = tf.stack([
			tf.range(batch_size),
			tf.to_int32(seq_lengths)
		], axis=1)
		self.indices = indices
		self.h = tf.gather_nd(h, indices, name='last_hidden')
		
		self.h_old_last_list = []

		h = h[:, :-1, :]
		h = tf.reshape(h, [-1, params.hidden_size])

		h_old_list = []
		for i, embeddings_old in enumerate(self.embeddings_old_list):
			h_old_temp = tf.scan(sum_embeddings, tf.transpose(embeddings_old, [1, 2, 0]))
			h_old_temp = tf.transpose(h_old_temp, [2, 0, 1])

			h_old_temp = tf.concat([
				tf.zeros([batch_size, 1, params.hidden_size], dtype=tf.float32), h_old_temp
			], axis=1)
			self.pre_act_old_list.append(h_old_temp)

			# Apply activation
			if params.bias_sharing:
				if params.activation == 'sigmoid':
					h_old_temp = tf.sigmoid(h_old_temp + bias)
				elif params.activation == 'tanh':
					h_old_temp = tf.tanh(h_old_temp + bias)
				elif params.activation == 'relu':
					h_old_temp = tf.nn.relu(h_old_temp + bias)
				else:
					print('Invalid value for activation: %s' % (params.activation))
					exit()
			else:
				if params.activation == 'sigmoid':
					h_old_temp = tf.sigmoid(h_old_temp + bias_old[i])
				elif params.activation == 'tanh':
					h_old_temp = tf.tanh(h_old_temp + bias_old[i])
				elif params.activation == 'relu':
					h_old_temp = tf.nn.relu(h_old_temp + bias_old[i])
				else:
					print('Invalid value for activation: %s' % (params.activation))
					exit()

			indices_old_temp = tf.stack([
				tf.range(batch_size),
				tf.to_int32(seq_lengths_old_list[i])
			], axis=1)
			self.h_old_last_list.append(tf.gather_nd(h_old_temp, indices_old_temp, name='last_hidden_old_' + str(i)))

			h_old_temp = h_old_temp[:, :-1, :]
			h_old_temp = tf.reshape(h_old_temp, [-1, params.hidden_size])

			h_old_list.append(h_old_temp)

		###################### Softmax logits ###############################

		if not params.num_samples:
			self.logits, self.logits_old, U_new = linear_TL(h, params.vocab_size, input_old=h_old_list, scope='softmax', U_pretrained=U_pretrained, bias_sharing=params.bias_sharing)
			loss_function = None
		else:
			print('Invalid value for params.num_samples: %s' % (params.num_samples))
			exit()

		# LL(x_new|theta_new) [in scalar form]
		self.loss_normed, self.labels, self.mask, self.loss_unnormed = masked_sequence_cross_entropy_loss(
			x,
			seq_lengths,
			self.logits,
			loss_function=loss_function,
			norm_by_seq_lengths=True,
			name="x"
		)

		# Pretraining optimiser
		if params.pretraining_target:
			pretrain_step = tf.Variable(0, trainable=False)
			self.pretrain_opt = gradients(
				opt=tf.train.AdamOptimizer(learning_rate=params.learning_rate),
				loss=self.loss_unnormed,
				vars=tf.trainable_variables(),
				step=pretrain_step
			)

		self.total_loss = tf.identity(self.loss_unnormed, name="total_loss")

		if params.ll_loss:
			W_reg_loss = 0.0
			for i, (W_old, W_new_indices) in enumerate(zip(W_old_list, W_new_indices_list)):
				W_new_temp = tf.gather(W, W_new_indices, axis=0)
				if params.projection:
					W_new_temp_proj = tf.matmul(W_new_temp, self.ll_proj_matrices_W[i])
				else:
					W_new_temp_proj = W_new_temp
				W_l2_loss = tf.nn.l2_loss((W_new_temp_proj - W_old), name='l2_loss_W_old_' + str(i))
				W_reg_loss += self.ll_lambda_list[i] * W_l2_loss

			U_reg_loss = 0.0
			for i, (U_old, W_new_indices) in enumerate(zip(U_old_list, W_new_indices_list)):
				U_new_temp = tf.gather(U_new, W_new_indices, axis=1)
				if params.projection:
					U_new_temp_proj = tf.matmul(self.ll_proj_matrices_U[i], U_new_temp)
				else:
					U_new_temp_proj = U_new_temp
				U_l2_loss = tf.nn.l2_loss((U_new_temp_proj - U_old), name='l2_loss_U_old_' + str(i))
				U_reg_loss += self.ll_lambda_list[i] * U_l2_loss
			
			self.total_reg_loss = W_reg_loss + U_reg_loss
			self.total_loss += self.total_reg_loss
			self.W_new_temp = W_new_temp
			self.U_new_temp = U_new_temp

		self.final_sal_loss_list = []
		self.final_sal_loss_mean_list = []
		self.sal_gamma_mask_list = []

		if params.sal_loss:
			# LL(x_old|theta_new) [in vector form]
			for i, logits_old_temp in enumerate(self.logits_old):
				self.loss_normed_old_temp_sal, self.labels_old_temp_sal, self.mask_old_temp_sal, self.loss_unnormed_old_temp_sal = masked_sequence_cross_entropy_loss_sal(
					x_old_list[i],
					seq_lengths_old_list[i],
					#self.logits_old,
					logits_old_temp,
					loss_function=loss_function,
					norm_by_seq_lengths=True
				)
				
				if params.sal_gamma == "automatic":
					sys.exit()
				else:
					ppl_old_temp = tf.exp(self.loss_normed_old_temp_sal)

					sal_gamma_mask = tf.less(
						ppl_old_temp,
						tf.convert_to_tensor(params.sal_threshold_list[i])
					)

					sal_gamma_mask = tf.reshape(sal_gamma_mask, [-1])

					sal_gamma_mask = tf.to_float(tf.where(
						sal_gamma_mask,
						tf.ones_like(self.loss_unnormed_old_temp_sal, dtype=tf.float32),
						tf.zeros_like(self.loss_unnormed_old_temp_sal, dtype=tf.float32)
					))

					final_sal_loss_temp = tf.multiply(sal_gamma_mask, self.loss_unnormed_old_temp_sal)
					final_sal_loss_temp_mean = tf.reduce_mean(final_sal_loss_temp)

					self.total_loss += (self.sal_gamma[i] * final_sal_loss_temp_mean)
					self.final_sal_loss_list.append(final_sal_loss_temp)
					self.final_sal_loss_mean_list.append(final_sal_loss_temp_mean)
					self.sal_gamma_mask_list.append(sal_gamma_mask)

		# Optimiser
		step = tf.Variable(0, trainable=False)
		self.opt = gradients(
			opt=tf.train.AdamOptimizer(learning_rate=params.learning_rate),
			loss=self.total_loss,
			vars=tf.trainable_variables(),
			step=step
		)


class DocNADE_TL_reload(object):
	def __init__(self, x, y, seq_lengths, params, W_initializer=None,
				 W_reload=None, W_embeddings_reload=None, W_prior_proj_reload=None, bias_reload=None, 
				 bias_bw_reload=None, V_reload=None, b_reload=None, b_bw_reload=None,
				 W_list_reload=None, bias_list_reload=None, lambda_embeddings_reload=None):
		self.x = x
		self.y = y
		self.seq_lengths = seq_lengths

		batch_size = tf.shape(x)[0]
		self.b_s = tf.shape(x)

		self.V = V_reload
		self.b = b_reload
		self.W = W_reload

		self.lambda_embeddings_list = lambda_embeddings_reload

		# Do an embedding lookup for each word in each sequence
		#with tf.device('/cpu:0'):
		W = tf.Variable(
			initial_value=W_reload,
			trainable=False
		)
		self.embeddings = tf.nn.embedding_lookup(W, x)

		if params['use_embeddings_prior']:
			self.embeddings_lambda_list = tf.Variable(
				initial_value=lambda_embeddings_reload,
				trainable=False
			)

			self.W_prior_list = []
			for i, W_embeddings in enumerate(W_embeddings_reload):
				W_prior = tf.Variable(
					initial_value=W_embeddings,
					trainable=False
				)
				embedding_prior = tf.nn.embedding_lookup(W_prior, x)
				embedding_prior_with_lambda = tf.scalar_mul(self.embeddings_lambda_list[i], embedding_prior)
				self.embeddings = tf.add(self.embeddings, embedding_prior_with_lambda)
				
				self.W_prior_list.append(W_prior)

		bias = tf.Variable(
			initial_value=bias_reload,
			trainable=False
		)

		# Compute the hidden layer inputs: each gets summed embeddings of
		# previous words
		def sum_embeddings(previous, current):
			return previous + current

		h = tf.scan(sum_embeddings, tf.transpose(self.embeddings, [1, 2, 0]))
		h = tf.transpose(h, [2, 0, 1])

		# add initial zero vector to each sequence, will then generate the
		# first element using just the bias term
		h = tf.concat([
			tf.zeros([batch_size, 1, params['hidden_size']], dtype=tf.float32), h
		], axis=1)
		self.pre_act = h

		# Apply activation
		if params['activation'] == 'sigmoid':
			h = tf.sigmoid(h + bias)
		elif params['activation'] == 'tanh':
			h = tf.tanh(h + bias)
		elif params['activation'] == 'relu':
			h = tf.nn.relu(h + bias)
		else:
			print('Invalid value for activation: %s' % (params['activation']))
			exit()
		self.aft_act = h

		# Extract final state for each sequence in the batch
		indices = tf.stack([
			tf.range(batch_size),
			tf.to_int32(seq_lengths)
		], axis=1)
		self.indices = indices
		self.h = tf.gather_nd(h, indices)

		h = h[:, :-1, :]
		h = tf.reshape(h, [-1, params['hidden_size']])
		
		###################### Softmax logits ###############################

		self.logits, _ = linear_TL_reload(h, params['vocab_size'], scope='softmax', W_initializer=None, 
										V_reload=V_reload, b_reload=b_reload)
		loss_function = None

		self.loss_normed, self.labels, self.mask, self.loss_unnormed = masked_sequence_cross_entropy_loss(
			x,
			seq_lengths,
			self.logits,
			loss_function=loss_function,
			norm_by_seq_lengths=True
		)