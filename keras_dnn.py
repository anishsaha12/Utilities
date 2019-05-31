import glob
import numpy as np
import tensorflow as tf
import random
import math

tf.enable_eager_execution()
### Speaker-specific data variables
spk_id = 'MILE_0000320';
train_data_dir = './speaker_categorized_data/';

### Hyper-parameters of the source DNN
in_mnet = './pretrained_kaldi.mnet';
in_dim = 741;
out_dim = 3500;
batch_size = 1;

### Epoch and Learning rate scheduler
num_epochs = 16;
initial_lr = 0.008;
final_lr = 0.00000008;
lr_list = list();
for epoch in range(num_epochs):
    lr_value = 0;
    if (epoch == 0):
       lr_value = initial_lr;
    elif (math.floor(epoch / 3.0) == 0):
       lr_value = lr_list[-1] / 2.0;
    elif (math.floor(epoch / 3.0) == 1):
       lr_value = lr_list[-1] / 1.8;
    else:
       lr_value = lr_list[-1] / 1.5;
    lr_list.append(max(lr_value, final_lr));

### Load my network
mnet_lines = [line.rstrip() for line in open(in_mnet)];
mnet_wts = list();
mnet_bss = list();
mnet_fcs = list();
mnet_trs = list();
for line in mnet_lines:
    line_contents = line.split();
    if ('affine' in line_contents[0]):
        if ('n' in line_contents[1]):
            mnet_trs.append(False);
        else:
            mnet_trs.append(True);
        mnet_wts.append(np.loadtxt(line_contents[2]));
        mnet_bss.append(np.loadtxt(line_contents[3]));
        mnet_fcs.append(line_contents[4]);
mnet = (mnet_wts, mnet_bss, mnet_fcs, mnet_trs);
num_layers = len(mnet[0]);

### Load training data
ark_file_list = glob.glob(train_data_dir + spk_id + '/*.ark');
sen_file_list = glob.glob(train_data_dir + spk_id + '/*.sen');
ark_file_list.sort();
sen_file_list.sort();

in_feats = np.empty((0, in_dim));
out_labels = list();

for idx in range(len(ark_file_list)):
    feats_mat = np.loadtxt(ark_file_list[idx]);
    labels_vec = list(np.loadtxt(sen_file_list[idx]));
    in_feats = np.append(in_feats, feats_mat, axis = 0);
    out_labels.extend(labels_vec);
    break;
out_labels = np.array(out_labels);
num_egs = len(out_labels);

#one hot vectoring
out_feats = np.zeros([len(out_labels), out_dim])
for idx in range(len(out_labels)):
   out_feats[idx, int(out_labels[idx])] = 1.0;

# (x_train, y_train),(x_test, y_test) = 

model = tf.keras.models.Sequential()

for layer in range(num_layers):
    layer_shape = mnet[0][layer].shape
   #  model.layers[0].set_weights(w)
    model.add(tf.keras.layers.Dense(units=layer_shape[0], input_dim=layer_shape[1], activation=mnet[2][layer]))
    model.layers[layer].set_weights(list([np.transpose(mnet[0][layer]),mnet[1][layer]]))
    model.layers[layer].trainable = mnet[3][layer]


# Instantiate an optimizer.
optimizer = tf.keras.optimizers.SGD(lr=1e-3)
opt = tf.train.GradientDescentOptimizer(learning_rate=0.008)

# Instantiate a loss function.
loss_fn = tf.keras.losses.categorical_crossentropy

# Create Batches
in_feats = in_feats.astype('float32')
out_feats = out_feats.astype('float32')
train_dataset = tf.data.Dataset.from_tensor_slices((in_feats,out_feats))
# train_dataset = train_dataset.shuffle(buffer_size=1024)
train_dataset = train_dataset.batch(batch_size)

### Train the network
gradients_history = list()

for epoch in range(0):
   for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
      with tf.GradientTape() as tape:
         # Run the forward pass of the layer: operations of layers recorded on the GradientTape.
         logits = model(x_batch_train)
         # Compute the loss value for this minibatch. usind CE loss
         loss_value = loss_fn(y_batch_train, logits)
      # Use gradient tape to retrieve gradients of the trainable variables with respect to the loss
      grads = tape.gradient(loss_value, model.trainable_variables)
      gradients_history.append(grads)
      # Run one step of gradient descent by updating the value of the variables to minimize the loss.
      opt.apply_gradients(zip(grads, model.trainable_variables))
      # logits = model(x_batch_train)
      # loss_value = loss_fn(y_batch_train, logits)
      # def loss():
      #    return loss_value
      # grads_and_vars = opt.compute_gradients(loss, model.trainable_variables)
      # opt.apply_gradients(grads_and_vars)
      # grads = [gv[0] for gv in grads_and_vars]
      # gradients_history.append(grads)
      print('Training loss (for one batch) at step' ,step, ':', np.mean(loss_value.numpy()))
      break

      # Log every 10 batches.
      # if step % 10 == 0:
      #    print('Training loss (for one batch) at step' ,step, ':', loss_value)
      #    print('Seen so far: %s samples' % ((step + 1) * batch_size))