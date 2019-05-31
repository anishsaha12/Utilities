import glob;
import numpy as np;
import tensorflow as tf;
import random;
import math;

### My utility functions
def mnet_train_batch (mnet, in_feats_batch, out_feats_batch, lr_value):
    [output_at_layers, gradient_at_layers] = mnet_backpropagate_and_probe(mnet, in_feats_batch, out_feats_batch);
    new_mnet, gradient_at_layers_of_weights_bias = mnet_update_parameters(mnet, lr_value, output_at_layers, gradient_at_layers);
    ### changes ANISH
    return new_mnet, gradient_at_layers_of_weights_bias;

def mnet_get_accuracy (fn_mnet, fn_in_feats, fn_out_labels):
    fn_pred_labels_dist = mnet_feed_forward(fn_mnet, fn_in_feats);
    fn_pred_labels = np.argmax(fn_pred_labels_dist, axis = 0);
    fn_pred_labels = fn_pred_labels.astype(int);
    fn_abs_diff = np.abs(fn_pred_labels - fn_out_labels);
    fn_num_errors = np.count_nonzero(fn_abs_diff);
    fn_accuracy = (len(fn_out_labels) - fn_num_errors + 0.0) * 100 / len(fn_out_labels);
    return fn_accuracy;

def mnet_get_ce_loss (fn_mnet, fn_in_feats, fn_out_labels):
    fn_pred_labels_dist = mnet_feed_forward(fn_mnet, fn_in_feats);
    fn_out_labels = fn_out_labels.astype(int);
    fn_pred_prob_for_correct_class = fn_pred_labels_dist[fn_out_labels, np.arange(len(fn_out_labels))];
    fn_ce_loss = -1.0 * np.log(fn_pred_prob_for_correct_class);
    return np.mean(fn_ce_loss);

def mnet_update_parameters (mnet_in, lr_value, fn_output_at_layers, fn_gradient_at_layers):
    # Assumption: First layer is a scale/shift layer and so, don't update its parameters
    mnet_out = mnet_in;
    fn_num_layers = len(mnet_out[0]);
    ### changes ANISH
    gradient_of_weights_n_bias_at_layers = list()
    ### changes ANISH END
    for idx in range(fn_num_layers):
        if (idx >= 1):
            mat_1 = fn_output_at_layers[idx-1];
            mat_2 = fn_gradient_at_layers[idx];
            gradient_weights = np.transpose(np.matmul(mat_1, mat_2));
            gradient_bias = np.sum(mat_2, axis = 0);
            ### changes ANISH
            gradient_of_weights_n_bias = list()
            gradient_of_weights_n_bias.append(gradient_weights)
            gradient_of_weights_n_bias.append(gradient_bias)
            gradient_of_weights_n_bias_at_layers.append(gradient_of_weights_n_bias)
            ### changes ANISH END
            if (mnet[3][idx]):
                mnet_out[0][idx] = mnet_out[0][idx] - (lr_value * gradient_weights);
                mnet_out[1][idx] = mnet_out[1][idx] - (lr_value * gradient_bias);
    ### changes ANISH
    return mnet_out,gradient_of_weights_n_bias_at_layers;

def mnet_backpropagate_and_probe (fn_mnet, fn_in_feats, fn_out_targets):
    output_at_layers = list();
    gradient_at_layers = list();
    fn_num_layers = len(fn_mnet[0]);
    fn_out_feats = np.transpose(fn_in_feats);
    for idx in range(fn_num_layers):
        fn_out_feats = np.matmul(fn_mnet[0][idx], fn_out_feats);
        fn_out_feats = np.add(fn_out_feats, fn_mnet[1][idx][:, np.newaxis]);
        if ('sigmoid' in fn_mnet[2][idx]):
            fn_out_feats = apply_sigmoid(fn_out_feats);
        elif ('softmax' in fn_mnet[2][idx]):
            fn_out_feats = apply_softmax(fn_out_feats);
        else:
            fn_out_feats = fn_out_feats;
        output_at_layers.append(fn_out_feats);
    predicted_output = np.transpose(fn_out_feats);
    for idx in reversed(range(fn_num_layers)):
        if ('softmax' in fn_mnet[2][idx]):
            ### changes ANISH
            gradient_at_layers.append(predicted_output - fn_out_targets);
            ### changes ANISH END
        elif ('sigmoid' in fn_mnet[2][idx]):
            ### changes ANISH
            # temp = apply_sigmoid(np.matmul(gradient_at_layers[-1], fn_mnet[0][idx + 1]));
            # gradient_at_layers.append(temp * (1 - temp));
            sigmoid_derivative = output_at_layers[idx] * (1.0 - output_at_layers[idx])
            downstream_gradient = np.matmul(gradient_at_layers[-1], fn_mnet[0][idx + 1])
            gradient_at_layers.append(downstream_gradient * np.transpose(sigmoid_derivative));
            ### changes ANISH END
        else:
            gradient_at_layers.append(np.ones([np.shape(gradient_at_layers[-1])[0], np.shape(fn_mnet[0][idx + 1])[1]]));
    gradient_at_layers = gradient_at_layers[::-1];
    return output_at_layers, gradient_at_layers;

def mnet_feed_forward (fn_mnet, fn_in_feats):
    fn_num_layers = len(fn_mnet[0]);
    fn_out_feats = np.transpose(fn_in_feats);
    for idx in range(fn_num_layers):
        fn_out_feats = np.matmul(fn_mnet[0][idx], fn_out_feats);
        fn_out_feats = np.add(fn_out_feats, fn_mnet[1][idx][:, np.newaxis]);
        if ('sigmoid' in fn_mnet[2][idx]):
            fn_out_feats = apply_sigmoid(fn_out_feats);
        elif ('softmax' in fn_mnet[2][idx]):
            fn_out_feats = apply_softmax(fn_out_feats);
        else:
            fn_out_feats = fn_out_feats;
    return fn_out_feats;

def apply_sigmoid (in_data):
    return (1.0 / (1.0 + np.exp(-1.0 * in_data)));

def apply_softmax(in_data):
    num_softmax = np.exp(in_data);
    den_softmax = np.sum(num_softmax, axis = 0);
    return num_softmax / den_softmax;

### Speaker-specific data variables
spk_id = 'MILE_0000320';
train_data_dir = './speaker_categorized_data/';

### Hyper-parameters of the source DNN
in_mnet = './pretrained_kaldi.mnet';
in_dim = 741;
out_dim = 3500;
batch_size = 128;

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




### Calculate accuracy
temp_accuracy = mnet_get_accuracy(mnet, in_feats, out_labels);
print ('Init Acc:',temp_accuracy);

### Calculate CE loss
temp_loss = mnet_get_ce_loss(mnet, in_feats, out_labels);
print ('Init Loss:',temp_loss);



gradients_history = list()

### Train the network
random_idx = list(range(num_egs));
num_batches = int(math.ceil(num_egs / batch_size));


#for epoch in range(num_epochs):
for epoch in range(3):
    # random.shuffle(random_idx);
    lr_value = lr_list[epoch];
    for batch in range(num_batches):
        start_idx = batch * batch_size;
        end_idx = start_idx + batch_size;
        sample_idx = random_idx[start_idx:min(num_egs, end_idx)];
        in_feats_batch = in_feats[sample_idx,:];
        out_labels_batch = out_labels[sample_idx];
        out_feats_batch = np.zeros([len(out_labels_batch), out_dim]);
        for idx in range(len(out_labels_batch)):
            out_feats_batch[idx, int(out_labels_batch[idx])] = 1.0;
        ### changes ANISH
        mnet, gradient_at_layers = mnet_train_batch(mnet, in_feats_batch, out_feats_batch, lr_value);
        gradients_history.append(gradient_at_layers)
        # break
    print('Acc:',mnet_get_accuracy(mnet, in_feats, out_labels))
    print('Loss:',mnet_get_ce_loss(mnet, in_feats, out_labels))