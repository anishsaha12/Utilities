import glob
import numpy as np
import random
import math

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


x_train, y_train = in_feats, out_feats

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def sigmoid_derv(s):
    return s * (1 - s)

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res

def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss

class MyNN:
    def __init__(self, x, y):
        self.x = x
        neurons = 128
        self.lr = 0.008
        ip_dim = x.shape[1]
        op_dim = y.shape[1]

        self.w1 = np.transpose(mnet[0][0])
        self.b1 = mnet[1][0]

        self.w2 = np.transpose(mnet[0][1])
        self.b2 = mnet[1][1]

        self.w3 = np.transpose(mnet[0][2])
        self.b3 = mnet[1][2]

        self.w4 = np.transpose(mnet[0][3])
        self.b4 = mnet[1][3]

        self.w5 = np.transpose(mnet[0][4])
        self.b5 = mnet[1][4]

        self.w6 = np.transpose(mnet[0][5])
        self.b6 = mnet[1][5]

        self.w7 = np.transpose(mnet[0][6])
        self.b7 = mnet[1][6]

        self.y = y

    def feedforward(self):
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = z1
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = sigmoid(z3)
        z4 = np.dot(self.a3, self.w4) + self.b4
        self.a4 = sigmoid(z4)
        z5 = np.dot(self.a4, self.w5) + self.b5
        self.a5 = sigmoid(z5)
        z6 = np.dot(self.a5, self.w6) + self.b6
        self.a6 = sigmoid(z6)
        z7 = np.dot(self.a6, self.w7) + self.b7
        self.a7 = softmax(z7)
        
    def backprop(self):
        loss = error(self.a7, self.y)
        print('Error :', loss)
        a7_delta = cross_entropy(self.a7, self.y) # w7

        z6_delta = np.dot(a7_delta, self.w7.T)
        a6_delta = z6_delta * sigmoid_derv(self.a6) #w6

        z5_delta = np.dot(a6_delta, self.w6.T)
        a5_delta = z5_delta * sigmoid_derv(self.a5) #w5

        z4_delta = np.dot(a5_delta, self.w5.T)
        a4_delta = z4_delta * sigmoid_derv(self.a4) #w4

        z3_delta = np.dot(a4_delta, self.w4.T)
        a3_delta = z3_delta * sigmoid_derv(self.a3) #w3

        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * sigmoid_derv(self.a2) # w2

        # print('_____LAYER O/P______')
        # print(self.a7)
        # print(self.a6)
        # print(self.a5)
        # print(self.a4)
        # print(self.a3)
        # print(self.a2)
        # print(self.a1)
        # print('_____GRAD______')
        # print(a7_delta)
        # print(a6_delta)
        # print(a5_delta)
        # print(a4_delta)
        # print(a3_delta)
        # print(a2_delta)

        self.w7 -= self.lr * np.dot(self.a6.T, a7_delta)
        self.b7 -= self.lr * np.sum(a7_delta, axis=0)
        self.w6 -= self.lr * np.dot(self.a5.T, a6_delta)
        self.b6 -= self.lr * np.sum(a6_delta, axis=0)
        self.w5 -= self.lr * np.dot(self.a4.T, a5_delta)
        self.b5 -= self.lr * np.sum(a5_delta, axis=0)
        self.w4 -= self.lr * np.dot(self.a3.T, a4_delta)
        self.b4 -= self.lr * np.sum(a4_delta, axis=0)
        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0)
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)

        # print('_____GRAD wrt WT______')
        # print(np.dot(self.a6.T, a7_delta))
        # print(np.dot(self.a5.T, a6_delta))
        # print(np.dot(self.a4.T, a5_delta))
        # print(np.dot(self.a3.T, a4_delta))
        # print(np.dot(self.a2.T, a3_delta))
        # print(np.dot(self.a1.T, a2_delta))
        # print('_____BIAS______')
        # print(np.sum(a7_delta, axis=0))
        # print(np.sum(a6_delta, axis=0))
        # print(np.sum(a5_delta, axis=0))
        # print(np.sum(a4_delta, axis=0))
        # print(np.sum(a3_delta, axis=0))
        # print(np.sum(a2_delta, axis=0))

    def predict(self, data):
        self.x = data
        self.feedforward()
        return self.a3.argmax()
            
size = 300
model = MyNN(x_train[range(size),:], np.array(y_train[range(size),:]))

epochs = 10
for x in range(epochs):
    model.feedforward()
    model.backprop()
        
# res = list()
# def get_acc(x, y):
#     acc = 0
#     for xx,yy in zip(x, y):
#         s = model.predict(xx)
#         res.append(s)
#         if s == np.argmax(yy):
#             acc +=1
#     print('\n',res,'\n')
#     return acc/len(x)*100
    
# print("Training accuracy : ", get_acc(x_train[range(1),:], np.array(y_train[range(1),:])))