import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

tf.random.set_seed(41)

from tensorflow.keras.layers import *
from data_load import *
import matplotlib.pyplot as plt


class Conv_Block(Layer):
    def __init__(self):
        filter_num = 128
        super(Conv_Block, self).__init__()
        self.c1 = Conv2D(filters=filter_num, kernel_size=(3, 3))
        self.act1 = PPNSPLUS(units=(1, 1, filter_num))
        #self.act1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.c2 = Conv2D(filters=filter_num, kernel_size=(3, 3))
        self.act2 = PPNSPLUS(units=(1, 1, filter_num))
        #self.act2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.c3 = Conv2D(filters=filter_num, kernel_size=(3, 3))
        self.act3 = PPNSPLUS(units=(1, 1, filter_num))
        #self.act3 = ReLU()

        self.c4 = Conv2D(filters=2, kernel_size=(3, 3))
        self.flatten = Flatten()
        self.fc1 = Dense(2,)

    def call(self, inputs, **kwargs):
        x = tf.cast(inputs, dtype=tf.float32)
        x = self.c1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.c2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.c3(x)
        x = self.act3(x)
        x = self.c4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

class LDA_cluster(Layer):
    def __init__(self):
        super(LDA_cluster, self).__init__()
        self.LDA = LDA_W()

    def call(self, label, pred):
        self.label = tf.argmax(label, axis=1)
        self.label_for_mask()
        inputs_0 = tf.boolean_mask(pred, self.mask_0)
        inputs_1 = tf.boolean_mask(pred, self.mask_1)
        mu_0 = tf.math.reduce_mean(inputs_0, axis=0, keepdims=True)
        mu_1 = tf.math.reduce_mean(inputs_1, axis=0, keepdims=True)

        between_s = tf.subtract(mu_0, mu_1)

        mean_dev_0 = tf.subtract(inputs_0, mu_0)
        cov_0 = tf.divide(tf.matmul(tf.transpose(mean_dev_0), mean_dev_0), tf.constant(len(inputs_0), dtype=tf.float32))
        mean_dev_1 = tf.subtract(inputs_1, mu_1)
        cov_1 = tf.divide(tf.matmul(tf.transpose(mean_dev_1), mean_dev_1), tf.constant(len(inputs_1), dtype=tf.float32))
        cov_sum = tf.add(cov_0, cov_1)

        loss = self.LDA(between_s, cov_sum)

        return loss

    def label_for_mask(self):
        self.mask_0 = self.label == 0
        self.mask_1 = self.label == 1


class Cluster_Loss(tf.keras.losses.Loss):
    def __init__(self):
        super(Cluster_Loss, self).__init__()

    def call(self, label, pred):
        return pred



class Standard_Centroid(tf.keras.losses.Loss):
    """학습 데이터를 통해 무게중심 구하기"""
    def __init__(self):
        super(Standard_Centroid, self).__init__()

    def call(self, label, pred):
        self.label = tf.argmax(label, axis=1)
        self.label_for_mask()
        inputs_0 = tf.boolean_mask(pred, self.mask_0)
        inputs_1 = tf.boolean_mask(pred, self.mask_1)
        center_0 = tf.math.reduce_mean(inputs_0, axis=0)
        center_1 = tf.math.reduce_mean(inputs_1, axis=0)

        return center_0, center_1

    def label_for_mask(self):
        self.mask_0 = self.label == 0
        self.mask_1 = self.label == 1


def scatter(predict, label, C0, C1, T_stat='Train', steps=None):
    predict = np.array(predict)
    idx = np.argmax(label, axis=1)

    alpha= 0.5
    plt.scatter(predict[idx == 0][:, 0], predict[idx == 0][:, 1], color="tab:red",alpha=alpha)
    plt.scatter(predict[idx == 1][:, 0], predict[idx == 1][:, 1], color="tab:blue",alpha=alpha)
    plt.scatter(C0[0], C0[1], color="k", marker='X')
    plt.scatter(C1[0], C1[1], color="tab:green", marker='X')
    #plt.xlim(-5,5)
    #plt.ylim(-5,5)
    plt.savefig("/home/gjtrj55/Tensorflow/" + str(T_stat) + str(steps))
    plt.clf()


def Distance_Accuracy(predict, C0, C1, label):
    pred = np.reshape(predict, (-1, 2))
    C0 = np.reshape(C0, (2,))
    C1 = np.reshape(C1, (2,))

    C0_pred_list = []
    C1_pred_list = []
    for idx in range(len(pred)):
        C0_pred_list.append(np.linalg.norm(C0 - pred[idx]))
        C1_pred_list.append(np.linalg.norm(C1 - pred[idx]))
    C_pred = np.c_[C0_pred_list, C1_pred_list]
    C_pred = np.argmin(C_pred, axis=1)

    y_idx = np.argmax(label, axis=1)
    correct = np.sum(y_idx == C_pred) / len(pred)
    return correct


if __name__ == '__main__':
    path = "/home/gjtrj55/Tensorflow/"
    """data loading"""
    (train_imgs, train_labels), (test_imgs, test_labels) = load_mnist(class_list=[4, 9])
    optimizer = tf.keras.optimizers.SGD()

    Conv_model = Conv_Block()
    LDA = LDA_cluster()
    Loss_obj = Cluster_Loss()


    batch_size = 200
    steps = int(np.ceil(len(train_labels)/batch_size))


    Train_Clustering_Loss = []
    Train_Clustering_Acc = []

    Test_Clustering_Loss = []
    Test_Clustering_Acc = []


    """초기 결과 값"""
    """Standard Centroid"""
    featur_out = Conv_model(train_imgs)
    C0, C1 = Standard_Centroid().call(label=train_labels, pred=featur_out)
    """Train data Check"""
    lda_loss = LDA.call(label=train_labels, pred=featur_out)
    loss = Loss_obj.call(label=train_labels, pred=lda_loss)
    Train_Clustering_Loss.append(loss)
    # Accuracy
    Acc = Distance_Accuracy(label=train_labels, predict=featur_out, C0=C0, C1=C1)
    Train_Clustering_Acc.append(Acc)
    print("init train acc : ", Acc)
    print("init train loss : ", loss)
    # Picture
    scatter(label=train_labels, predict=featur_out, C0=C0, C1=C1, T_stat='Train', steps=0)


    """Test data Check"""
    featur_out = Conv_model(test_imgs)
    lda_loss = LDA.call(label=test_labels, pred=featur_out)
    loss = Loss_obj.call(label=test_labels, pred=lda_loss)
    Test_Clustering_Loss.append(loss)
    # Accuracy
    Acc = Distance_Accuracy(label=test_labels, predict=featur_out, C0=C0, C1=C1)
    Test_Clustering_Acc.append(Acc)
    print("init test acc : ", Acc)
    print("init test loss : ", loss)
    # Picture
    scatter(label=test_labels, predict=featur_out, C0=C0, C1=C1, T_stat='Test', steps=0)



    for i in range(1):
        for idx in range(steps):
            """batch data"""
            print(idx, "steps")
            batch_train_imgs = np.array(train_imgs)[idx*batch_size:(idx+1)*batch_size]
            batch_train_label = np.array(train_labels)[idx*batch_size:(idx+1)*batch_size]

            """forward"""
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(Conv_model.variables)
                tape.watch(LDA.variables)
                featur_out = Conv_model(batch_train_imgs)
                lda_loss = LDA(label=batch_train_label, pred=featur_out)
                loss = Loss_obj(batch_train_label, lda_loss)

            """backward"""
            Conv_grads = tape.gradient(loss, Conv_model.variables)
            LDA_grads = tape.gradient(loss, LDA.variables)

            """update"""
            optimizer.apply_gradients(grads_and_vars=zip(Conv_grads, Conv_model.variables))
            optimizer.apply_gradients(grads_and_vars=zip(LDA_grads, LDA.variables))


            """Standard Centroid"""
            featur_out = Conv_model(train_imgs)
            C0, C1 = Standard_Centroid().call(label=train_labels, pred=featur_out)
            """Train data Check"""
            lda_loss = LDA.call(label=train_labels, pred=featur_out)
            loss = Loss_obj.call(label=train_labels, pred=lda_loss)
            Train_Clustering_Loss.append(loss)
            # Accuracy
            Acc = Distance_Accuracy(label=train_labels, predict=featur_out, C0=C0, C1=C1)
            Train_Clustering_Acc.append(Acc)
            print("train acc : ", Acc)
            print("train loss : ", loss)
            # Picture
            scatter(label=train_labels, predict=featur_out, C0=C0, C1=C1, T_stat='Train', steps=idx+1)


            """Test data Check"""
            featur_out = Conv_model(test_imgs)
            lda_loss = LDA.call(label=test_labels, pred=featur_out)
            loss = Loss_obj.call(label=test_labels, pred=lda_loss)
            Test_Clustering_Loss.append(loss)
            # Accuracy
            Acc = Distance_Accuracy(label=test_labels, predict=featur_out, C0=C0, C1=C1)
            Test_Clustering_Acc.append(Acc)
            print("test acc : ", Acc)
            print("test loss : ", loss)
            # Picture
            scatter(label=test_labels, predict=featur_out, C0=C0, C1=C1, T_stat='Test', steps=idx+1)


    with open(path + "Train_Clustering_Loss.txt", 'w') as f:
        for j in np.reshape(Train_Clustering_Loss, (-1)):
            f.write(str(j))
            f.write('\n')
    with open(path + "Train_Clustering_Acc.txt", 'w') as f:
        for j in np.reshape(Train_Clustering_Acc, (-1)):
            f.write(str(j))
            f.write('\n')


    with open(path + "Test_Clustering_Loss.txt", 'w') as f:
        for j in np.reshape(Test_Clustering_Loss, (-1)):
            f.write(str(j))
            f.write('\n')
    with open(path + "Test_Clustering_Acc.txt", 'w') as f:
        for j in np.reshape(Test_Clustering_Acc,(-1)):
            f.write(str(j))
            f.write('\n')
