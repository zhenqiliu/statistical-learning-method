import pandas as pd
import numpy as np
import random
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class Knn(object):

    def __init__(self):
        self.k = 10
        
    def predict(self,testset,trainset,train_labels):
        predict = []
        count = 0

        for test_vec in testset:
            print(count)
            count += 1

            knn_list = []       
            max_index = -1      # index of max distance
            max_dist = 0        # max distance of current knn list

            # initiailize with first k points in training set
            for i in range(self.k):
                label = train_labels[i]
                train_vec = trainset[i]

                dist = np.linalg.norm(train_vec - test_vec)         # calc distance

                knn_list.append((dist,label))

            # process the rest data in training set 
            for i in range(self.k,len(train_labels)):
                label = train_labels[i]
                train_vec = trainset[i]

                dist = np.linalg.norm(train_vec - test_vec)         # calc distance 

                # find the max index in current knn list
                if max_index < 0:
                    for j in range(self.k):
                        if max_dist < knn_list[j][0]:
                            max_index = j
                            max_dist = knn_list[max_index][0]

                # replace the max index point with current point if its distance is smaller
                if dist < max_dist:
                    knn_list[max_index] = (dist,label)
                    max_index = -1
                    max_dist = 0


            # vote
            class_total = 10
            class_count = [0] * class_total
            for dist,label in knn_list:
                class_count[label] += 1
            max_votes = max(class_count)

           # find the label with the max votes
            for i in range(class_total):
                if max_votes == class_count[i]:
                    predict.append(i)
                    break

        return np.array(predict)


if __name__ == '__main__':

    print('Start read data')

    time_1 = time.time()

    raw_data = pd.read_csv('../../data/mnist/train.csv',header=0)
    data = raw_data.values

    imgs = data[0::,1::]
    labels = data[::,0]

    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=23323)
    print(train_features.shape)

    time_2 = time.time()
    print('read data cost ',time_2 - time_1,' second','\n')

    print('Start training')
    print('No training needed for knn')
    time_3 = time.time()
    print('training cost ',time_3 - time_2,' second','\n')

    print('Start predicting')
    knn = Knn()
    test_predict = knn.predict(test_features,train_features,train_labels)
    time_4 = time.time()
    print('predicting cost ',time_4 - time_3,' second','\n')

    score = accuracy_score(test_labels,test_predict)
    print("The accruacy score is ", score)
