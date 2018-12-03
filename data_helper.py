#coding=utf-8
import numpy as np
from config import config
import csv

class Dataset(object):
    def __init__(self, data_source):
        self.data_source = data_source
        self.index_in_epoch = 0
        self.alphabet = config.alphabet
        self.alphabet_size = config.alphabet_size
        self.num_classes = config.nums_classes
        self.l0 = config.l0
        self.epochs_completed = 0
        self.batch_size = config.batch_size
        self.example_nums = config.example_nums
        self.doc_image = []
        self.label_image = []

    def next_batch(self):
        start = self.index_in_epoch
        self.index_in_epoch += self.batch_size
        if self.index_in_epoch > self.example_nums:
            self.epochs_completed += 1

            perm = np.arange(self.example_nums)
            np.random.shuffle(perm)
            self.doc_image = self.doc_image[perm]
            self.label_image = self.label_image[perm]

            start = 0
            self.index_in_epoch = self.batch_size
            assert self.batch_size < self.example_nums

        end = self.index_in_epoch
        batch_x = np.array(self.doc_image[start:end], dtype='int64')
        batch_y = np.array(self.label_image[start:end], dtype='float32')

        return batch_x, batch_y

    def dataset_read(self):
        # doc_vec indicates all characters in one doc, dco_image indicates all docs
        # label_class indicates num of classes
        # doc_count indicates num of docs
        docs = []
        label = []
        doc_count = 0
        csvfile = open(self.data_source, 'r')
        for line in csv.reader(csvfile, delimiter=',', quotechar='"'):
            content = line[1] + '. ' + line[2]
            docs.append(content.lower())
            label.append(line[0])
            doc_count = doc_count + 1

        print('introducing embedding matrix and dictionary')
        embedding_w, embedding_dic = self.onehot_dic_build()

        doc_image = []
        label_image = []
        print('processing documents')
        for i in range(doc_count):
            doc_vec = self.doc_process(docs[i], embedding_dic)
            doc_image.append(doc_vec)
            label_class = np.zeros(self.num_classes, dtype='float32')
            label_class[int(label[i]) - 1] = 1
            label_image.append(label_class)

        del embedding_w, embedding_dic
        self.doc_image = np.asarray(doc_image, dtype='int64')
        self.label_image = np.array(label_image, dtype='float32')
        # print(self.doc_image.shape)
        # print(self.label_image.shape)
        # print(self.doc_image[:5])
        # print(self.label_image[:5])

    def doc_process(self, doc, embedding_dic):
        min_len = min(self.l0, len(doc))
        doc_vec = np.zeros(self.l0, dtype='int64')
        for j in range(min_len):
            if doc[j] in embedding_dic:
                doc_vec[j] = int(embedding_dic[doc[j]])
            else:
                doc_vec[j] = int(embedding_dic['UNK'])

        return doc_vec

    def onehot_dic_build(self):

        alphabet = self.alphabet
        embedding_dic = {}
        embedding_w = []

        embedding_dic['UNK'] = 0
        embedding_w.append(np.zeros(len(alphabet), dtype='float32'))

        for i, alpha in enumerate(alphabet):
            onehot = np.zeros(len(alphabet), dtype='float32')
            embedding_dic[alpha] = i + 1
            onehot[i] = 1
            embedding_w.append(onehot)

        embedding_w = np.array(embedding_w, dtype='float32')

        return embedding_w, embedding_dic

if __name__ == '__main__':
    data = Dataset('data/ag_news_csv/train.csv')
    data.dataset_read()

