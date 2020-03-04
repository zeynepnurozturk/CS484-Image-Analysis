import numpy as np
import os
import cv2
import torch
from tqdm import tqdm
import preprocess
import network
import scipy.io as sio


def create_training_data(dataPath):

    keySet = ['n01615121', 'n02099601', 'n02123159', 'n02129604', 'n02317335', 'n02391049', 'n02410509', 'n02422699', 'n02481823', 'n02504458']
    # valueSet = ['eagle', 'dog', 'cat', 'tiger', 'star', 'zebra', 'bison', 'antelope', 'chimpanzee', 'elephant']
    valueSet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    categoryDict = dict(zip(keySet, valueSet))

    training_data = []

    for category in keySet:  # do dogs and cats
        path = os.path.join(dataPath, category)
        label = categoryDict[category]

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path, img), 1)
                new_array = preprocess.preprocess(img_array)
                training_data.append([new_array, label])
            except Exception as e:
                print(e)
                print("N'oluyor?")

    return training_data


def train_network(dataPath, matPath, networkPath, epochSize, learningRate, hiddenSize, batchSize ):

    try:
        m_data = sio.loadmat(matPath)
        im_data = m_data['images']
        l_data = np.squeeze(m_data['labels'])
    except:
        print("\nPreprocessing train data\n")
        training_data = create_training_data(dataPath)

        imgs = []
        lbs = []

        for featureVector, label in training_data:
            imgs.append(featureVector)
            lbs.append(label)

        im_data = np.squeeze(imgs).transpose()
        l_data = np.asarray(lbs)

        sio.savemat(matPath, {'images': im_data, 'labels': l_data})

    trainset = network.ProcessedDataset(im_data, l_data)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,  shuffle=True, num_workers=0)

    net = network.Feedforward(hiddenSize)

    network.train(net, trainloader, networkPath, learningRate, epochSize)
