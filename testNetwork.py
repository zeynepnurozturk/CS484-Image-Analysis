import os
import numpy as np
import cv2
import torch
from tqdm import tqdm
import network
import selective_search
import scipy.io as sio
import preprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# dataset
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, images, label_box):
        self.images = images
        self.label_box = label_box

    def __getitem__(self, index):
        x = torch.from_numpy(self.images[:, index])
        y = self.label_box[index]
        return x, y

    def __len__(self):
        return self.images.shape[1]


# read test data
def create_image_data(dataPath):

    keySet = ['n01615121', 'n02099601', 'n02123159', 'n02129604', 'n02317335', 'n02391049', 'n02410509',
              'n02422699', 'n02481823', 'n02504458']
    valueSet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    categoryDict = dict(zip(keySet, valueSet))

    bounding_box = []

    for line in open(os.path.join(dataPath, 'bounding_box.txt'), 'r'):
        line = line.rstrip("\n\r")
        elements = line.split(', ')

        bounding_box.append(
            (categoryDict[elements[0]], int(elements[1]), int(elements[2]), int(elements[3]), int(elements[4]))
        )

    image_data = []
    path = os.path.join(dataPath, 'images')

    for img in tqdm(os.listdir(path)):
        try:
            index = int(os.path.splitext(img)[0])
            label = bounding_box[index][0]
            img_array = cv2.imread(os.path.join(path, img), 1)
            image_data.append([img_array, label, bounding_box[index][1:]])
        except:
            print("N'oluyor?")

    return image_data


# preprocess test data
def process_test_data(dataPath):
    image_data = create_image_data(dataPath)

    processed_images = []

    for img_t in image_data:
        img = img_t[0]
        label = img_t[1]
        box_t = img_t[2]

        boxes = selective_search.selective_search(img, mode='fast', random=False)
        boxes = selective_search.box_filter(boxes, min_size=20, topN=400)

        postBox = []
        boxIDs = []
        boxID = 0
        for box in boxes:

            new_img = img[box[0]:box[3], box[1]:box[2]]

            postBox.append(preprocess.preprocess(new_img))
            boxIDs.append(boxID)
            boxID += 1

        postBox = np.squeeze(postBox).transpose()

        processed_images.append(((img, label, box_t, boxes), postBox, boxIDs))

        print("%d/%d processed" % (len(processed_images), len(image_data)))


    return processed_images


def test_network(dataPath, matPath, networkPath, resultpath, hiddenSize, batchSize):

    try:
        m_data = sio.loadmat(matPath)
        processed_images = m_data['processed_images']
        processed_images = processed_images.tolist()

        tmp = []
        for x in processed_images:

            x[0] = x[0].tolist()[0]
            x[0][1] = x[0][1][0][0]
            x[0][2] = tuple(x[0][2][0])

            x[2] = np.squeeze(x[2])

            tmp.append((x[0], x[1], x[2]))

        processed_images = tmp

    except:
        print("\nPreprocessing test data\n")
        processed_images = process_test_data(dataPath)
        sio.savemat(matPath, {'processed_images': processed_images})

    # save candidate window examples
    for i in range(10):
        im = processed_images[i*10][0][0]

        fig = plt.figure()
        ax = fig.add_subplot(111)


        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

        boxes = processed_images[i*10][0][3]

        for box in boxes:
            # Create a Rectangle patch

            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor=np.random.rand(3,), facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)

        plt.savefig('%s/boxes_%d.png' % (resultpath, i))
        plt.cla()
        plt.clf()
        plt.close(fig)


    net = network.Feedforward(hiddenSize)
    net.load_state_dict(torch.load(networkPath))
    net.eval()

    img_results = []

    # test images on network
    print("Testing each image")
    for img in processed_images:
        testset = TestDataset(img[1], img[2])
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=0)  # TODO bach size

        results = []
        for i, data in enumerate(testloader, 0):

            inputs, ids = data

            output = net(inputs)
            values, labels = torch.max(output, 1)


            for j in range(len(values)):
                results.append((values.detach().numpy()[j], labels.detach().numpy()[j], ids.detach().numpy()[j]))

        results.sort(key=lambda el: el[0])

        img_results.append((img[0][0], (img[0][1], img[0][2]), (results[-1][1], img[0][3][results[-1][2]])))

    keySet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    valueSet = ['n01615121', 'n02099601', 'n02123159', 'n02129604', 'n02317335', 'n02391049', 'n02410509',
                'n02422699', 'n02481823', 'n02504458']
    categoryDict = dict(zip(keySet, valueSet))

    # save localization result examples
    s = 0
    for i in range(10):

        for j in [1, 3]:

            result = img_results[i*10 + j]
            im = result[0]

            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

            box = result[1][1]
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            box = result[2][1]
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

            plt.annotate('Given window(Red): ' + categoryDict[result[1][0]], (0, 0), (0, -15), xycoords='axes fraction', textcoords='offset points', va='top', size=10, color='r')
            plt.annotate('Predicted window(Blue): ' + categoryDict[result[2][0]], (0, 0), (0, -30), xycoords='axes fraction', textcoords='offset points', va='top', size=10, color='b')

            plt.savefig('%s/local_%d.png' % (resultpath, s))
            plt.cla()
            plt.clf()
            plt.close(fig)
            s += 1

    return img_results
