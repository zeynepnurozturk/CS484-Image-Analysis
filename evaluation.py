import numpy as np
from tabulate import tabulate
from mlxtend.evaluate import confusion_matrix

keySet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
valueSet = ['n01615121', 'n02099601', 'n02123159', 'n02129604', 'n02317335', 'n02391049', 'n02410509',
          'n02422699', 'n02481823', 'n02504458']
categoryDict = dict(zip(keySet, valueSet))

result_file = None

def multi_class_confision(imageResults):
    y_target = []
    y_predicted = []
    for result in imageResults:
        y_target.append(result[1][0])
        y_predicted.append(result[2][0])
    cm = confusion_matrix(y_target=y_target,
                          y_predicted=y_predicted,
                          binary=False)

    table = []
    for i in range(10):
        table.append([categoryDict[i]] + cm[i].tolist())

    result_file.write("Confusion Matrix:\n\n")
    result_file.write(tabulate(table, headers=['', categoryDict[0], categoryDict[1], categoryDict[2], categoryDict[3], categoryDict[4], categoryDict[5], categoryDict[6], categoryDict[7], categoryDict[8], categoryDict[9]], tablefmt='orgtbl'))
    result_file.write("\n")

    recall = np.diag(cm) / np.sum(cm, axis=1)
    precision = np.diag(cm) / np.sum(cm, axis=0)
    # overall recall and precision
    overall_recall = np.mean(recall)
    overall_precision = np.mean(precision)
    accuracy = np.diag(cm) / 10
    accuracy = np.mean(accuracy)
    return recall, precision, overall_recall, overall_precision, accuracy


def calculata_area_accuracy(imageResult):
    g = imageResult[1][1]
    b = imageResult[2][1]

    xA = max(g[0], b[0])
    yA = max(g[1], b[1])
    xB = min(g[2], b[2])
    yB = min(g[3], b[3])

    intersection = abs(max(xB - xA, 0)) * abs(max(yB - yA, 0))

    areaG = (g[2] - g[0]) * (g[3] - g[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])

    union = areaB + areaG - intersection

    return intersection / union


def eval_results(imageResults, resultpath):
    global result_file
    result_file = open("%s/results.txt" % resultpath, "w")

    recall, precision, overall_recall, overall_precision, overall_accuracy = multi_class_confision(imageResults)

    for i in range(10):
        result_file.write("\nCategory: [%s], recall: [%.3f], precision: [%.3f]\n" % (categoryDict[i], recall[i], precision[i]))

    result_file.write("\nOverall Recall: [%.3f], Overall Precision: [%.3f], Overall Accuracy: [%.3f]\n" % (overall_recall, overall_precision, overall_accuracy))

    correct = 0

    for result in imageResults:
        if result[1][0] == result[2][0]:
            overlap_ratio = calculata_area_accuracy(result)
            if overlap_ratio > 0.5:
                correct += 1

    correctRatio = correct / len(imageResults)

    result_file.write("\nLocalization Accuracy: Correct Ratio: [%d/%d] = [%.3f]\n" % (correct, len(imageResults), correctRatio))

    result_file.close()
