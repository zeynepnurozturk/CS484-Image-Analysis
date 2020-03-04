import testNetwork
import trainNetwork
import evaluation
import os.path

print("Begin")

# paths
trainDataPath = "./data/train/"
testDataPath = "./data/test/"
trainMatPath = "train.mat"
networkPath = 'network.pth'
testMatPath = "test.mat"
resultPath = "./result"

# network parameters
epochSize = 800
learningRate = 0.0005
hiddenSize = 256
batchSize = 32

# if there is no trained network, train network
if not os.path.isfile(networkPath):
    print("\nTrainin network\n")
    trainNetwork.train_network(trainDataPath, trainMatPath, networkPath, epochSize, learningRate, hiddenSize, batchSize)
    print("\nTrainin finished\n")

# test network
print("Testing network")
image_results = testNetwork.test_network(testDataPath, testMatPath, networkPath, resultPath, hiddenSize, batchSize)
print("Testing finished")

# evaluate results
print("Evaluating results")
evaluation.eval_results(image_results, resultPath)

print("Done")
