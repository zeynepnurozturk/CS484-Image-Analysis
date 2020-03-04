import torch
import torch.nn as nn
import torch.optim as optim


# dataset
class ProcessedDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        x = torch.from_numpy(self.images[:, index])
        y = torch.tensor(self.labels[index])
        return x, y

    def __len__(self):
        return self.images.shape[1]


# network model
class Feedforward(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(2048, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def train(net, trainloader, savePath, learningRate, epochSize):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learningRate, momentum=0.9)

    net.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net.to(device)

    for epoch in range(epochSize):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('[%d] loss: %.4f' % (epoch + 1, running_loss/10))

    torch.save(net.state_dict(), savePath)

