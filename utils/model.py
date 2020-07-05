import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LSTM(nn.Module):
    def __init__(self,
            hidden_size=512,
            num_layers=2,
            num_classes=43,
            learning_rate=0.0001,
            n_components=100):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.layer_norm = nn.LayerNorm([38, hidden_size])
        self.embedding = nn.Embedding(10001, n_components, padding_idx=0)

        self.lstm = nn.LSTM(n_components, hidden_size, num_layers, batch_first=True, dropout=0.5)
        # self.rnn = nn.RNN(n_components, hidden_size, num_layers, batch_first=True)
        self.linear_1 = nn.Linear(hidden_size, 2048)
        self.relu = nn.ReLU()

        self.linear_2 = nn.Linear(2048, num_classes)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()

        self.criterion = nn.BCELoss()
        # self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)


    def forward(self, input_data):
        input_data = self.embedding(input_data)

        h0 = torch.zeros(self.num_layers, input_data.size(0), self.hidden_size).float()
        c0 = torch.zeros(self.num_layers, input_data.size(0), self.hidden_size).float()

        output, _ = self.lstm(input_data, (h0, c0))
        # output = self.relu(output)
        # output, _ = self.rnn(input_data, h0)

        output = output[:, -1, :]
        output = self.relu(self.linear_1(output))
        output = self.sigmoid(self.linear_2(output))
        # output = self.softmax(self.linear_2(output))

        return output


    def train_model(self, batch_x, batch_y):
        output = self.forward(input_data=batch_x)
        loss = self.criterion(output, batch_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def compute_loss(self, x_data, y_target):
        with torch.no_grad():
            output = self.forward(input_data=x_data)

            return F.binary_cross_entropy(output, y_target).item()
            # return F.cross_entropy(output, y_target).item()


    def evaluate(self, x_data, y_target, categories):
        with torch.no_grad():
            output = self.forward(x_data)
            # output = torch.argsort(output, dim=1, descending=True)

            # predictions = [np.where(out < 6)[0] for out in output]
            predictions = list()
            for out in output:
                assert len(out) == len(categories)

                predictions.append(sorted(range(len(out)), key=lambda i: out[i], reverse=True)[:6])

            target = [np.where(y == 1)[0] for y in y_target]
            intersections = [set(target[i]).intersection(set(predictions[i])) \
                                for i in range(len(target))]

            recall = [len(intersections[i]) / len(target[i]) \
                        for i in range(len(target))]

            return np.mean(recall)


    def predict(self, x_data, categories):
        with torch.no_grad():
            output = self.forward(x_data)

            predictions = list()
            for out in output:
                assert len(out) == len(categories)

                cate = categories[sorted(range(len(out)), key=lambda i: out[i], reverse=True)[:6]]
                predictions.append(cate)

            # output = torch.argsort(output, dim=1, descending=True)
            # predictions = [categories[np.where(out < 6)[0]] for out in output]

        return predictions


    def save(self, path):
        torch.save(self.state_dict(), path)


    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        self.eval()


class BI_LSTM(nn.Module):
    def __init__(self,
            hidden_size=512,
            num_layers=2,
            num_classes=43,
            learning_rate=0.0001,
            n_components=100):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.layer_norm = nn.LayerNorm([38, hidden_size])
        self.embedding = nn.Embedding(10001, n_components, padding_idx=0)

        self.lstm = nn.LSTM(n_components, hidden_size, num_layers, batch_first=True, dropout=0.5, bidirectional=True)
        # self.rnn = nn.RNN(n_components, hidden_size, num_layers, batch_first=True)
        # self.linear_1 = nn.Linear(hidden_size, hidden_size)
        # self.relu = nn.ReLU()

        self.linear_2 = nn.Linear(hidden_size*2, num_classes)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()

        # self.criterion = nn.BCELoss()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)


    def forward(self, input_data):
        input_data = self.embedding(input_data)

        h0 = torch.zeros(self.num_layers*2, input_data.size(0), self.hidden_size).float()
        c0 = torch.zeros(self.num_layers*2, input_data.size(0), self.hidden_size).float()

        output, _ = self.lstm(input_data, (h0, c0))
        # output = self.relu(output)
        # output, _ = self.rnn(input_data, h0)

        output = output[:, -1, :]
        # output = self.relu(self.linear_1(output))
        output = self.sigmoid(self.linear_2(output))
        # output = self.softmax(self.linear_2(output))

        return output


    def train_model(self, batch_x, batch_y):
        output = self.forward(input_data=batch_x)
        loss = self.criterion(output, batch_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def compute_loss(self, x_data, y_target):
        with torch.no_grad():
            output = self.forward(input_data=x_data)

            # return F.binary_cross_entropy(output, y_target).item()
            return F.cross_entropy(output, y_target).item()


    def evaluate(self, x_data, y_target, categories):
        with torch.no_grad():
            output = self.forward(x_data)
            # output = torch.argsort(output, dim=1, descending=True)

            # predictions = [np.where(out < 6)[0] for out in output]
            predictions = list()
            for out in output:
                assert len(out) == len(categories)

                predictions.append(sorted(range(len(out)), key=lambda i: out[i], reverse=True)[:6])

            target = [np.where(y == 1)[0] for y in y_target]
            intersections = [set(target[i]).intersection(set(predictions[i])) \
                                for i in range(len(target))]

            recall = [len(intersections[i]) / len(target[i]) \
                        for i in range(len(target))]

            return np.mean(recall)


    def predict(self, x_data, categories):
        with torch.no_grad():
            output = self.forward(x_data)

            predictions = list()
            for out in output:
                assert len(out) == len(categories)

                cate = categories[sorted(range(len(out)), key=lambda i: out[i], reverse=True)[:6]]
                predictions.append(cate)

            # output = torch.argsort(output, dim=1, descending=True)
            # predictions = [categories[np.where(out < 6)[0]] for out in output]

        return predictions


    def save(self, path):
        torch.save(self.state_dict(), path)


    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        self.eval()


class LINEAR(nn.Module):
    def __init__(self,
            hidden_size=512,
            num_classes=43,
            learning_rate=0.0001,
            n_components=100):
        super(LINEAR, self).__init__()

        # self.embedding = nn.Embedding(1001, n_components, padding_idx=0)

        self.linear_1 = nn.Linear(n_components, hidden_size)
        self.dropout = nn.Dropout(0.4)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()

        self.linear_2 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

        self.criterion = nn.BCELoss()
        # self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)


    def forward(self, input_data):
        # input_data = self.embedding(input_data)
        # input_data = input_data.view((input_data.size(0), -1))

        output = self.relu(self.dropout(self.batch_norm(self.linear_1(input_data))))
        output = self.sigmoid(self.linear_2(output))

        return output


    def train_model(self, batch_x, batch_y):
        output = self.forward(input_data=batch_x)
        loss = self.criterion(output, batch_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def compute_loss(self, x_data, y_target):
        with torch.no_grad():
            output = self.forward(input_data=x_data)

            return F.binary_cross_entropy(output, y_target).item()
            # return F.cross_entropy(output, y_target).item()


    def evaluate(self, x_data, y_target, categories):
        with torch.no_grad():
            output = self.forward(x_data)
            # output = torch.argsort(output, dim=1, descending=True)

            # predictions = [np.where(out < 6)[0] for out in output]
            predictions = list()
            for out in output:
                assert len(out) == len(categories)

                predictions.append(sorted(range(len(out)), key=lambda i: out[i], reverse=True)[:6])

            target = [np.where(y == 1)[0] for y in y_target]
            intersections = [set(target[i]).intersection(set(predictions[i])) \
                                for i in range(len(target))]

            recall = [len(intersections[i]) / len(target[i]) \
                        for i in range(len(target))]

            return np.mean(recall)


    def predict(self, x_data, categories):
        with torch.no_grad():
            output = self.forward(x_data)

            predictions = list()
            for out in output:
                assert len(out) == len(categories)

                cate = categories[sorted(range(len(out)), key=lambda i: out[i], reverse=True)[:6]]
                predictions.append(cate)

            # output = torch.argsort(output, dim=1, descending=True)
            # predictions = [categories[np.where(out < 6)[0]] for out in output]

        return predictions


    def save(self, path):
        torch.save(self.state_dict(), path)


    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        self.eval()


class LINEAR_2L(nn.Module):
    def __init__(self,
            hidden_size=512,
            num_classes=43,
            learning_rate=0.0001,
            n_components=100):
        super(LINEAR_2L, self).__init__()

        self.linear_1 = nn.Linear(n_components, hidden_size)
        self.dropout_1 = nn.Dropout(0.4)
        self.batch_norm_1 = nn.BatchNorm1d(num_features=hidden_size)
        self.relu = nn.ReLU()

        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.dropout_2 = nn.Dropout(0.4)
        self.batch_norm_2 = nn.BatchNorm1d(num_features=hidden_size)
        self.relu_2 = nn.ReLU()

        self.linear_3 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)


    def forward(self, input_data):
        output = self.relu(self.dropout_1(self.batch_norm_1(self.linear_1(input_data))))
        output = self.relu(self.dropout_2(self.batch_norm_2(self.linear_2(output))))
        output = self.sigmoid(self.linear_3(output))

        return output


    def train_model(self, batch_x, batch_y):
        output = self.forward(input_data=batch_x)
        loss = self.criterion(output, batch_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def compute_loss(self, x_data, y_target):
        with torch.no_grad():
            output = self.forward(input_data=x_data)

            return F.binary_cross_entropy(output, y_target).item()


    def evaluate(self, x_data, y_target, categories):
        with torch.no_grad():
            output = self.forward(x_data)
            # output = torch.argsort(output, dim=1, descending=True)

            # predictions = [np.where(out < 6)[0] for out in output]
            predictions = list()
            for out in output:
                assert len(out) == len(categories)

                predictions.append(sorted(range(len(out)), key=lambda i: out[i], reverse=True)[:6])

            target = [np.where(y == 1)[0] for y in y_target]
            intersections = [set(target[i]).intersection(set(predictions[i])) \
                                for i in range(len(target))]

            recall = [len(intersections[i]) / len(target[i]) \
                        for i in range(len(target))]

            return np.mean(recall)


    def predict(self, x_data, categories):
        with torch.no_grad():
            output = self.forward(x_data)

            predictions = list()
            for out in output:
                assert len(out) == len(categories)

                cate = categories[sorted(range(len(out)), key=lambda i: out[i], reverse=True)[:6]]
                predictions.append(cate)

            # output = torch.argsort(output, dim=1, descending=True)
            # predictions = [categories[np.where(out < 6)[0]] for out in output]

        return predictions


    def save(self, path):
        torch.save(self.state_dict(), path)


    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        self.eval()