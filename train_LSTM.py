import pandas as pd
import numpy as np
import torch
import ujson

from utils.model import LSTM

NUM_EPOCHES = 100
BATCH_SIZE = 512

HIDDEN_SIZE = 256
NUM_LAYERS = 2
NUM_CLASSES = 43
LEARNING_RATE = 0.0001
N_COMPONENTS = 10


def load_csv_data(path, state=None):
    data = pd.read_json(path, orient='records')

    index = data['idx'].values
    # X = data['tokens'].values
    X = data['word_index'].values
    X = np.array([np.array(i).astype(int) for i in X])

    print('Loaded ', path)

    if state == 'train':
        categories_index = data['categories_index'].values
        categories_index = np.array([i for i in categories_index])

        Y = torch.tensor(categories_index).float()
        return index, X, Y

    else:
        return index, X


def train(data_x, 
          data_x_origin,
          data_y_origin,
          data_y,
          dir_,
          test_x,
          test_idx,
          dev_x,
          dev_idx,
          categories):
    print('Training model ......')

    model = LSTM(hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                num_classes=NUM_CLASSES,
                learning_rate=LEARNING_RATE,
                n_components=N_COMPONENTS)
    model.train()

    split = int(len(data_x) * 0.9)
    split_o = int(len(data_x_origin) * 0.9)
    train_x = data_x[:split]
    train_y = data_y[:split]
    train_x_origin = data_x_origin[:split_o]
    train_y_origin = data_y_origin[:split_o]
    valid_x = data_x[split:]
    valid_y = data_y[split:]
    valid_x_origin = data_x_origin[split_o:]
    valid_y_origin = data_y_origin[split_o:]

    metrics = {'Train': {'Recall@6': [], 
                         'Loss': []},
                'Valid': {'Recall@6': [], 
                          'Loss': []}}

    for epoch in range(1, NUM_EPOCHES+1):
        print('Epoch: ', epoch)

        iteration = len(train_x) // BATCH_SIZE
        for iter_ in range(iteration):
            batch_x = train_x[iter_*BATCH_SIZE:(iter_+1)*BATCH_SIZE]
            batch_y = train_y[iter_*BATCH_SIZE:(iter_+1)*BATCH_SIZE]

            model.train_model(batch_x=batch_x, batch_y=batch_y)

            if (iter_+1) % 5 == 0:
                train_loss = model.compute_loss(x_data=train_x,
                                                y_target=train_y)
                train_recall = model.evaluate(x_data=train_x_origin,
                                              y_target=train_y_origin,
                                              categories=categories)
                metrics['Train']['Loss'].append(train_loss)
                metrics['Train']['Recall@6'].append(train_recall)
                                              
                valid_loss = model.compute_loss(x_data=valid_x,
                                                y_target=valid_y)
                valid_recall = model.evaluate(x_data=valid_x_origin,
                                              y_target=valid_y_origin,
                                              categories=categories)
                metrics['Valid']['Loss'].append(valid_loss)
                metrics['Valid']['Recall@6'].append(valid_recall)

                print(f'Iter {iter_+1}, TRAIN Loss: {round(train_loss, 4)}, Recall@6: {round(train_recall, 4)}, VALID Loss: {round(valid_loss, 4)}, Recall@6: {round(valid_recall, 4)}')

                if train_recall > 0.4:
                    test_predictions = model.predict(x_data=test_x, categories=categories)
                    test_predict = pd.DataFrame(data={'idx': test_idx, 'categories': test_predictions})
                    test_predict = test_predict \
                        .merge(pd.read_json('./public_phases_1_2/test_unlabeled.json', lines=True), \
                                on='idx')
                    test_predict = test_predict.to_dict(orient='records')

                    with open(f'{dir_}/eval_{int(train_recall*1000)}.json', 'w') as f:
                        for item in test_predict:
                            item['categories'] = list(item['categories'])
                            ujson.dump(item, f)
                            f.write('\n')
                        print('Saved eval.json')

                    # dev_predictions = model.predict(x_data=dev_x, categories=categories)
                    # dev_predict = pd.DataFrame(data={'idx': dev_idx, 'categories': dev_predictions})
                    # dev_predict = dev_predict \
                    #     .merge(pd.read_json('./public_phases_1_2/dev_unlabeled.json', lines=True), \
                    #             on='idx')
                    # dev_predict = dev_predict.to_dict(orient='records')

                    # with open(f'{dir_}/dev_{int(train_recall*1000)}.json', 'w') as f:
                    #     for item in dev_predict:
                    #         item['categories'] = list(item['categories'])
                    #         ujson.dump(item, f)
                    #         f.write('\n')
                    #     print('Saved dev.json')


    return model, metrics


if __name__ == '__main__':
    dir_ = f'./output_2/exp_LSTM_{NUM_LAYERS}L_{NUM_EPOCHES}E_{HIDDEN_SIZE}H_{N_COMPONENTS}F'
    print(dir_)

    train_path = './data/train_3.json'
    test_path = './data/test_3.json'
    dev_path = './data/dev_3.json'

    _, data_x, data_y = load_csv_data(path=train_path, state='train')
    test_idx, test_x = load_csv_data(path=test_path)
    dev_idx, dev_x = load_csv_data(path=dev_path)

    # model_word2vec = Word2Vec(
    #                     np.concatenate((data_x, test_x, dev_x), axis=0), \
    #                     size=N_COMPONENTS, \
    #                     min_count=1, \
    #                     negative=10)
    # model_word2vec.wv['<PAD>'] = np.zeros((N_COMPONENTS), dtype=float)
    # model_word2vec.save(f'{dir_}/word2vec.model')

    # data_x_vec = list()
    # for x in data_x:
    #     data_x_vec.append(model_word2vec.wv[x])
    # data_x_vec = torch.tensor(data_x_vec).float()

    # print('Converted word to vector')

    # train_data_x = list()
    # train_data_y = list()
    # for x, y in zip(data_x, data_y):
    #     for i in np.where(y == 1)[0]:
    #         train_data_x.append(x)
    #         train_data_y.append(i)

    # train_data_x = torch.LongTensor(train_data_x)
    # train_data_y = torch.LongTensor(train_data_y)
    # print(train_data_x.size(), train_data_y.size())

    data_x = torch.LongTensor(data_x)
    data_y = torch.FloatTensor(data_y)
    test_x = torch.LongTensor(test_x)
    dev_x = torch.LongTensor(dev_x)

    categories = pd.read_json('./public_phases_1_2/categories.json').values.reshape(-1)
    model, metrics = train(data_x=data_x,
                           data_y=data_y,
                           data_x_origin=data_x,
                           data_y_origin=data_y,
                           dir_=dir_,
                           test_x=test_x,
                           test_idx=test_idx,
                           dev_x=dev_x,
                           dev_idx=dev_idx,
                           categories=categories)

    # with open(f'{dir_}/metrics.json', 'w') as f:
    #     ujson.dump(metrics, f)
    #     print('Saved metrics.json')

    # test_x_vec = list()
    # for x in test_x:
    #     test_x_vec.append(model_word2vec.wv[x])
    # test_x_vec = torch.tensor(test_x_vec).float()

    # dev_x_vec = list()
    # for x in dev_x:
    #     dev_x_vec.append(model_word2vec.wv[x])
    # dev_x_vec = torch.tensor(dev_x_vec).float()