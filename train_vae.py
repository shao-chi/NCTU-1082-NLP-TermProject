import pandas as pd
import numpy as np
import torch
import ujson

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline

from utils.text2image_model import VAE, MultilabelClassification

NUM_EPOCHES = 200
BATCH_SIZE = 64

HIDDEN_SIZE = 512
LATENT_SIZE = 32

NUM_CLASSES = 43
LEARNING_RATE = 0.00001
N_COMPONENTS = 100


def load_csv_data(path, state=None):
    data = pd.read_json(path, orient='records')

    index = data['idx'].values
    X = data['preprocessed_text'].values

    if state == 'train':
        categories_index = data['categories_index'].values
        categories_index = np.array([i for i in categories_index])

        categories_index = torch.tensor(categories_index).float()

        image = data['gif_image'].values
        image = np.array([np.array(i).reshape((-1)) for i in image])

        image = torch.tensor(image / 255).float()

        print('Loaded ', path)

        return index, X, categories_index, image

    else:
        print('Loaded ', path)

        return index, X


def train(data_x, data_y, image, dir_, test_idx, test_x, dev_idx, dev_x, categories):
    print('Training model ......')

    model_vae = VAE(text_size=N_COMPONENTS,
            image_size=80*80*3,
            hidden_size=HIDDEN_SIZE,
            latent_size=LATENT_SIZE,
            learning_rate=LEARNING_RATE)
    model_class = MultilabelClassification(n_components=N_COMPONENTS,
                                          hidden_size=HIDDEN_SIZE,
                                          num_classes=NUM_CLASSES,
                                          learning_rate=LEARNING_RATE)
    model_vae.train()
    model_class.train()

    split = int(len(data_x) * 0.8)
    train_x = data_x[:split]
    train_y = data_y[:split]
    train_image = image[:split]

    valid_x = data_x[split:]
    valid_y = data_y[split:]
    valid_image = image[split:]

    metrics = {'Train': {'Recall@6': [], 
                         'Loss': [],
                         'VAE_Loss': []},
                'Valid': {'Recall@6': [], 
                          'Loss': [],
                          'VAE_Loss': []}}

    for epoch in range(1, NUM_EPOCHES+1):
        print('Epoch: ', epoch)

        iteration = len(train_x) // BATCH_SIZE
        for iter_ in range(iteration):
            batch_x = train_x[iter_*BATCH_SIZE:(iter_+1)*BATCH_SIZE]
            batch_y = train_y[iter_*BATCH_SIZE:(iter_+1)*BATCH_SIZE]
            batch_image = train_image[iter_*BATCH_SIZE:(iter_+1)*BATCH_SIZE]

            model_vae.train_model(batch_x=batch_x, batch_image=batch_image)

            recontruction = model_vae.predict(x_data=batch_x)
            model_class.train_model(batch_x=batch_x,
                                    batch_image=recontruction,
                                    batch_y=batch_y)

            if (iter_+1) % 100 == 0:
                recontruction = model_vae.predict(x_data=train_x)
                train_loss = model_class.compute_loss(x_data=train_x,
                                                      image=recontruction,
                                                      y_target=train_y)
                train_recall = model_class.evaluate(x_data=train_x,
                                                    image=recontruction,
                                                    y_target=train_y)
                train_vae_loss = model_vae.compute_loss(x_data=train_x,
                                                        image_target=train_image)
                metrics['Train']['Loss'].append(train_loss)
                metrics['Train']['Recall@6'].append(train_recall)
                metrics['Train']['VAE_Loss'].append(train_vae_loss)

                recontruction = model_vae.predict(x_data=valid_x)
                valid_loss = model_class.compute_loss(x_data=valid_x,
                                                      image=recontruction,
                                                      y_target=valid_y)
                valid_recall = model_class.evaluate(x_data=valid_x,
                                                    image=recontruction,
                                                    y_target=valid_y)
                valid_vae_loss = model_vae.compute_loss(x_data=valid_x,
                                                        image_target=valid_image)
                metrics['Valid']['Loss'].append(valid_loss)
                metrics['Valid']['Recall@6'].append(valid_recall)
                metrics['Valid']['VAE_Loss'].append(valid_vae_loss)

                print(f'Iter {iter_+1}, TRAIN VAE Loss: {round(train_vae_loss)}, VALID VAE Loss: {round(valid_vae_loss)}')
                print(f'--------------- TRAIN Loss: {round(train_loss, 4)}, Recall@6: {round(train_recall, 4)}, VALID Loss: {round(valid_loss, 4)}, Recall@6: {round(valid_recall, 4)}')

                if valid_recall > 0.3:
                    model_vae.save(path=f'{dir_}/VAE_model_{int(valid_recall*10000)}.pkl')
                    model_class.save(path=f'{dir_}/model_{int(valid_recall*10000)}.pkl')

                    recontruction = model_vae.predict(x_data=test_x)
                    test_predictions = model_class.predict(x_data=recontruction, categories=categories)
                    test_predict = pd.DataFrame(data={'idx': test_idx, 'categories': test_predictions})
                    test_predict = test_predict \
                        .merge(pd.read_json('./public_phases_1_2/test_unlabeled.json', lines=True), \
                                on='idx')
                    test_predict = test_predict.to_dict(orient='records')

                    with open(f'{dir_}/eval_{int(valid_recall*10000)}.json', 'w') as f:
                        for item in test_predict:
                            ujson.dump(item, f)
                            f.write('\n')
                        print(f'Saved eval_{int(valid_recall*10000)}.json')

                    recontruction = model_vae.predict(x_data=test_x)
                    dev_predictions = model_class.predict(x_data=recontruction, categories=categories)
                    dev_predict = pd.DataFrame(data={'idx': dev_idx, 'categories': dev_predictions})
                    dev_predict = dev_predict \
                        .merge(pd.read_json('./public_phases_1_2/dev_unlabeled.json', lines=True), \
                                on='idx')
                    dev_predict = dev_predict.to_dict(orient='records')

                    with open(f'{dir_}/dev_{int(valid_recall*10000)}.json', 'w') as f:
                        for item in dev_predict:
                            ujson.dump(item, f)
                            f.write('\n')
                        print(f'Saved dev_{int(valid_recall*10000)}.json')

    return metrics


if __name__ == '__main__':
    dir_ = f'./output_2/exp_VAE_{NUM_EPOCHES}E_{HIDDEN_SIZE}H_{N_COMPONENTS}F'
    print(dir_)

    train_path = './data/train_2.json'
    test_path = './data/test_2.json'
    dev_path = './data/dev_2.json'

    _, data_x, data_y, data_image = load_csv_data(path=train_path, state='train')
    test_idx, test_x = load_csv_data(path=test_path)
    dev_idx, dev_x = load_csv_data(path=dev_path)

    vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 3))
    data_x = vectorizer.fit_transform(data_x)
    test_x = vectorizer.fit_transform(test_x)
    dev_x = vectorizer.fit_transform(dev_x)
    print('Converted word to vector')

    decom = TruncatedSVD(n_components=N_COMPONENTS, algorithm='arpack', random_state=1, n_iter=5)
    data_x = decom.fit_transform(data_x)
    test_x = decom.fit_transform(test_x)
    dev_x = decom.fit_transform(dev_x)
    print('Truncated SVD')

    data_x = torch.tensor(data_x).float()
    test_x = torch.tensor(test_x).float()
    dev_x = torch.tensor(dev_x).float()

    categories = pd.read_json('./public_phases_1_2/categories.json').values.reshape(-1)

    metrics = train(data_x=data_x,
                    data_y=data_y, 
                    image=data_image,
                    dir_=dir_,
                    test_idx=test_idx,
                    test_x=test_x,
                    dev_idx=dev_idx,
                    dev_x=dev_x,
                    categories=categories)

    with open(f'{dir_}/metrics.json', 'w') as f:
        ujson.dump(metrics, f)
        print('Saved metrics.json')