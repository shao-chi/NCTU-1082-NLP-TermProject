import pandas as pd
import numpy as np
import ujson

from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

def load_csv_data(path, state=None):
    data = pd.read_json(path, orient='records')

    index = data['idx'].values

    word_index = data['preprocessed_text'].values
    # word_index = np.array([i for i in word_index])
    X = word_index

    print('Loaded ', path)

    if state == 'train':
        categories_index = data['categories_index'].values
        categories_index = np.array([y for y in categories_index])
        Y = categories_index

        return index, X, Y

    else:
        return index, X

def train(data_x, data_y, categories, dev_x, test_x, dev_idx, test_idx, dir_):
    print('Training model ......')

    split = int(data_x.shape[0] * 0.9)
    train_x = data_x[:split]
    train_y = data_y[:split]
    valid_x = data_x[split:]
    valid_y = data_y[split:]

    for c in range(len(categories)):
        print(categories[c])

        model = SVC(gamma='scale', probability=True)
        model.fit(train_x, train_y[:, c])

        
        train_proba = model.predict_proba(train_x)[:, 1].reshape((-1, 1))
        valid_proba = model.predict_proba(valid_x)[:, 1].reshape((-1, 1))
        test_proba = model.predict_proba(test_x)[:, 1].reshape((-1, 1))
        dev_proba = model.predict_proba(dev_x)[:, 1].reshape((-1, 1))

        if c == 0:
            train_proba_list = train_proba
            valid_proba_list = valid_proba
            test_proba_list = test_proba
            dev_proba_list = dev_proba

        else:
            train_proba_list = np.concatenate((train_proba_list, train_proba), axis=1)
            valid_proba_list = np.concatenate((valid_proba_list, valid_proba), axis=1)
            test_proba_list = np.concatenate((test_proba_list, test_proba), axis=1)
            dev_proba_list = np.concatenate((dev_proba_list, dev_proba), axis=1)

    train_predict = np.argsort(train_proba_list, axis=1)
    valid_predict = np.argsort(valid_proba_list, axis=1)
    train_predict = np.array([categories[np.where(predict > 36)[0]] for predict in train_predict])
    valid_predict = np.array([categories[np.where(predict > 36)[0]] for predict in valid_predict])

    train_recall = [
        len(set(train_predict[i]).intersection(set(np.where(train_y[i] == 1)[0]))) \
                / np.where(train_y[i] == 1)[0].shape[0] \
            for i in range(train_y.shape[0])]
    valid_recall = [
        len(set(valid_predict[i]).intersection(set(np.where(valid_y[i] == 1)[0]))) \
                / np.where(valid_y[i] == 1)[0].shape[0] \
            for i in range(valid_y.shape[0])]

    print('train ', np.mean(np.array(train_recall)))
    print('valid ', np.mean(np.array(valid_recall)))

    test_predict = np.argsort(test_proba_list, axis=1)
    test_predict = np.array([categories[np.where(predict > 36)[0]] for predict in test_predict])

    dev_predict = np.argsort(dev_proba_list, axis=1)
    dev_predict = np.array([categories[np.where(predict > 36)[0]] for predict in dev_predict])

    test_predict = pd.DataFrame(data={'idx': test_idx, 'categories': list(test_predict)})
    test_predict = test_predict \
        .merge(pd.read_json('./drive/My Drive/data/test_unlabeled.json', lines=True), \
                on='idx')
    test_predict.to_json(f'{dir_}/eval.json', orient='records', lines=True)
    print('Saved eval.json')

    dev_predict = pd.DataFrame(data={'idx': dev_idx, 'categories': list(dev_predict)})
    dev_predict = dev_predict \
        .merge(pd.read_json('./drive/My Drive/data/dev_unlabeled.json', lines=True), \
                on='idx')
    dev_predict.to_json(f'{dir_}/dev.json', orient='records', lines=True)
    print('Saved dev.json')

    return model
    

if __name__ == '__main__':
    dir_ = './drive/My Drive/data/output_2/exp_SVC_10F_SS'

    train_path = './drive/My Drive/data/train_2.json'
    test_path = './drive/My Drive/data/test_2.json'
    dev_path = './drive/My Drive/data/dev_2.json'
    _, data_x, data_y = load_csv_data(path=train_path, state='train')
    test_idx, test_x = load_csv_data(path=test_path)
    dev_idx, dev_x = load_csv_data(path=dev_path)

    categories = categories = pd.read_json('./drive/My Drive/data/categories.json').values.reshape(-1)

    print('Vectorizing ... ', end='--->')
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 3))
    X = vectorizer.fit_transform(data_x)
    test_x = vectorizer.fit_transform(test_x)
    dev_x = vectorizer.fit_transform(dev_x)
    print('  Done !!!')

    print('Decomposition ... ', end='--->')
    decomposition = TruncatedSVD(n_components=10)
    X = decomposition.fit_transform(X)
    test_x = decomposition.fit_transform(test_x)
    dev_x = decomposition.fit_transform(dev_x)
    print('  Done !!!')

    print('Scaling ... ', end='--->')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    test_x = scaler.fit_transform(test_x)
    dev_x = scaler.fit_transform(dev_x)
    print('  Done !!!')

    model = train(data_x=X,
                  data_y=data_y,
                  categories=categories,
                  dev_x=dev_x,
                  test_x=test_x,
                  dev_idx=dev_idx,
                  test_idx=test_idx,
                  dir_=dir_)