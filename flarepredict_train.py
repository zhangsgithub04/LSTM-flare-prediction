import argparse
from LSTM_Flare import LSTM_Flare
from sklearn.utils import class_weight
from keras.models import *
from keras import regularizers
import numpy as np

from flarepredict_utils import *


def train_model(args):
    flare_label = get_flare_category(args)
    train_data_file = get_training_input(args)
    model_id = get_model_id(args)
    model_dir = get_model_dir(flare_label, model_id)
    n_features = get_n_features(flare_label)

    lstm_flare = LSTM_Flare()
    print("Starting training with a model with id:", model_id, 'training data file:', train_data_file)
    print('Loading data set...')
    X_train_data, y_train_data = lstm_flare.load_data(datafile=train_data_file,
                                                      flare_label=flare_label, series_len=series_len,
                                                      start_feature=start_feature, n_features=n_features,
                                                      mask_value=mask_value)
    print('Done loading data...')
    X_train = np.array(X_train_data)
    y_train = np.array(y_train_data)
    y_train_tr = lstm_flare.data_transform(y_train)

    class_weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                      classes =np.unique(y_train), y=y_train)
    class_weight_ = {0: class_weights[0], 1: class_weights[1]}
    # print(class_weight_)

    model = lstm_flare.lstm(nclass, n_features, series_len)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print('Training is in progress, please wait until it is done...')
    history = model.fit(X_train, y_train_tr,
                        epochs=epochs, batch_size=batch_size,
                        verbose=False, shuffle=True, class_weight=class_weight_)
    model.save(model_dir)
    print('\nFinished training the', flare_label,
          'flare model, you may use the flarepredict_test.py program to make prediction.')


'''
Command line parameters parser
'''

if __name__ == "__main__":
    flare_label = str(sys.argv[1]).strip().upper()
    if not flare_label in flares:
        print('Flare label must be one of: C, M, M5')
        sys.exit()
    args = {'train_data_file': 'data/LSTM_' + flare_label +'_sample_run/normalized_training.csv',
            'flare': flare_label,
            'modelid': 'custom_model_id'
            }
    train_model(args)
