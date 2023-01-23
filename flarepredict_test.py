import argparse
from LSTM_Flare import LSTM_Flare
from sklearn.utils import class_weight
from keras.models import *
import numpy as np
import csv


from flarepredict_utils import *

def test_model(args):
    flare_label = get_flare_category(args)
    filepath = get_test_input(args)
    model_id = get_model_id(args)
    model_dir = get_model_dir(flare_label, model_id)
    models_dir = get_models_dir(flare_label)
    exists = are_model_files_exist(models_dir, model_id, flare=flare_label)

    if not exists:
        print("\nModel id", model_id,
                " does not exist for flare " + flare_label + "." + '\nPlease make sure to run training task with this id first')
        sys.exit()
    print('model_dir:', model_dir)
    model = load_model(model_dir)
    # Test
    lstm_flare = LSTM_Flare()
    n_features = get_n_features(flare_label)
    result_file = get_result_file(flare_label, model_id)
    print("Starting testing with a model with id:", model_id, 'testing data file:', filepath)
    print("Loading data set...")
    X_test_data, y_test_data = lstm_flare.load_data(datafile=filepath,
                                                    flare_label=flare_label, series_len=series_len,
                                                    start_feature=start_feature, n_features=n_features,
                                                    mask_value=mask_value)
    print("Done loading data...")
    print("Formatting and mapping the data...")
    X_test = np.array(X_test_data)
    y_test = np.array(y_test_data)
    y_test_tr = lstm_flare.data_transform(y_test)
    print("Prediction is in progress, please wait until it is done...")
    classes = model.predict(X_test, batch_size=batch_size, verbose=0, steps=None)
    print("Finished the prediction task..")
    with open(result_file, 'w', encoding='UTF-8') as result_csv:
        w = csv.writer(result_csv)
        with open(filepath, encoding='UTF-8') as data_csv:
            reader = csv.reader(data_csv)
            i = -1
            for line in reader:
                if i == -1:
                    line.insert(0, 'Predicted Label')
                else:
                    if (classes[i][0] >= 0.5 and flare_label == 'C') or (
                            classes[i][0] >= 0.75 and flare_label == 'M5') or (
                            classes[i][0] >= 0.6 and flare_label == 'M'):
                        line.insert(0, 'Positive')
                    else:
                        line.insert(0, 'Negative')
                i += 1
                w.writerow(line)
    return result_file

if __name__ == "__main__":
    flare_label = str(sys.argv[1]).strip().upper()
    if not flare_label in flares:
        print('Flare label must be one of: C, M, M5')
        sys.exit()    
    from flarepredict_test import test_model

    args = {'test_data_file': 'data/LSTM_' + flare_label + '_sample_run/normalized_testing.csv',
            'flare': flare_label,
            'modelid': 'custom_model_id'}
    custom_result = test_model(args)
