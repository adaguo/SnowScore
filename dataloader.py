import os
import random
import pandas as pd

def load(df, path_prefix, files, binary_label):
    """
    path_prefix:
    files:
    """
    for fname in files:
        label = fname.split('.')[0].split('_')[1]
        
        if binary_label:
            label = 0 if int(label) < 5 else 1

        with open(os.path.join(path_prefix, fname)) as f:
            review = f.read()
        df = df.append([[review, label]],ignore_index=True)
        
    return df

def load_imdb_dataset(binary_label, seeds = 230):
    """
    seeds: random seeds
    """
    # data
    # train 25k 50% pos, and 50% neg
    # dataset needs download from 
    # https://www.kaggle.com/iarunava/imdb-movie-reviews-dataset
    
    imdb_path = './aclimdb/'
    
    # Load the dataset
    trainSet = pd.DataFrame()
    devSet = pd.DataFrame()
    testSet = pd.DataFrame()
    
    # prepare trainning data
    # eg: 0_9.txt
    # train total: 25k 50% of total
    for cat in ['pos', 'neg']:
        dset_path = os.path.join(imdb_path, "train", cat)
        files = os.listdir(dset_path)
        trainSet = load(trainSet, dset_path, files, binary_label)       
    
    # prepare test and dev set
    # dev total: 12.5k, 25% of total
    # test total: 12.5k, 25% of total
    for cat in ['pos', 'neg']:
        dset_path = os.path.join(imdb_path, "test", cat)
        files = os.listdir(dset_path)
        files.sort()
        random.seed(seeds)
        random.shuffle(files)
        split = int(0.5 * len(files))
        files_dev = files[:split]
        files_test = files[split:]

        devSet = load(devSet, dset_path, files_dev, binary_label)
        testSet = load(testSet, dset_path, files_test, binary_label)
    
    trainSet.columns = ['review', 'sentiment']
    devSet.columns = ['review', 'sentiment']
    testSet.columns = ['review', 'sentiment']
    
    # Return the dataset
    return trainSet, devSet, testSet

def GetDataSet(binary_label = False):
    """
    only need to call this fun to get data set
    """   
    exists_train = os.path.isfile('./train_data.csv')
    exists_dev = os.path.isfile('./dev_data.csv')
    exists_test = os.path.isfile('./test_data.csv')

    exists_train_b = os.path.isfile('./train_data_b.csv')
    exists_dev_b = os.path.isfile('./dev_data_b.csv')
    exists_test_b = os.path.isfile('./test_data_b.csv')

    if exists_train and exists_dev and exists_test and not binary_label:
        trainSet = pd.read_csv('./train_data.csv', index_col=False, encoding='utf-8')
        devSet = pd.read_csv('./dev_data.csv', index_col=False, encoding='utf-8')
        testSet = pd.read_csv('./test_data.csv', index_col=False, encoding='utf-8')
    elif exists_train_b and exists_dev_b and exists_test_b and binary_label:
        trainSet = pd.read_csv('./train_data_b.csv', index_col=False, encoding='utf-8')
        devSet = pd.read_csv('./dev_data_b.csv', index_col=False, encoding='utf-8')
        testSet = pd.read_csv('./test_data_b.csv', index_col=False, encoding='utf-8')
    else:
        trainSet, devSet, testSet = load_imdb_dataset(binary_label)
        if binary_label:
            trainSet.to_csv("train_data_b.csv", index=False, encoding='utf-8')
            devSet.to_csv("dev_data_b.csv", index=False, encoding='utf-8')
            testSet.to_csv("test_data_b.csv", index=False, encoding='utf-8')
        else :
            trainSet.to_csv("train_data.csv", index=False, encoding='utf-8')
            devSet.to_csv("dev_data.csv", index=False, encoding='utf-8')
            testSet.to_csv("test_data.csv", index=False, encoding='utf-8')
    
    return trainSet, devSet, testSet


