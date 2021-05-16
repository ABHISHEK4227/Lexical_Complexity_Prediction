import csv
import pandas as pd

class Dataset(object):

    def __init__(self, type = 'single'):
        self.type = type

        train_path = "datasets/train/lcp_{}_train.tsv".format(self.type)
        trial_path = "datasets/trial/lcp_{}_trial.tsv".format(self.type)
        test_path = "datasets/test/lcp_{}_test.tsv".format(self.type)

        self.df_train,self.df_train_id,self.df_y_train = self.readfile(train_path)
        self.df_test,self.df_test_id,self.df_y_test = self.readfile(test_path)
        self.df_trial,self.df_trial_id,self.df_y_trial = self.readfile(trial_path)

    def readfile(self, file_path):
        df = pd.read_csv(file_path,encoding='utf-8', delimiter='\t', quotechar='\t', keep_default_na=False)
        df_id = df['id']
        df = df.drop(['id'],axis=1)
        df_y = df['complexity']
        df = df.drop(['complexity'],axis=1)
        return df,df_id,df_y
        
        
if __name__ == '__main__':
    ds = Dataset()
    dm = Dataset('multi')