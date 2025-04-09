from pandas import read_csv, concat
dataset = read_csv('/kaggle/input/stanford-ribonanza-rna-folding/train_data_QUICK_START.csv', chunksize=100000)
dataset = concat(dataset)

dataset.sequence.size

dms, a3 = dataset[dataset.experiment_type=='DMS_MaP'].reset_index(drop=True), dataset[dataset.experiment_type=='2A3_MaP'].reset_index(drop=True)

rmdb = read_csv('/kaggle/input/rmdb-rna-mapping-database-2023-data/rmdb_data.v1.3.0.csv')

a3_exp = ['1M7','NMIA','BzCN']
a3 = concat([a3, rmdb.query('experiment_type in @a3_exp and SN_filter==1')], axis=0).reset_index(drop=True)
               
dms_exp = ['BzCN_cotx', 'DMS_cotx', 'DMS_M2_seq', 'DMS']
dms = concat([dms, rmdb.query('(experiment_type in @dms_exp) and SN_filter==1')], axis=0).reset_index(drop=True)

dms.experiment_type.unique()

x_a3 = a3.sequence
y_a3 = a3.filter(regex='reactivity_[0-9]')#.values.where((a3.filter(regex='reactivity_error*')<1).values, a3.filter(regex='reactivity_[0-9]').values, np.nan )
y_a3 = y_a3.query('@y_a3.fillna(0).sum(1)!=0')
x_a3 = x_a3.iloc[y_a3.index].reset_index(drop=True)
y_a3 = y_a3.reset_index(drop=True)

x_dms = dms.sequence
y_dms = dms.filter(regex='reactivity_[0-9]')#.values(.where(dms.filter(regex='reactivity_error*')<1)
y_dms = y_dms.query('@y_dms.fillna(0).sum(1)!=0')
x_dms = x_dms.iloc[y_dms.index].reset_index(drop=True)
y_dms = y_dms.reset_index(drop=True)

#!pip install keras_nlp
