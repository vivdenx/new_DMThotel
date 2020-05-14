# Include in preprocessing script

import pandas as pd

df = pd.read_csv('./../2nd-assignment-dmt-2020/training_set_VU_DM.csv')

df['number_of_comp'] = 8 - df[['comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate', 'comp6_rate', 'comp7_rate', 'comp8_rate']].isnull().sum(axis=1)
df['comp_rate_dif'] = df[['comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate', 'comp6_rate', 'comp7_rate', 'comp8_rate']].sum(axis=1)
df['comp_availability'] = df[['comp1_inv', 'comp2_inv', 'comp3_inv', 'comp4_inv', 'comp5_inv', 'comp6_inv', 'comp7_inv', 'comp8_inv']].sum(axis=1)
df['comp_price_dif'] = df[['comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff', 'comp4_rate_percent_diff', 'comp5_rate_percent_diff', 'comp6_rate_percent_diff', 'comp7_rate_percent_diff', 'comp8_rate_percent_diff']].sum(axis=1)

column_list = ['number_of_comp', 'comp_rate_dif', 'comp_availability', 'comp_price_dif']

for column_name in column_list:
    print(f'percentage of non 0 values in {column_name}')
    print(df[df[column_name] != 0].shape[0]/df.shape[0])
