import pandas as pd
pd.options.mode.chained_assignment = None

def pre_process(json_file_path, file_name):
    df = pd.read_json(json_file_path, orient='records', lines=True)

    # remove 3 star reviews
    df_no3 = df[df['overall'].isin([1,2,4,5])]

    dict_class = {
    1 : 0,
    2 : 0,
    4 : 1,
    5 : 1
    }

    # map reviews to sentiment classification
    df_no3['sentiment'] = df_no3['overall'].map(dict_class)
    df_no3['rev_sum'] = df_no3['summary'] + ' ' + df_no3['reviewText']

    # get only relevant columns, remove duplicates, drop nulls and pickle
    return df_no3[['rev_sum', 'sentiment']].drop_duplicates(subset=['rev_sum', 'sentiment'], keep='first').dropna().to_pickle("../data/pickled_dfs/df_{}.pkl".format(file_name))