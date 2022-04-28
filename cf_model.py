# import packages
import pandas as pd
import numpy as np
import pickle

# load necessary data
df = pd.read_csv('data/00_raw/transactions_train.csv')
articles = pd.read_csv('data/00_raw/articles.csv')

# merge datasets
articles = articles[['article_id', 'product_code']]
df = df.merge(articles, how='left', on='article_id')

## creating collaborative filtering dataset

# ohe all features
ohe_article = pd.get_dummies(df['product_code'])

# get names of all features
article_names = list(ohe_article.columns)

# concat all ohe with customers
cf_df = pd.concat([df['customer_id'], ohe_article], axis=1)

# get number of purchases by each customer
cf_df = cf_df.groupby(['customer_id'])[article_names].sum().reset_index()

# output df (cf model)
cf_df.to_csv('models/cf_df.csv')

# split to train - test to make recommendations
from sklearn.model_selection import train_test_split

y = cf_df['customer_id']
X = cf_df.drop(columns='customer_id')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)

# get cosine similarity pairwise rankings
from sklearn.metrics.pairwise import cosine_similarity

rankings = cosine_similarity(X_test, X_train, dense_output=False)

# obtain recommndations from rankings
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

recommendations = {}
for i, pred in enumerate(rankings):
    idxs = list(np.argpartition(np.array(pred), -20)[-20:])
    recs = []
    for idx in idxs:
        cid = y_train.loc[[idx]].values[0]
        rec = list(df[df['customer_id'] == cid].groupby(['article_id'])['customer_id'].count().sort_values(ascending=False).index.values.astype('int'))
        recs.extend(rec)

    recs = list(set(recs))
    
    try:
        recs = recs[:12]
    except:
        pass
    
    cur_customer = y_test.iloc[[i]].values[0]
    recommendations[cur_customer] = recs

# output recommendations in pickle form
pickle.dump(recommendations, open("models/cf_rec.pickle", "wb"))
