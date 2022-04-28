import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from os import listdir
from PIL import Image
import random
from sklearn.metrics.pairwise import cosine_similarity

# title of streamlit app
st.title("Welcome to H&M!")

@st.cache(suppress_st_warning=True)
def load_data():
    cf = pd.read_csv('data/01_processed/app.csv')
    labels = pd.read_csv('data/01_processed/articles_app.csv')
    articles = pd.read_csv('data/00_raw/articles.csv')
    return cf, labels, articles
cf, labels, articles = load_data()
imgs = listdir('images')
imgs = [img for img in imgs if img[-4:] == '.jpg']
user_preferences = {}

disp = random.sample(imgs, 6)

image1 = Image.open('images/{}'.format(disp[0]))
image2 = Image.open('images/{}'.format(disp[1]))
image3 = Image.open('images/{}'.format(disp[2]))

image_caption1 = str(labels[labels['prod_small']==int(disp[0][1:4])]['article_id'].values[0]) + ' - ' + labels[labels['prod_small']==int(disp[0][1:4])]['prod_name'].values[0]
image_caption2 = str(labels[labels['prod_small']==int(disp[1][1:4])]['article_id'].values[0]) + ' - ' + labels[labels['prod_small']==int(disp[1][1:4])]['prod_name'].values[0]
image_caption3 = str(labels[labels['prod_small']==int(disp[2][1:4])]['article_id'].values[0]) + ' - ' + labels[labels['prod_small']==int(disp[2][1:4])]['prod_name'].values[0]

st.image([image1, image2, image3], caption=[image_caption1, image_caption2, image_caption3], width=200)

image4 = Image.open('images/{}'.format(disp[3]))
image5 = Image.open('images/{}'.format(disp[4]))
image6 = Image.open('images/{}'.format(disp[5]))

image_caption4 = str(labels[labels['prod_small']==int(disp[3][1:4])]['article_id'].values[0]) + ' - ' + labels[labels['prod_small']==int(disp[3][1:4])]['prod_name'].values[0]
image_caption5 = str(labels[labels['prod_small']==int(disp[4][1:4])]['article_id'].values[0]) + ' - ' + labels[labels['prod_small']==int(disp[4][1:4])]['prod_name'].values[0]
image_caption6 = str(labels[labels['prod_small']==int(disp[5][1:4])]['article_id'].values[0]) + ' - ' + labels[labels['prod_small']==int(disp[5][1:4])]['prod_name'].values[0]

st.image([image4, image5, image6], caption=[image_caption4, image_caption5, image_caption6], width=200)

with st.form("my_form"):

    st.write('Select your favorite items:')
    item1 = st.checkbox(image_caption1)
    item2 = st.checkbox(image_caption2)
    item3 = st.checkbox(image_caption3)
    item4 = st.checkbox(image_caption4)
    item5 = st.checkbox(image_caption5)
    item6 = st.checkbox(image_caption6)

    submitted = st.form_submit_button("Submit")
    if submitted:

        if item1:
            pref1 = str(labels[labels['prod_small']==int(disp[0][1:4])]['product_code'].values[0])
            user_preferences[pref1] = 1

        if item2:
            pref2 = str(labels[labels['prod_small']==int(disp[1][1:4])]['product_code'].values[0])
            user_preferences[pref2] = 1

        if item3:
            pref3 = str(labels[labels['prod_small']==int(disp[2][1:4])]['product_code'].values[0])
            user_preferences[pref3] = 1

        if item4:
            pref4 = str(labels[labels['prod_small']==int(disp[3][1:4])]['product_code'].values[0])
            user_preferences[pref4] = 1

        if item5:
            pref5 = str(labels[labels['prod_small']==int(disp[4][1:4])]['product_code'].values[0])
            user_preferences[pref5] = 1

        if item6:
            pref6 = str(labels[labels['prod_small']==int(disp[5][1:4])]['product_code'].values[0])
            user_preferences[pref6] = 1

        all_items = list(cf.columns)
        user_cf = {}
        for item in all_items:
            if item in user_preferences:
                user_cf[item] = 1
            else:
                user_cf[item] = 0
        user_cf_df = pd.DataFrame([user_cf])

        rankings = cosine_similarity(user_cf_df, cf, dense_output=False)
        rankings = rankings.reshape(rankings.shape[1])
        top_match_user_idx = np.argmax(rankings)
        top_match_user = cf.loc[[top_match_user_idx]].values
        top_match_user = top_match_user.reshape(top_match_user.shape[1])
        result = pd.DataFrame()
        result['product_code'] = all_items
        result['bought'] = top_match_user
        result = result[result['bought']>=1]
        result = list(result.product_code.values)
        result_dict = dict(zip(articles.product_code, articles.prod_name))

        st.write("We think you'd like...")
        for rec in result:
            st.write(result_dict[int(rec)])
            st.write()

        st.stop()