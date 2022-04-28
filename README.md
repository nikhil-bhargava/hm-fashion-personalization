# H&M Personalized Fashion Recommendations

## Introduction

H&M Group has over 53 online marketplaces and approximately 5,000 stores. With thousands of options of  clothes, it’s difficult for customers to find what they like. By personalizing the shopping experience, they believe they can solve this issue. Therefore, the goal of this project is to create a model that will provide product recommendations given H&M transaction, article, and customer data

## Data

The raw data for this project can be downloaded [here](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations).

![hm-fashion 003](https://user-images.githubusercontent.com/31523376/165660262-8e4965e2-deaf-41a3-80d5-18977fecd2fb.jpeg)

## Models

Two main approaches were used for this project, a User-User Collaborative filtering and a Neural Network Hybrid Recommender. A better understanding of the two models can be seen below.

![hm-fashion 006](https://user-images.githubusercontent.com/31523376/165660373-d6769691-e6fd-45ba-a81d-7f2c602eddc5.jpeg)

![hm-fashion 007](https://user-images.githubusercontent.com/31523376/165660380-c39f5a1d-8b66-4aba-99c3-04b1f2296a70.jpeg)

To bypass model training, download the pickle files located [here](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations) and move them to the `models/` folder.

## Getting Started

To re-train the models, and make predictions:
  1. Download the data [here](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations). Move data to the `data/00_raw` folder.
  2. To train the collaborative filtering model, and make predictions, run the script: `python cf_model.py`.
  3. To train the Neural Network Hybrid Recommender, and make predictions, run the script: `python nn_model.py`.

## Demo

Given a new customer joins H&M, and has never purchased anything, how do we provide product recommendations? This app created is intended to solve that issue by asking new users to input preferences at the start. Recommendations will then be provided using the collaborative filtering algorithm.

To run the app locally,
  1. Download the data [here](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations). Move data to the `data/00_raw` folder.
  2. Download the data [here](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations). Move data to the `data/01_processed` folder.
  3. Use the following command to run the app: `streamlit run app.py` or `streamlit run app_local.py`

The app created on streamlit and deployed on GCP App Engine, can be found [here](https://hm-fashion-rec.uc.r.appspot.com/).
