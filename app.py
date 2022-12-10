import functools
from pathlib import Path

import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.shared import JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd
import plotly.express as px
from typing import List
import re
from datetime import datetime
from sklearn import metrics, preprocessing


import numpy as np
# from surprise import Reader, Dataset, SVD
# from sklearn import metrics, preprocessing
from tensorflow.keras import models, layers, utils
from tensorflow.keras.models import load_model



def main() -> None:
    st.header("Netflix movie recommendation:")

    # Products
    dtf_products = pd.read_excel("data_movies.xlsx", sheet_name="products")

    dtf_products = dtf_products[~dtf_products["genres"].isna()]
    dtf_products["product"] = range(0,len(dtf_products))
    dtf_products["name"] = dtf_products["title"].apply(lambda x: re.sub("[\(\[].*?[\)\]]", "", x).strip())
    dtf_products["date"] = dtf_products["title"].apply(lambda x: int(x.split("(")[-1].replace(")","").strip()) 
                                                                if "(" in x else np.nan)

    ## add features
    dtf_products["date"] = dtf_products["date"].fillna(9999)
    dtf_products["old"] = dtf_products["date"].apply(lambda x: 1 if x < 2000 else 0)

    # Users
    dtf_users = pd.read_excel("data_movies.xlsx", sheet_name="users").head(2000)

    dtf_users["user"] = dtf_users["userId"].apply(lambda x: x-1)

    dtf_users["timestamp"] = dtf_users["timestamp"].apply(lambda x: datetime.fromtimestamp(x))
    dtf_users["daytime"] = dtf_users["timestamp"].apply(lambda x: 1 if 6<int(x.strftime("%H"))<20 else 0)
    dtf_users["weekend"] = dtf_users["timestamp"].apply(lambda x: 1 if x.weekday() in [5,6] else 0)

    dtf_users = dtf_users.merge(dtf_products[["movieId","product","name"]], how="left")
    dtf_users = dtf_users.rename(columns={"rating":"y"})

    dtf_products_use = dtf_products[["product","name","old","genres"]].set_index("product")
    dtf_context = dtf_users[["user","product","daytime","weekend"]]

    dtf_ratings = dtf_users[["user","product","y"]]

    tmp = dtf_ratings.copy()
    dtf_up_pivot = tmp.pivot_table(index="user", columns="product", values="y")
    missing_cols = list(set(dtf_products_use.index) - set(dtf_up_pivot.columns))
    for col in missing_cols:
        dtf_up_pivot[col] = np.nan
    dtf_up_pivot = dtf_up_pivot[sorted(dtf_up_pivot.columns)]

    dtf_up_normed = pd.DataFrame(preprocessing.MinMaxScaler(feature_range=(0.5,1)).fit_transform(dtf_up_pivot.values), columns=dtf_up_pivot.columns, index=dtf_up_pivot.index)

    tags = [i.split("|") for i in dtf_products_use["genres"].unique()]
    columns = list(set([i for lst in tags for i in lst]))
    columns.remove('(no genres listed)')
    for col in columns:
        dtf_products_use[col] = dtf_products_use["genres"].apply(lambda x: 1 if col in x else 0)

    features = dtf_products_use.drop(["genres","name"], axis=1).columns
    context = dtf_context.drop(["user","product"], axis=1).columns

    # Recommend unrated movies to users:
    unrated_df = dtf_up_normed.stack(dropna=False).reset_index().rename(columns={0:"y"})

    ## add features
    unrated_df = unrated_df.merge(dtf_products_use.drop(["genres","name"], axis=1), how="left", left_on="product", right_index=True)

    # add context
    unrated_df[context] = 0 #--> simulate production for a weekday night

    # filename = 'finalized_model.sav'
    # loaded_rs = tf.saved_model.load(filename)
    loaded_model = load_model('model.h5')

    # data = loaded_rs.Y_data_n
    dtf_product_merge = dtf_users.merge(dtf_products[["movieId","product","genres"]], how="left")

    movie_rating_df = dtf_product_merge

    st.subheader("Example of User - Movie rating data")

    st.sidebar.subheader("Filter Displayed User Accounts")

    user_rating_count = movie_rating_df.groupby(["user"])["y"].agg('count').reset_index()
    user_rating_count_filter = user_rating_count[user_rating_count["y"] > 4]


    users = list(user_rating_count_filter.user.unique())
    user_selections = st.sidebar.selectbox(
        "Select Accounts to View", options=users, index=1
    )

    movie_id = movie_rating_df.loc[movie_rating_df.user == user_selections]

    with st.expander("Raw Dataframe"):
        st.write(movie_rating_df)

    # movie_selected_df = movie_id.merge(movie_titles, left_on='product', right_on='id', how='inner')
    # movie_selected_df["user"] = movie_selected_df["user"].astype(int)
    # movie_selected_df["item"] = movie_selected_df["item"].astype(int)
    st.subheader("Selected User Account and Rating History")
    # st.write(movie_selected_df)
    st.write(movie_id[["user","product","name","genres","daytime","weekend","y"]].rename(columns={"product": "movieId", "y": "rating"}))
    cellsytle_jscode = JsCode(
        """
    function(params) {
        if (params.value > 0) {
            return {
                'color': 'white',
                'backgroundColor': 'forestgreen'
            }
        } else if (params.value < 0) {
            return {
                'color': 'white',
                'backgroundColor': 'crimson'
            }
        } else {
            return {
                'color': 'white',
                'backgroundColor': 'slategray'
            }
        }
    };
    """
    )

    gb = GridOptionsBuilder.from_dataframe(movie_id)
    # gb.configure_columns(
    #     (
    #         "last_price_change",
    #         "total_gain_loss_dollar",
    #         "total_gain_loss_percent",
    #         "today's_gain_loss_dollar",
    #         "today's_gain_loss_percent",
    #     ),
    #     cellStyle=cellsytle_jscode,
    # )
    # gb.configure_pagination()
    gb.configure_columns(("product", "name", "genres"), pinned=True)
    gridOptions = gb.build()

    unrated_to_recommend = unrated_df[unrated_df["y"].isna()]
    unrated_to_recommend["yhat"] = loaded_model.predict([unrated_to_recommend["user"], unrated_to_recommend["product"], unrated_to_recommend[features], unrated_to_recommend[context]])

    if st.button('Show Recommendation for user'):
        recommended_title = unrated_to_recommend[unrated_to_recommend["user"] == user_selections].sort_values(by=['yhat'],ascending=False).head(5)
        final_result = recommended_title[["user","product"]].merge(dtf_products_use.reset_index()[["product","name","genres"]], how="inner", on="product")
        
        st.subheader("Top 5 Movie recommendation for user: " + str(user_selections))
        st.write(final_result)



if __name__ == "__main__":
    st.set_page_config(
        "Fidelity Account View by Gerard Bentley",
        "📊",
        initial_sidebar_state="expanded",
        layout="wide",
    )
    main()
