import functools
from pathlib import Path

import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.shared import JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd
import plotly.express as px
from typing import List

import numpy as np
import pickle


# Define the MF class for the matrix factorization model
class MF(object):
    """docstring for CF"""
    def __init__(self, Y_data, K, lam = 0.1, Xinit = None, Winit = None, 
            learning_rate = 0.5, max_iter = 1000, print_every = 100, user_based = 1):
        self.Y_raw_data = Y_data
        self.K = K
        # regularization parameter
        self.lam = lam
        # learning rate for gradient descent
        self.learning_rate = learning_rate
        # maximum number of iterations
        self.max_iter = max_iter
        # print results after print_every iterations
        self.print_every = print_every
        # user-based or item-based
        self.user_based = user_based
        # number of users, items, and ratings. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(Y_data[:, 0])) + 1 
        self.n_items = int(np.max(Y_data[:, 1])) + 1
        self.n_ratings = Y_data.shape[0]
        
        if Xinit is None: # new
            self.X = np.random.randn(self.n_items, K)
        else: # or from saved data
            self.X = Xinit 
        
        if Winit is None: 
            self.W = np.random.randn(K, self.n_users)
        else: # from saved data
            self.W = Winit
            
        # normalized data, update later in normalized_Y function
        self.Y_data_n = self.Y_raw_data.copy()
        self.Y_data_n = self.Y_data_n.astype(float)


    def normalize_Y(self):
        if self.user_based:
            user_col = 0
            item_col = 1
            n_objects = self.n_users

        # if we want to normalize based on item, just switch first two columns of data
        else: # item bas
            user_col = 1
            item_col = 0 
            n_objects = self.n_items

        users = self.Y_raw_data[:, user_col] 
        self.mu = np.zeros((n_objects,))

        for n in range(n_objects):
            # row indices of rating done by user n
            # since indices need to be integers, we need to convert
            ids = np.where(users == n)[0].astype(np.int32)
            # indices of all ratings associated with user n
            item_ids = self.Y_data_n[ids, item_col] 
            # and the corresponding ratings 
            ratings = self.Y_data_n[ids, 2].astype(float)
            # take mean
            m = np.mean(ratings)
            # print(m)
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            self.mu[n] = m
            # normalize
            self.Y_data_n[ids, 2] = ratings - self.mu[n]
            
            print("user: " + str(n) + "> rating: "+ str(ratings) + " - y_data_n: " + str(self.Y_data_n[ids, 2]) + " = norm: " + str(ratings - self.mu[n]))


    """
    When there are new data, we update the Utility matrix by adding these new records to the end of the matrix, then doing the normalization as well as update the X and W
    """
    def add(self, new_data):
        """
        Update Y_data matrix when new ratings come.
        For simplicity, suppose that there is no new user or item.
        """
        self.Y_raw_data = np.concatenate((self.Y_raw_data, new_data), axis = 0)
        self.Y_data_n = self.Y_raw_data.copy().astype(float)
        self.n_users = np.max(self.Y_raw_data[:, 0].astype(int)) + 1
        self.n_items = np.max(self.Y_raw_data[:, 1].astype(int)) + 1
        self.normalize_Y()
        self.updateX()
        self.updateW()


    # Calculate the loss function:
    def loss(self):
        L = 0 
        for i in range(self.n_ratings):
            # user, item, rating
            n, m, rate = int(self.Y_data_n[i, 0]), int(self.Y_data_n[i, 1]), self.Y_data_n[i, 2]
            L += 0.5*(rate - self.X[m, :].dot(self.W[:, n]))**2
        
        # take average
        L /= self.n_ratings
        # regularization, don't ever forget this 
        L += 0.5*self.lam*(np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro'))
        return L


    def updateX(self):
        for m in range(self.n_items):
            user_ids, ratings = self.get_users_who_rate_item(m)
            Wm = self.W[:, user_ids]
            # gradient
            grad_xm = -(ratings - self.X[m, :].dot(Wm)).dot(Wm.T)/self.n_ratings + \
                                               self.lam*self.X[m, :]
            self.X[m, :] -= self.learning_rate*grad_xm.reshape((self.K,))
    
    def updateW(self):
        for n in range(self.n_users):
            item_ids, ratings = self.get_items_rated_by_user(n)
            Xn = self.X[item_ids, :]
            # gradient
            grad_wn = -Xn.T.dot(ratings - Xn.dot(self.W[:, n]))/self.n_ratings + \
                        self.lam*self.W[:, n]
            self.W[:, n] -= self.learning_rate*grad_wn.reshape((self.K,))


    # Get items which has been rated by user and user who has rated an item, and the respectively rating
    def get_items_rated_by_user(self, user_id):
        """
        get all items which are rated by user user_id, and the corresponding ratings
        """
        ids = np.where(self.Y_data_n[:,0] == user_id)[0] 
        item_ids = self.Y_data_n[ids, 1].astype(np.int32) # indices need to be integers
        ratings = self.Y_data_n[ids, 2]
        return (item_ids, ratings)
        
        
    def get_users_who_rate_item(self, item_id):
        """
        get all users who rated item item_id and get the corresponding ratings
        """
        ids = np.where(self.Y_data_n[:,1] == item_id)[0] 
        user_ids = self.Y_data_n[ids, 0].astype(np.int32)
        ratings = self.Y_data_n[ids, 2]
        return (user_ids, ratings)


    def fit(self):
        self.normalize_Y()
        for it in range(self.max_iter):
            self.updateX()
            self.updateW()
            if (it + 1) % self.print_every == 0:
                rmse_train = self.evaluate_RMSE(self.Y_raw_data)
                print ('iter =', it + 1, ', loss =', self.loss(), ', RMSE train =', rmse_train)


    def pred(self, u, i):
        """ 
        predict the rating of user u for item i 
        if you need the un
        """
        u = int(u)
        i = int(i)
        if self.user_based:
            bias = self.mu[u]
        else: 
            bias = self.mu[i]
        pred = self.X[i, :].dot(self.W[:, u]) + bias 
        # truncate if results are out of range [0, 5]
        if pred < 0:
            return 0 
        if pred > 5: 
            return 5 
        return pred 
        
    
    def pred_for_user(self, user_id):
        """
        predict ratings one user give all unrated items
        """
        ids = np.where(self.Y_data_n[:, 0] == user_id)[0]
        items_rated_by_u = [int(i) for i in self.Y_data_n[ids, 1].tolist()]      
        # st.write(items_rated_by_u)         
        
        y_pred = self.X.dot(self.W[:, user_id])
        # print(y_pred.shape)

        for i in items_rated_by_u:
            # st.write(self.mu[i])
            y_pred[i] = y_pred[i] + self.mu[i]

        predicted_ratings= []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                predicted_ratings.append((i, y_pred[i]))
        
        return predicted_ratings

    def evaluate_RMSE(self, rate_test):
        n_tests = rate_test.shape[0]
        SE = 0 # squared error
        for n in range(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            SE += (pred - rate_test[n, 2])**2 

        RMSE = np.sqrt(SE/n_tests)
        return RMSE


def main() -> None:
    st.header("Netflix movie recommendation:")
    movie_titles = pd.read_csv('movie_titles.csv', delimiter =",", encoding='utf-8', names=["id","year","title"])

    filename = 'finalized_model_2.sav'
    loaded_rs = pickle.load(open(filename, 'rb'))

    data = loaded_rs.Y_data_n

    movie_rating_df = pd.DataFrame(data, columns=['user','item','rating'])

    st.subheader("Example of User - Movie rating data")

    st.sidebar.subheader("Filter Displayed User Accounts")

    user_rating_count = movie_rating_df.groupby(["user"])["rating"].agg('count').reset_index()
    user_rating_count_filter = user_rating_count[user_rating_count["rating"] > 4]


    users = list(user_rating_count_filter.user.unique())
    user_selections = st.sidebar.selectbox(
        "Select Accounts to View", options=users, index=1
    )

    movie_id = movie_rating_df.loc[movie_rating_df.user == user_selections]

    with st.expander("Raw Dataframe"):
        st.write(movie_rating_df)

    movie_selected_df = movie_id.merge(movie_titles, left_on='item', right_on='id', how='inner')
    movie_selected_df["user"] = movie_selected_df["user"].astype(int)
    movie_selected_df["item"] = movie_selected_df["item"].astype(int)
    st.subheader("Selected User Account and Rating History")
    st.write(movie_selected_df)
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

    gb = GridOptionsBuilder.from_dataframe(movie_selected_df)
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
    gb.configure_columns(("id", "title", "year"), pinned=True)
    gridOptions = gb.build()


    if st.button('Show Recommendation for user'):
        st.subheader("Top 5 Movie recommendation for user: " + str(user_selections))
        recommended = pd.DataFrame(loaded_rs.pred_for_user(int(user_selections)), columns=["Movie", "Predict_Rating"]).sort_values(by=['Predict_Rating'],ascending=False).head(5)
        recommended_title = recommended.merge(movie_titles, left_on='Movie', right_on='id', how='inner')[["Movie","title","Predict_Rating"]]
        st.write(recommended_title)



if __name__ == "__main__":
    st.set_page_config(
        "Fidelity Account View by Gerard Bentley",
        "📊",
        initial_sidebar_state="expanded",
        layout="wide",
    )
    main()
