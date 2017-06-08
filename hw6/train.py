#!/usr/bin/env python
# -- coding: utf-8 --
"""
Matrix Factorization For Movie Ratings Prediction
"""

import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint
from model import History, build_model
from utils import dump_history

np.random.seed(0)

VALIDATION_SPLIT = 0.1
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
LOG_DIR = os.path.join(BASE_DIR, 'log')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def main():
    """ Main function """
    parser = ArgumentParser()
    parser.add_argument('--dnn', action='store_true', help='Use DNN model')
    parser.add_argument('--normal', action='store_true', help='Normalize ratings')
    parser.add_argument('--dim', type=int, default=128, help='Specify latent dimensions')
    args = parser.parse_args()

    ratings_df = pd.read_csv(os.path.join(BASE_DIR, 'data/train.csv'),
                             sep=',')
    max_user_id = ratings_df['UserID'].drop_duplicates().max()
    max_movie_id = ratings_df['MovieID'].drop_duplicates().max()
    print("{:d} ratings loaded".format(len(ratings_df)))

    users = ratings_df['UserID'].values - 1
    print("Users: {:s}, shape = {:s}".format(str(users), str(users.shape)))
    movies = ratings_df['MovieID'].values - 1
    print("Movies: {:s}, shape = {:s}".format(str(movies), str(movies.shape)))
    ratings = ratings_df['Rating'].values
    print("Ratings: {:s}, shape = {:s}".format(str(ratings), str(ratings.shape)))

    users_df = pd.read_csv(os.path.join(BASE_DIR, 'data/users.csv'), sep='::', engine='python')
    users_age = (users_df['Age'] - np.mean(users_df['Age'])) / np.std(users_df['Age'])
    movies_df = pd.read_csv(os.path.join(BASE_DIR, 'data/movies.csv'), sep='::', engine='python')

    all_genres = np.array([])
    for genres in movies_df['Genres']:
        for genre in genres.split('|'):
            all_genres = np.append(all_genres, genre)
    all_genres = np.unique(all_genres)

    users_info = np.zeros((max_user_id, 23))
    movies_info = np.zeros((max_movie_id, all_genres.shape[0]))

    for idx, user_id in enumerate(users_df['UserID']):
        gender = 1 if users_df['Gender'][idx] == 'M' else 0
        occu = np.zeros(np.max(np.unique(users_df['Occupation'])) + 1)
        occu[users_df['Occupation'][idx]] = 1
        tmp = [gender, users_age[idx]]
        tmp.extend(occu)
        users_info[user_id - 1] = tmp

    for idx, movie_id in enumerate(movies_df['movieID']):
        genres = movies_df['Genres'][idx].split('|')
        tmp = np.zeros(all_genres.shape[0])
        for genre in genres:
            tmp[np.where(all_genres == genre)[0][0]] = 1
        movies_info[movie_id - 1] = tmp

    if args.normal:
        mean = np.mean(ratings)
        std = np.std(ratings)
        ratings = (ratings - mean) / std

    if args.dnn:
        model = build_model(max_user_id, max_movie_id, args.dim, 'dnn', users_info, movies_info)
        model.compile(loss='mse', optimizer='adam')
    else:
        model = build_model(max_user_id, max_movie_id, args.dim, 'mf', users_info, movies_info)
        model.compile(loss='mse', optimizer='adam')

    model.summary()

    model_name = os.path.join(MODEL_DIR, "{epoch:02d}_dnn_model.hdf5"
                              if args.dnn else "{epoch:02d}_mf_model.hdf5")
    checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=0,
                                 save_best_only=True, mode='min')
    history = History()
    callbacks_list = [checkpoint, history]

    indices = np.random.permutation(users.shape[0])
    val_num = int(users.shape[0] * VALIDATION_SPLIT)
    users = users[indices]
    movies = movies[indices]
    ratings = ratings[indices]
    tra_users = users[:-val_num]
    tra_movies = movies[:-val_num]
    tra_ratings = ratings[:-val_num]
    val_users = users[-val_num:]
    val_movies = movies[-val_num:]
    val_ratings = ratings[-val_num:]

    model.fit([tra_users, tra_movies], tra_ratings,
              batch_size=256,
              epochs=300 if args.dnn else 30,
              validation_data=([val_users, val_movies], val_ratings),
              callbacks=callbacks_list)

    dump_history(LOG_DIR, history)

if __name__ == '__main__':

    main()
