import pandas as pd

from CollaborativeFiltering import CollaborativeFiltering

if __name__ == '__main__':

    ratings = pd.read_csv('./ml-1m/ratings.dat', engine='python', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'])

    collaborative_filtering = CollaborativeFiltering(ratings)

    user_id               = 1
    similarity_user_count = 5
    recommend_user_count  = 5

    result = collaborative_filtering.recommend_item(user_id, similarity_user_count, recommend_user_count)

    print(result)