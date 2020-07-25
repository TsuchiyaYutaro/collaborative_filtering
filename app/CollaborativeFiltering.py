import pandas as pd
import numpy as np

class CollaborativeFiltering():
    def __init__(self, original_df, sim_type='cos'):
        
        self.original_df = original_df.pivot_table(index='user_id', columns='movie_id', values='rating', fill_value=0)
        self.sim_type = sim_type
        
        self.similarity_df = self._calc_sim()
        
    def _calc_sim(self):
        
        def _cos_sim(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
        def _euc_dis(x, y):   
            return -np.sqrt(np.sum((x - y) ** 2))
    
        result = pd.DataFrame(index=self.original_df.index)

        if self.sim_type=='cos':
            for i in self.original_df.index:
                target = self.original_df[self.original_df.index == i].values[0]
                tmp = pd.DataFrame(self.original_df.apply(lambda x: _cos_sim(target, x), axis=1), columns=[i])
                result = pd.concat([result, tmp], axis=1)
        else:
            for i in self.original_df.index:
                target = self.original_df[self.original_df.index == i].values[0]
                tmp = pd.DataFrame(self.original_df.apply(lambda x: _euc_dis(target, x), axis=1), columns=[i])
                result = pd.concat([result, tmp], axis=1)

        return result
    
    def recommend_item(self, user_id, similarity_user_count, recommend_user_count):
        # TODO: Refactoring
        similarity_user = self.similarity_df.iloc[user_id]
        similarity_user = similarity_user.sort_values(ascending=False)
        similarity_user = similarity_user.iloc[1:similarity_user_count+1]
        similarity_user_list = similarity_user.index.tolist()
        
        user_original_df = self.original_df.reset_index()
        similarity_user = user_original_df[user_original_df['user_id'].isin(similarity_user_list)]

        target = self.original_df[self.original_df.index==user_id].T
        target = target[target[1]==0].T
        target_movie_list = target.columns.tolist()

        similarity_user = similarity_user[target_movie_list]

        similarity_user_agg = similarity_user.sum()
        similarity_user_agg = similarity_user_agg.sort_values(ascending=False)
        similarity_user_agg = similarity_user_agg.iloc[0:recommend_user_count]
        similarity_user_agg = similarity_user_agg.index.tolist()
        
        return similarity_user_agg