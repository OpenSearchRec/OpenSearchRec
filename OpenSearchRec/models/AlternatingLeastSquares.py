from pydantic import BaseModel
import implicit
from scipy.sparse import csr_matrix
import numpy as np
from typing import List


class AlternatingLeastSquares:
    def __init__(self, **implicit_model_params):
        self.model = implicit.als.AlternatingLeastSquares(**implicit_model_params)

    def fit(self, user_ids: List[str], item_ids: List[str], interaction_strengths: List[float]):
        user_ids = [str(user_id) for user_id in user_ids]
        item_ids = [str(item_id) for item_id in item_ids]

        self.user_id_to_user_idx = {}
        self.item_id_to_item_idx = {}

        for user_id in user_ids:
            if user_id not in self.user_id_to_user_idx:
                self.user_id_to_user_idx[user_id] = len(self.user_id_to_user_idx)
        for item_id in item_ids:
            if item_id not in self.item_id_to_item_idx:
                self.item_id_to_item_idx[item_id] = len(self.item_id_to_item_idx)

        user_col = [self.user_id_to_user_idx[user_id] for user_id in user_ids]
        item_col = [self.item_id_to_item_idx[item_id] for item_id in item_ids]

        num_users = len(self.user_id_to_user_idx)
        num_items = len(self.item_id_to_item_idx)

        user_items = \
            csr_matrix(
                (np.array(interaction_strengths), (np.array(user_col), np.array(item_col))),
                shape=(num_users, num_items))

        self.model.fit(user_items)

        return self.model, self.user_id_to_user_idx, self.item_id_to_item_idx

    def get_user_embedding(self, user_id):
        if str(user_id) not in self.user_id_to_user_idx:
            return None
        user_idx = self.user_id_to_user_idx[str(user_id)]
        return self.model.user_factors[user_idx].tolist()

    def get_item_embedding(self, item_id):
        if str(item_id) not in self.item_id_to_item_idx:
            return None
        item_idx = self.item_id_to_item_idx[str(item_id)]
        return self.model.item_factors[item_idx].tolist()
