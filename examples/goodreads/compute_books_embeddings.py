import pandas as pd
import numpy as np
import json

from OpenSearchRec.models import (
    AlternatingLeastSquares
)

als_factors = 50
iterations = 50


def get_interactions():
    interactions_df = pd.read_csv("data/goodreads_interactions.csv")

    interactions_df = interactions_df.dropna()

    print("Num interactions", len(interactions_df))

    user_ids = list(interactions_df["user_id"])
    item_ids = list(interactions_df["book_id"])
    ratings = list(interactions_df["rating"])

    user_ids = [str(uid) for uid in user_ids]
    item_ids = [str(iid) for iid in item_ids]
    interaction_strengths = [rating_to_score(r) for r in ratings]

    return user_ids, item_ids, interaction_strengths


def rating_to_score(rating):
    score = 1
    if rating == 4:
        score = 2
    if rating == 5:
        score = 4
    return score


user_ids, item_ids, interaction_strengths = get_interactions()

book_id_set = set(item_ids)

print("len(user_ids)", len(user_ids))
print("len(item_ids)", len(item_ids))
print("len(interaction_strengths)", len(interaction_strengths))
print("len(book_id_set)", len(book_id_set))

print("Fitting AlternatingLeastSquares Model")
als = AlternatingLeastSquares(factors=als_factors, regularization=0.01, iterations=iterations, use_gpu=False)
als.fit(user_ids, item_ids, interaction_strengths)

print("Done Fitting AlternatingLeastSquares Model")

# Map the book ids used in goodreads_interactions.csv to the book ids used elsewhere
book_id_map_df = pd.read_csv("data/book_id_map.csv")
book_id_to_book_embedding_dict = {}
for interactions_book_id, book_id in zip(book_id_map_df["book_id_csv"], book_id_map_df["book_id"]):
    if str(interactions_book_id) in book_id_set:
        book_id_to_book_embedding_dict[str(book_id)] = als.get_item_embedding(str(interactions_book_id))


print("len(book_id_to_book_embedding_dict.keys())", len(book_id_to_book_embedding_dict.keys()))

print("Saving to file")
with open("alternating_least_square_embeddings/books_als_embeddings.json", "w") as f:
    json.dump(book_id_to_book_embedding_dict, f)
