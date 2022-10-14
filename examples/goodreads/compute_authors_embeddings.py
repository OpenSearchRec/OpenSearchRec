import pandas as pd
import numpy as np
import json

from OpenSearchRec.models import (
    AlternatingLeastSquares
)

als_factors = 50
iterations = 50


def get_interactions():
    authors = {}
    with open("data/goodreads_book_authors.json", "r") as f:
        for author_info in f.readlines():
            author_info = json.loads(author_info)
            authors[author_info["author_id"]] = author_info

    book_id_to_authors_id_map = {}
    author_set = set()
    with open("data/goodreads_books.json", "r") as f:
        for book_line in f:
            book = json.loads(book_line)
            book_id = int(book["book_id"])
            if book_id not in book_id_to_authors_id_map:
                authors_id_list = [int(author["author_id"]) for author in book["authors"] ]
                book_id_to_authors_id_map[book_id] = authors_id_list

                author_set.update(authors_id_list)

    print("Num authors", len(author_set))
    print("Max author id", max(author_set))

    interactions_df = pd.read_csv("data/goodreads_interactions.csv")
    interactions_df = interactions_df.dropna()

    print("Num interactions", len(interactions_df))

    book_id_map_df = pd.read_csv("data/book_id_map.csv")
    book_interaction_id_to_book_id = {
        int(row["book_id_csv"]): int(row["book_id"])
        for _, row in book_id_map_df.iterrows()
    }

    user_ids = []
    author_ids = []
    interaction_strengths = []
    for idx, rating_row in interactions_df.iterrows():
        book_interaction_id = int(rating_row["book_id"])
        book_id = book_interaction_id_to_book_id[book_interaction_id]
        for author_id in book_id_to_authors_id_map.get(book_id, []):
            if idx % 1000000 == 0:
                print(idx, "/", len(interactions_df))

            user_ids.append(rating_row["user_id"])
            author_ids.append(author_id)

            interaction_strengths.append(rating_to_score(rating_row["rating"]))

    return user_ids, author_ids, interaction_strengths


def rating_to_score(rating):
    score = 1
    if rating == 4:
        score = 2
    if rating == 5:
        score = 4
    return score


user_ids, author_ids, interaction_strengths = get_interactions()

author_id_set = set(author_ids)

print("len(user_ids)", len(user_ids))
print("len(author_ids)", len(author_ids))
print("len(interaction_strengths)", len(interaction_strengths))
print("len(author_id_set)", len(author_id_set))

print("Fitting AlternatingLeastSquares Model")
als = AlternatingLeastSquares(factors=als_factors, regularization=0.01, iterations=iterations, use_gpu=False)
als.fit(user_ids, author_ids, interaction_strengths)

print("Done Fitting AlternatingLeastSquares Model")


author_id_to_embedding_dict = {}
for author_id in set(author_ids):
    author_id_to_embedding_dict[str(author_id)] = als.get_item_embedding(str(author_id))


print("len(author_id_to_embedding_dict.keys())", len(author_id_to_embedding_dict.keys()))

print("Saving to file")
with open("alternating_least_square_embeddings/authors_als_embeddings.json", "w") as f:
    json.dump(author_id_to_embedding_dict, f)
