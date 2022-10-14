import requests
import json
from tqdm import tqdm

api_base_url = "http://localhost:8000"
index_name = "goodreads_authors"
indexing_batch_size = 10000


with open("alternating_least_square_embeddings/authors_als_embeddings.json", "r") as f:
    author_id_to_author_als_embedding_dict = json.load(f)


def index_authors():
    authors = []
    with open("data/goodreads_book_authors.json", "r") as f:
        for author_info in f.readlines():
            authors.append(json.loads(author_info))

    print("Number of authors: ", len(authors))

    resp = requests.post(f"{api_base_url}/index/create_index/{index_name}", json={
        "text_fields": [
            "author_name"
        ],
        "numeric_fields": [
            "author_quality_signal",
            "author_popularity_signal"
        ],
        "embedding_fields": {
            "author_alternating_least_squares_embeddings": {
                "embedding_dimension": 50,
                "enable_approximate_nearest_embedding_search": True
            }
        }
    })

    print("created index", resp, resp.text)

    authors_batch = []
    for author in tqdm(authors):
        author_document = {
            "id": author["author_id"],
            "text_fields": {
                "author_name": author["name"]
            },
            "numeric_fields": {
                "author_quality_signal": author["average_rating"],
                "author_popularity_signal": author["ratings_count"]
            },
            "embedding_fields": {}
        }

        if str(author["author_id"]) in author_id_to_author_als_embedding_dict:
            author_document["embedding_fields"]["author_alternating_least_squares_embeddings"] = \
                author_id_to_author_als_embedding_dict[str(author["author_id"])]

        authors_batch.append(author_document)

        if len(authors_batch) % indexing_batch_size == 0:
            resp = requests.post(f"{api_base_url}/item/bulk_index/{index_name}", json=authors_batch)
            authors_batch = []

    if len(authors_batch) > 0:
        resp = requests.post(f"{api_base_url}/item/bulk_index/{index_name}", json=authors_batch)
        authors_batch = []
        assert resp.status_code == 200, resp.text


if __name__ == "__main__":
    index_authors()
