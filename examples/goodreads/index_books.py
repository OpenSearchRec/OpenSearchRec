import datetime
import requests
import json
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

api_base_url = "http://localhost:8000"
index_name = "goodreads_books"
indexing_batch_size = 1000


model = SentenceTransformer('all-mpnet-base-v2')


def get_author_id_to_author_info_dict():
    authors = {}
    with open("data/goodreads_book_authors.json", "r") as f:
        for author_info in f.readlines():
            author_info = json.loads(author_info)
            authors[author_info["author_id"]] = author_info
    return authors


with open("alternating_least_square_embeddings/books_als_embeddings.json", "r") as f:
    book_id_to_book_embedding_dict = json.load(f)


def index_books():
    resp = requests.post(f"{api_base_url}/index/create_index/{index_name}", json={
        "text_matching_config": {
            "text_matching_type": "boolean_matching",
            "settings": {
                "enable_ngram_tokenizer": True,
                "ngram_tokenizer_min_gram": 3,
                "ngram_tokenizer_max_gram": 4
            }
        },
        "text_fields": [
            "title",
            "description",
            "authors",
            "publisher"
        ],
        "categorical_fields": [
            "tags"
        ],
        "numeric_fields": [
            "book_quality_signal",
            "book_popularity_signal"
        ],
        "date_fields": [
            "published_date"
        ],
        "embedding_fields": {
            "book_alternating_least_squares_embeddings": {
                "embedding_dimension": 50,
                "enable_approximate_nearest_embedding_search": True
            },
            "title_embedding": {
                "embedding_dimension": 768,
                "enable_approximate_nearest_embedding_search": True
            }
        },
        "number_of_shards": 1,
    })
    print("created index", resp, resp.text)

    assert resp.status_code == 200, f"resp.status_code = {resp.status_code}"

    authors = get_author_id_to_author_info_dict()

    with open("data/goodreads_books.json", "r") as f:
        books_batch = []
        for book_line in tqdm(f):
            book = json.loads(book_line)

            authors_info_list = [authors.get(author["author_id"]) for author in book["authors"] if author["author_id"] in authors]
            authors_name_list = [author["name"] for author in authors_info_list]
            authors_index_info_list = [{"author_id": author["author_id"], "author_name": author["name"]} for author in authors_info_list]

            item_tags = ["authorid_" + author["author_id"] for author in authors_info_list]
            if len(book["publisher"]) > 0:
                item_tags.append("publisher_" + book["publisher"])

            book_item = {
                "id": book["book_id"],
                "text_fields": {
                    "title": book["title"],
                    "description": book["description"],
                    "authors": ", ".join(authors_name_list),
                    "publisher": book["publisher"]
                },
                "categorical_fields": {
                    "tags": item_tags
                },
                "date_fields": {},
                "numeric_fields": {},
                "embedding_fields": {
                    "title_embedding": model.encode(book["title"]).tolist(),
                },
                "extra_information":{
                    "url": book["url"],
                    "image_url": book["image_url"],
                    "authors_info_list": authors_index_info_list
                }
            }

            if len(book["average_rating"]) > 0:
                book_item["numeric_fields"]["book_quality_signal"] =  float(book["average_rating"])

            if len(book["ratings_count"]) > 0:
                book_item["numeric_fields"]["book_popularity_signal"] =  float(book["ratings_count"]) 

            if book["book_id"] in book_id_to_book_embedding_dict:
                book_item["embedding_fields"]["book_alternating_least_squares_embeddings"] = book_id_to_book_embedding_dict[book["book_id"]]            

            try:
                if len(book["publication_year"]) == 4 and len(book["publication_month"]) in [1,2] and len(book["publication_day"]) in [1,2]:
                    book_item["date_fields"]["published_date"] = \
                        str(datetime.datetime(year=int(book["publication_year"]),
                                              month=int(book["publication_month"]),
                                              day=int(book["publication_day"])))
            except Exception as e:
                print()
                print("Date error:")
                print(e)
                print(int(book["publication_year"]), int(book["publication_month"]), int(book["publication_day"]))

            books_batch.append(book_item)


            if len(books_batch) % indexing_batch_size == 0:
                resp = requests.post(f"{api_base_url}/item/bulk_index/{index_name}", json=books_batch)
                print("bulk index resp", resp, resp.text)

                books_batch = []

        if len(books_batch) > 0:
            resp = requests.post(f"{api_base_url}/item/bulk_index/{index_name}", json=books_batch)
            print("bulk index resp", resp, resp.text)
            books_batch = []
            assert resp.status_code == 200, resp.text


if __name__ == "__main__":
    index_books()