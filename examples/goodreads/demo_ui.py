from flask import Flask, request, render_template
import json
import requests

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from sentence_transformers import SentenceTransformer

from OpenSearchRec.post_ranking import select_items_greedily_and_with_similarity_suppression


opensearchrec_base_url = "http://localhost:8000"
index_name = "goodreads_books"
author_index_name = "goodreads_authors"

model = SentenceTransformer('all-mpnet-base-v2')

app = Flask(__name__)

@app.route('/')
def homepage():
    q = request.args.get("q", "")

    if len(q) > 0:
        books_search_request = {
            "text_matching": [
                {
                    "query": q,
                    "text_fields_names_and_weights_dict": {
                        "title": 1.5,
                        "description": 0.1,
                        "authors": 1,
                        "publisher": 0.5
                    },
                    "use_ngram_fields_if_available": True,
                    "required": True,
                    "minimum_should_match": "49%"
                },
                {
                    "query": q,
                    "text_fields_names_and_weights_dict": {
                        "title": 3,
                        "description": 0.2,
                        "authors": 2,
                        "publisher": 1
                    },
                    "use_ngram_fields_if_available": False,
                    "required": False,
                    "minimum_should_match": 1
                },
            ],
            "numeric_fields_boosting": [
                {
                "input_value_config": {
                    "field_name": "book_popularity_signal",
                    "default_field_value_if_missing": 0
                },
                "boost_function_config": {
                    "boost_function_input_value_rescaling_factor": 1,
                    "boost_function_increasing_function_type": "log1p"
                },
                "scoring_factor_config": {
                    "score_multiplier_constant_weight": 1,
                    "score_multiplier_variable_boost_weight": 1,
                    "score_multiplier_minimum_value": 0,
                    "score_multiplier_maximum_value": 10
                }
                }
            ],
            "embedding_fields_to_return": [],  # don't return embeddings to reduce data transfer and latency 
            "limit": 50,
            "start": 0
        }

        print("q:", q )
        resp = requests.post(f"{opensearchrec_base_url}/search/{index_name}",
                             json=books_search_request
                             )
        print("resp", resp)
        print("resp", resp.text[:1000])
        resp_json = resp.json()

        hits = resp_json["results"]

        authors_search_request = {
            "text_matching": [
                {
                    "query": q,
                    "text_fields_names_and_weights_dict": {
                        "author_name": 1,
                    },
                    "use_ngram_fields_if_available": True,
                    "required": True,
                    "minimum_should_match": "90%"
                },
                {
                    "query": q,
                    "text_fields_names_and_weights_dict": {
                        "author_name": 2,
                    },
                    "use_ngram_fields_if_available": False,
                    "required": False,
                    # "minimum_should_match": 1
                }
            ],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "author_popularity_signal",
                        "default_field_value_if_missing": 0
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 1,
                        "boost_function_increasing_function_type": "log1p"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 6,
                        "score_multiplier_variable_boost_weight": 1,
                        "score_multiplier_minimum_value": 0,
                        "score_multiplier_maximum_value": 12
                    }
                }
            ],
            "embedding_fields_to_return": [],  # don't return embeddings to reduce data transfer and latency 
            "limit": 1,
            "start": 0
        }

        print("q:", q )
        resp = requests.post(f"{opensearchrec_base_url}/search/{author_index_name}",
                             json=authors_search_request
                             )
        print("resp", resp)
        print("resp", resp.text[:1000])
        resp_json = resp.json()

        authors_hits = resp_json["results"]

    else:
        hits = []
        authors_hits = []

    return render_template("search.html", q=q, hits=hits, authors_hits=authors_hits)


@app.route('/book/<book_id>')
def book(book_id):
    book_resp = requests.get(f"{opensearchrec_base_url}/item/{index_name}/{book_id}")
    book_json = book_resp.json()

    print("book_json", book_json)

    if "book_alternating_least_squares_embeddings" in book_json["embedding_fields"]:

        similar_books_resp = requests.post(f"{opensearchrec_base_url}/search/{index_name}", json={
            "embedding_fields_boosting": [
                {
                "input_value_config": {
                    "field_name": "book_alternating_least_squares_embeddings",
                    "target_embedding": book_json["embedding_fields"]["book_alternating_least_squares_embeddings"],
                    # "default_embedding_comparison_value_if_missing": 0,
                    "embedding_comparison_metric": "cosine_similarity"
                },
                "boost_function_config": {
                    "boost_function_input_value_rescaling_factor": 1,
                    "boost_function_increasing_function_type": "none"
                },
                "scoring_factor_config": {
                    "score_multiplier_constant_weight": 1,
                    "score_multiplier_variable_boost_weight": 1,
                    "score_multiplier_minimum_value": 0,
                    "score_multiplier_maximum_value": 10
                }
                }
            ],
            "numeric_fields_boosting": [
                {
                "input_value_config": {
                    "field_name": "book_popularity_signal",
                    "default_field_value_if_missing": 0
                },
                "boost_function_config": {
                    "boost_function_input_value_rescaling_factor": 1,
                    "boost_function_increasing_function_type": "log1p"
                },
                "scoring_factor_config": {
                    "score_multiplier_constant_weight": 6,
                    "score_multiplier_variable_boost_weight": 1,
                    "score_multiplier_minimum_value": 0,
                    "score_multiplier_maximum_value": 12
                }
                }
            ],
            "id_exclude_list": [book_json["id"]],
            "limit": 20,
            "start": 0
        })
        print("similar_books_resp", similar_books_resp)
        print("similar_books_resp", similar_books_resp.text)
        similar_books_resp_json = similar_books_resp.json()
    else:
        similar_books_resp_json = {"results": []}


    return render_template("book.html", book=book_json, similar_books=similar_books_resp_json["results"])


@app.route('/author/<author_id>')
def author(author_id):

    author_resp = requests.get(f"{opensearchrec_base_url}/item/{author_index_name}/{author_id}")
    author_json = author_resp.json()

    books_by_author = \
        requests.post(
            f"{opensearchrec_base_url}/search/{index_name}",
            json={
                "categorical_matching": [
                    {
                        "categorical_field_name": "tags",
                        "values_list": [
                            "authorid_" + str(author_json["id"])
                        ],
                        "score_multiplier": 0,
                        "minimum_should_match": 1,
                        "required": True
                    }
                ],
                "numeric_fields_boosting": [
                    {
                    "input_value_config": {
                        "field_name": "book_popularity_signal",
                        "default_field_value_if_missing": 0
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 1,
                        "boost_function_increasing_function_type": "log1p"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 1,
                        "score_multiplier_variable_boost_weight": 0.5,
                        "score_multiplier_minimum_value": 0,
                        # "score_multiplier_maximum_value": 5
                    }
                    }
                ],
                "embedding_fields_to_return": [],  # don't return embeddings to reduce data transfer and latency 
                "limit": 1000
            })
    books_by_author = books_by_author.json()["results"]

    if "author_alternating_least_squares_embeddings" in author_json.get("embedding_fields", {}):

        similar_authors_resp = requests.post(f"{opensearchrec_base_url}/search/{author_index_name}", json={
            "embedding_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "author_alternating_least_squares_embeddings",
                        "target_embedding": author_json["embedding_fields"]["author_alternating_least_squares_embeddings"],
                        "embedding_comparison_metric": "cosine_similarity"
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 1,
                        "boost_function_increasing_function_type": "none"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 1,
                        "score_multiplier_variable_boost_weight": 1,
                        "score_multiplier_minimum_value": 0,
                        "score_multiplier_maximum_value": 3
                    }
                }
            ],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "author_popularity_signal",
                        "default_field_value_if_missing": 0
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 1,
                        "boost_function_increasing_function_type": "log1p"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 6,
                        "score_multiplier_variable_boost_weight": 1,
                        "score_multiplier_minimum_value": 0,
                        "score_multiplier_maximum_value": 12
                    }
                }
            ],
            "id_exclude_list": [author_json["id"]],
            # "embedding_fields_to_return": [],  # don't return embeddings to reduce data transfer and latency 
            "limit": 5,
            "start": 0
        })
        recommended_authors_json = similar_authors_resp.json()["results"]
    else:
        recommended_authors_json = []

    print("author_json", author_json)
    print("recommended_authors_json", recommended_authors_json)

    books_scores = [b["score"] for b in books_by_author]
    books_by_author_titles = [b["item"]["text_fields"]["title"].lower() for b in books_by_author]
    tfidf = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        smooth_idf=True
    ).fit_transform(books_by_author_titles)
    cosine_similarities = linear_kernel(tfidf, tfidf)
    cosine_similarities[cosine_similarities < 0] = 0
    cosine_similarities[cosine_similarities > 1] = 1
    idx_list = \
        select_items_greedily_and_with_similarity_suppression(
            items_weights=np.array(books_scores), items_similarity_matrix=np.array(cosine_similarities),
            suppress_supression=True)
    books_by_author = [books_by_author[idx] for idx in idx_list]

    return render_template("author.html",
                           author=author_json, books_by_author=books_by_author,
                           recommended_authors_json=recommended_authors_json)


@app.route('/search_autocomplete')
def search_autocomplete():
    term = request.args.get("term")
    print("term:", term)
    resp = requests.post(f"{opensearchrec_base_url}/search/{index_name}", json={
        "text_matching": [
            {
                "query": term,
                "text_fields_names_and_weights_dict": {
                    "title": 1,
                },
                "use_ngram_fields_if_available": True,
                "required": True,
                "minimum_should_match": 1
            }
        ],
        "numeric_fields_boosting": [
            {
            "input_value_config": {
                "field_name": "book_popularity_signal",
                "default_field_value_if_missing": 0
            },
            "boost_function_config": {
                "boost_function_input_value_rescaling_factor": 1,
                "boost_function_increasing_function_type": "log1p"
            },
            "scoring_factor_config": {
                "score_multiplier_constant_weight": 1,
                "score_multiplier_variable_boost_weight": 0.5,
                "score_multiplier_minimum_value": 0,
                "score_multiplier_maximum_value": 5
            }
            }
        ],
        "limit": 50,
        "start": 0
    })
    print("resp", resp)
    resp_json = resp.json()

    titles = [{
        "label": item["item"]["text_fields"]["title"],
        "value": item["item"]["text_fields"]["title"]
     } for item in resp_json["results"][:]]


    authors_search_request = {
        "text_matching": [
            {
                "query": term,
                "text_fields_names_and_weights_dict": {
                    "author_name": 1,
                },
                "use_ngram_fields_if_available": True,
                "required": True,
                "minimum_should_match": "90%"
            },
            {
                "query": term,
                "text_fields_names_and_weights_dict": {
                    "author_name": 2,
                },
                "use_ngram_fields_if_available": False,
                "required": True,
                "minimum_should_match": "90%"
            }
        ],
        "numeric_fields_filtering": [
            {
            "input_value_config": {
                "field_name": "author_popularity_signal",
                "default_field_value_if_missing": 0
            },
            "minimum_value": 1000,
            }
        ],
        "numeric_fields_boosting": [
            {
                "input_value_config": {
                    "field_name": "author_popularity_signal",
                    "default_field_value_if_missing": 0
                },
                "boost_function_config": {
                    "boost_function_input_value_rescaling_factor": 1,
                    "boost_function_increasing_function_type": "log1p"
                },
                "scoring_factor_config": {
                    "score_multiplier_constant_weight": 6,
                    "score_multiplier_variable_boost_weight": 1,
                    "score_multiplier_minimum_value": 0,
                    "score_multiplier_maximum_value": 12
                }
            }
        ],
        "embedding_fields_to_return": [],  # don't return embeddings to reduce data transfer and latency 
        "limit": 1,
        "start": 0
    }

    resp = requests.post(f"{opensearchrec_base_url}/search/{author_index_name}",
                            json=authors_search_request)
    print("resp", resp)
    print("resp", resp.text[:1000])
    resp_json = resp.json()

    authors_hits = resp_json["results"]
    authors = [
        {
            "label": "Author: " + item["item"]["text_fields"]["author_name"],
            "value": item["item"]["text_fields"]["author_name"]
        } for item in resp_json["results"][:3]
    ]

    print(titles)
    print(authors)

    return json.dumps(authors + titles)


if __name__ == "__main__":
    app.run(port=5555, debug=True)