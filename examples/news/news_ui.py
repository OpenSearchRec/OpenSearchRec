from datetime import datetime, timedelta
from flask import Flask, request, render_template

import rapidfuzz

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from sentence_transformers import SentenceTransformer

from OpenSearchRec.retrieval import (
    ElasticSearchRetrievalClientSettings,
    ElasticSearchRetrievalClient,
    SearchConfig
)

from OpenSearchRec.post_ranking import (
    select_items_greedily_and_with_similarity_suppression,
    select_top_k_centroids
)

index_name = "news2"

embedding_model = SentenceTransformer("all-mpnet-base-v2")

use_rapidfuzz_for_filtering = True

open_search_rec_settings = \
    ElasticSearchRetrievalClientSettings(
        database_type="elasticsearch",
        elasticsearch_host="http://localhost:9200",
        elasticsearch_index_prefix="opensearchrec_index_prefix_",
        elasticsearch_alias_prefix="opensearchrec_alias_prefix_",
        elasticsearch_username="elastic",
        elasticsearch_password="admin",
        elasticsearch_verify_certificates=False
    )

open_search_rec = ElasticSearchRetrievalClient(open_search_rec_settings)


app = Flask(__name__)


def get_articles_similarity_matrix(
        articles_list,
        similarity_multiplier_for_different_articles_from_same_source=0,
        max_days_difference_for_non_zero_similarity=1.5):
    article_texts = [
        a.item["text_fields"]["article_title"] + a.item["text_fields"]["article_text_beginning"]
        for a in articles_list
    ]

    article_dates = [
        a.item["date_fields"]["publish_date"] for a in articles_list
    ]

    text_parsing_args = {
        "strip_accents": "unicode",
        "lowercase": True,
        "ngram_range": (1, 1)
    }
    tfidf_vectorizer = TfidfVectorizer(
        **text_parsing_args,
        smooth_idf=True,
        analyzer="word",
    )

    tf_idf = tfidf_vectorizer.fit_transform(article_texts)

    cosine_similarities = linear_kernel(tf_idf, tf_idf)

    if similarity_multiplier_for_different_articles_from_same_source is not None:
        article_sources = [
            a.item["text_fields"]["article_source_name"]
            for a in articles_list
        ]
        for i in range(len(article_sources)):
            for j in range(i + 1, len(article_sources)):
                if (i != j) and (article_sources[i] == article_sources[j]):
                    cosine_similarities[i, j] *= similarity_multiplier_for_different_articles_from_same_source
                    cosine_similarities[j, i] *= similarity_multiplier_for_different_articles_from_same_source

    if max_days_difference_for_non_zero_similarity is not None:
        for i in range(len(article_sources)):
            for j in range(i+1, len(article_sources)):
                if (i != j):
                    if article_dates[i] is None or article_dates[j] is None:
                        cosine_similarities[i, j] = 0
                        cosine_similarities[j, i] = 0
                    elif abs((article_dates[i] - article_dates[j]).total_seconds()) > timedelta(days=max_days_difference_for_non_zero_similarity).total_seconds():
                        cosine_similarities[i, j] = 0
                        cosine_similarities[j, i] = 0

    cosine_similarities[cosine_similarities < 0] = 0
    cosine_similarities[cosine_similarities > 1] = 1

    return cosine_similarities


@app.route('/')
def home():
    request_json = {
        "date_fields_boosting": [
            {
                "input_value_config": {
                    "field_name": "publish_date",
                    "target_date": str(datetime.utcnow()),
                    "time_units": "hours"
                },
                "boost_function_config": {
                    "decay_boost_type": "exponential",
                    "decay_boost_offset": 24,
                    "decay_boost_decay_rate": 0.5,
                    "decay_boost_decay_scale": 12
                },
                "scoring_factor_config": {
                    "score_multiplier_constant_weight": 0,
                    "score_multiplier_variable_boost_weight": 1,
                    "score_multiplier_minimum_value": 0,
                    "score_multiplier_maximum_value": 2
                }
            }
        ],
        "embedding_fields_to_return": [],  # don't return embeddings to reduce data transfer and latency 
        "limit": 1000,
        "start": 0
    }

    search_response = open_search_rec.search(index_name, SearchConfig.parse_obj(request_json))
    hits = search_response.results

    if len(hits) > 0:
        articles_scores = [hit.score for hit in hits]
        article_similarity_matrix = get_articles_similarity_matrix(hits)
        centroids_results = \
            select_top_k_centroids(
                num_centroids=50,
                items_similarity_matrix=np.array(article_similarity_matrix),
                items_weights=np.array(articles_scores))
        centroid_idx_list = centroids_results["sorted_centroid_idx_list"]
        centroid_scores_list = centroids_results["sorted_centroid_scores_list"]
        hits = [hits[idx] for idx in centroid_idx_list]
        for idx, hit in enumerate(hits):
            hits[idx] = hit.item.json_serializable_dict()
            hits[idx]["centroid_score"] = centroid_scores_list[idx]

    return render_template(
        "home.html", hits=hits[:100]
    )


@app.route('/search')
def search():
    q = request.args.get("q", "")[:1000]  # max query length
    search_type = request.args.get("search_type", "relevance")
    ordering = request.args.get("ordering", "relevance_sort")
    min_date = request.args.get("min_date", "")
    max_date = request.args.get("max_date", "")
    target_date_boost = request.args.get("target_date_boost", "")

    if len(q) > 0:

        sentence_embedding = embedding_model.encode(q).tolist()

        request_json = {
            "date_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "publish_date",
                        "target_date": target_date_boost + " 00:00:00" if len(target_date_boost) > 0 else str(datetime.utcnow()),
                        "time_units": "hours"
                    },
                    "boost_function_config": {
                        "decay_boost_type": "exponential",
                        "decay_boost_offset": 24 if len(target_date_boost) > 0 else 24,
                        "decay_boost_decay_rate": 0.5,
                        "decay_boost_decay_scale": 12
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 1 if (len(q) > 0 and search_type != "centroids") else 0,
                        "score_multiplier_variable_boost_weight": 1,
                        "score_multiplier_minimum_value": 0,
                        "score_multiplier_maximum_value": 2
                    }
                }
            ],
            "embedding_fields_to_return": [],  # don't return embeddings to reduce data transfer and latency 
            "limit": 1000,
            "start": 0
        }

        request_json["text_matching"] = [
            {
                "query": q,
                "text_fields_names_and_weights_dict": {
                    "article_title": 1.5,
                    "article_text_beginning": 0.5,
                    "article_text": 0.1,
                    "article_source_name": 1,
                    "article_authors": 1
                },
                "use_ngram_fields_if_available": True,
                "required": False,
                "minimum_should_match": 1
            },
            {
                "query": q,
                "text_fields_names_and_weights_dict": {
                    "article_title": 1.5,
                    "article_text_beginning": 0.5,
                    "article_source_name": 1,
                    "article_authors": 1
                },
                "use_ngram_fields_if_available": False,
                "required": False,
                "minimum_should_match": 1
            },
        ]

        request_json["embedding_fields_boosting"] = [
            {
                "input_value_config": {
                    "field_name": "title_embedding",
                    "target_embedding": sentence_embedding,
                    "default_embedding_comparison_value_if_missing": 0,
                    "embedding_comparison_metric": "cosine_similarity"
                },
                "boost_function_config": {
                    "boost_function_input_value_rescaling_factor": 1,
                    "boost_function_increasing_function_type": "none"
                },
                "scoring_factor_config": {
                    "score_multiplier_constant_weight": 3,
                    "score_multiplier_variable_boost_weight": 1,
                    "score_multiplier_minimum_value": 0,
                    "score_multiplier_maximum_value": 5
                }
            }
        ]

        if min_date != "" or max_date != "":
            date_filter = {
                "input_value_config": {
                    "field_name": "publish_date",
                }
            }
            if min_date != "":
                date_filter["minimum_value"] = min_date + " 00:00:00"
            if max_date != "":
                date_filter["maximum_value"] = max_date + " 00:00:00"
            request_json["date_fields_filtering"] = [date_filter]

        search_response = open_search_rec.search(index_name, SearchConfig.parse_obj(request_json))
        hits = search_response.results

    else:
        hits = []

    if use_rapidfuzz_for_filtering and len(hits) > 0 and len(q) > 0:
        hit_texts = [
            hit.item.get("text_fields", {}).get("article_title", "") + hit.item.get("text_fields", {}).get("article_text_beginning", "")
            for hit in hits
        ]

        text_match_best_token_scores = []
        text_match_average_token_scores = []
        for text in hit_texts:
            text_tokens = [rapidfuzz.utils.default_process(text_token) for text_token in text.split(" ")]
            query_tokens = [rapidfuzz.utils.default_process(query_token) for query_token in q.split(" ")]
            scores_per_query_token = []
            for query_token in query_tokens:
                best_score = 0
                for text_token in text_tokens:
                    score = rapidfuzz.distance.JaroWinkler.similarity(query_token, text_token)
                    best_score = max(best_score, score)
                scores_per_query_token.append(best_score)

            text_match_best_token_scores.append(max(scores_per_query_token))
            text_match_average_token_scores.append(np.mean(scores_per_query_token))

        valid_idxs = [idx for idx, score in enumerate(text_match_best_token_scores) if score > 0.8]

        filtered_hits = [
            hit for idx, hit in enumerate(hits) if idx in valid_idxs
        ]

        hits = filtered_hits

    if len(hits) > 0:
        if search_type == "relevance_extra_spread":
            articles_scores = [hit.score for hit in hits]
            article_similarity_matrix = get_articles_similarity_matrix(hits)
            idx_list = \
                select_items_greedily_and_with_similarity_suppression(
                    items_weights=np.array(articles_scores), items_similarity_matrix=np.array(article_similarity_matrix),
                    suppress_supression=True)
            hits = [hits[idx] for idx in idx_list]
        elif search_type == "centroids":
            articles_scores = [hit.score for hit in hits]
            article_similarity_matrix = get_articles_similarity_matrix(hits)
            centroids_results = \
                select_top_k_centroids(
                    num_centroids=20,
                    items_similarity_matrix=np.array(article_similarity_matrix),
                    items_weights=np.array(articles_scores))
            centroid_idx_list = centroids_results["sorted_centroid_idx_list"]
            hits = [hits[idx] for idx in centroid_idx_list]

    search_type_relevance_checked = "checked" if search_type == "relevance" else ""
    search_type_relevance_spread_checked = "checked" if search_type == "relevance_extra_spread" else ""
    search_type_centroid_checked = "checked" if search_type == "centroids" else ""

    relevance_sort_checked = "checked" if ordering == "relevance_sort" else ""
    chronological_sort_checked = "checked" if ordering == "chronological_sort" else ""

    if ordering == "chronological_sort":
        hits = sorted(hits, reverse=True, key=lambda hit: hit["item"]["date_fields"]["publish_date"])

    return render_template(
        "search.html", q=q, hits=hits[:100],
        min_date=min_date, max_date=max_date,
        target_date_boost=target_date_boost,
        search_type_relevance_checked=search_type_relevance_checked,
        search_type_relevance_spread_checked=search_type_relevance_spread_checked,
        search_type_centroid_checked=search_type_centroid_checked,
        relevance_sort_checked=relevance_sort_checked,
        chronological_sort_checked=chronological_sort_checked,
    )


if __name__ == "__main__":
    app.run(port=5000, debug=True)
