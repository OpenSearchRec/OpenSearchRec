from sentence_transformers import SentenceTransformer


from news_utils import (
    get_page_url_from_domain,
    get_article_dict
)

from OpenSearchRec.retrieval import (
    ElasticSearchRetrievalClientSettings,
    ElasticSearchRetrievalClientAsync,
    ElasticSearchRetrievalClient,
    TextMatchingType,
    IndexConfig,
    TextMatchingConfig,
    TextMatchingConfigSettings,
    EmbeddingConfig,
    EmbeddingComparisonMetric,
    SearchItem
)


def load_current_articles(news_sources_dict,
                          open_search_rec,
                          embedding_model_path="all-mpnet-base-v2",
                          memoize_articles=False,
                          delete_index_if_exits=False):

    index_name = "news2"
    index_config = IndexConfig(
        text_matching_config={
            "text_matching_type": "bm25",
            "settings": {
                "bm25_k1": 0.25,  # small boost for when a term matches multiple time in text
                "bm25_b": 0.0,  # no penalization of longer text
                "enable_ngram_tokenizer": True,
                "analyzer": "english",
                "ngram_tokenizer_min_gram": 2,
                "ngram_tokenizer_max_gram": 4
            }
        },
        text_fields=[
            "article_title",
            "article_text",
            "article_text_beginning",
            "article_source_name",
            "article_authors"
        ],
        categorical_fields=[
            "article_authors"
        ],
        date_fields=[
            "publish_date"
        ],
        embedding_fields={
            "title_embedding": EmbeddingConfig(
                embedding_dimension=768,
                enable_approximate_nearest_embedding_search=True,
                approximate_nearest_embedding_search_metric=EmbeddingComparisonMetric.cosine_similarity
            )
        }
    )

    model = SentenceTransformer(embedding_model_path)

    if delete_index_if_exits:
        try:
            result = open_search_rec.delete_index(index_name)  # throws error if index doesn't exist
            print(result)
        except Exception as e:
            print("delete index error:", e)

        result = open_search_rec.create_index(index_name, index_config)  # throws error if index exists already
        print(result)

    else:
        try:
            result = open_search_rec.create_index(index_name, index_config)  # throws error if index exists already
        except Exception as e:
            print(e)

    for news_source_name, source_domain in news_sources_dict.items():
        print()
        print(source_domain)
        domain_urls = get_page_url_from_domain(source_domain, memoize_articles=memoize_articles)
        print("num_urls:", len(domain_urls))
        for article_url in domain_urls[:]:
            print()
            try:
                article_dict = get_article_dict(article_url)
            except Exception as e:
                print(e)
                print(article_url)
                article_dict = None

            if article_dict is not None:
                try:
                    # print(article_dict["normalized_url"])
                    article = \
                        SearchItem(
                            id=article_dict["normalized_url"].replace("/", "_").replace(".", "_"),
                            text_fields={
                                "article_title": article_dict["title"],
                                "article_text": article_dict["text"],
                                "article_text_beginning": " ".join(article_dict["text"].split(" ")[:100]), # first ~100 words
                                "article_source_name": news_source_name,
                                "article_authors": ", ".join(article_dict["authors"])
                            },
                            date_fields={
                                "publish_date": article_dict["publish_date"],
                            },
                            categorical_fields={
                                "article_authors": article_dict["authors"],
                                "article_source_name": news_source_name
                            },
                            embedding_fields={
                                "title_embedding": model.encode(article_dict["title"]).tolist(),
                            },
                            extra_information={
                                "top_image": article_dict["top_image"],
                                "movies": article_dict["movies"],
                                "article_url": article_url
                            }
                        )

                    # print(article)

                    r = open_search_rec.index_item(index_name, article)
                    print(r)
                except Exception as e:
                    print("Exception:")
                    print(e)


if __name__ == "__main__":
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
    print(open_search_rec_settings)

    open_search_rec = ElasticSearchRetrievalClient(open_search_rec_settings)
    print(open_search_rec)

    news_sources = {
        "New York Times": "https://www.nytimes.com",
        "New York Post": "https://www.nypost.com",
        "CNN": "https://www.cnn.com",
        "Fox News": "https://www.foxnews.com",
        "BBC": "https://www.bbc.com",
        "The Washington Post": "https://www.washingtonpost.com/",
        "The Hill": "https://thehill.com/",
        "Newsweek": "https://www.newsweek.com/",
        "Market Watch": "https://www.marketwatch.com/",
        "Politico": "https://www.politico.com/",
        "Washington Examiner": "https://www.washingtonexaminer.com/",
        "TechCrunch": "https://techcrunch.com/",
        "CNET": "https://www.cnet.com/",
        "Axios": "https://www.axios.com/",
        "Ars Technica": "https://arstechnica.com",
        "Wired": "https://www.wired.com/",
        "Engadget": "https://www.engadget.com/",
    }
    print(news_sources)

    load_current_articles(news_sources, open_search_rec, delete_index_if_exits=True)
