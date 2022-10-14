from datetime import datetime
import math
import pytest
import time

from OpenSearchRec.retrieval import (
    ElasticSearchRetrievalClientSettings,
    ElasticSearchRetrievalClientAsync,
    ElasticSearchRetrievalClient,
    IndexConfig,
    SearchItem,
    SearchConfig,
    SearchResults
)

opensearch_settings = \
    ElasticSearchRetrievalClientSettings(
        database_type="opensearch",
        elasticsearch_host="https://localhost:9200",
        elasticsearch_index_prefix="opensearchrec_index_prefix_",
        elasticsearch_alias_prefix="opensearchrec_alias_prefix_",
        elasticsearch_username="admin",
        elasticsearch_password="admin",
        elasticsearch_verify_certificates=False
    )

elasticsearch_settings = \
    ElasticSearchRetrievalClientSettings(
        database_type="elasticsearch",
        elasticsearch_host="http://localhost:9201",
        elasticsearch_index_prefix="opensearchrec_index_prefix_",
        elasticsearch_alias_prefix="opensearchrec_alias_prefix_",
        elasticsearch_username="elastic",
        elasticsearch_password="admin",
        elasticsearch_verify_certificates=False
    )


pytest_index_name = "pytest_index"
pytest_alias_name = "pytest_alias"


def setup_index(sync_client, index_name):
    indices = sync_client.list_all_indexes()
    if pytest_index_name in indices:
        delete_index = sync_client.delete_index(pytest_index_name)
        print("delete_index", delete_index)

    indices = sync_client.list_all_indexes()
    assert pytest_index_name not in indices

    index_config_json = {
        "text_matching_config": {
            "text_matching_type": "bm25",
            "settings": {
                "bm25_k1": 1.2,
                "bm25_b": 0.75,
                "enable_ngram_tokenizer": True,
                "ngram_tokenizer_min_gram": 2,
                "ngram_tokenizer_max_gram": 4
            }
        },
        "text_fields": [
            "title",
            "description",
            "source_name"
        ],
        "categorical_fields": [
            "tags",
            "authors"
        ],
        "numeric_fields": [
            "popularity",
            "quality_signal"
        ],
        "date_fields": [
            "published_date",
            "last_updated_date"
        ],
        "geolocation_fields": [
            "location"
        ],
        "embedding_fields": {
            "embedding1_name": {
                "embedding_dimension": 5,
                "enable_approximate_nearest_embedding_search": True,
                "approximate_nearest_embedding_search_metric": "cosine_similarity"
            },
            "embedding2_name": {
                "embedding_dimension": 20,
                "enable_approximate_nearest_embedding_search": True
            },
            "embedding3_name": {
                "embedding_dimension": 768,
                "enable_approximate_nearest_embedding_search": True,
                "approximate_nearest_embedding_search_metric": "cosine_similarity"
            },
            "all_zeros_embedding_field": {
                "embedding_dimension": 2,
                "enable_approximate_nearest_embedding_search": True,
                "approximate_nearest_embedding_search_metric": "l2_norm"
            }
        },
        "number_of_shards": 1,
        "number_of_replicas": 1
    }

    create_index = sync_client.create_index(pytest_index_name, IndexConfig(**index_config_json))
    assert type(create_index) == dict
    print("create_index", create_index)

    indices = sync_client.list_all_indexes()
    print(indices)
    assert pytest_index_name in indices
    assert type(indices) == list


def index_and_update_items(sync_client, index_name):
    item1 = {
        "id": "1",
        "text_fields": {
            "title": "title",
            "description": "description",
            "source_name": "source_name"
        },
        "categorical_fields": {
            "tags": [
                "tag1",
                "tag2",
                "tag3"
            ],
            "authors": "author1"
        },
        "numeric_fields": {
            "popularity": 1000000,
            "quality_signal": 4.9
        },
        "date_fields": {
            "published_date": "2020-01-01 01:01:01",
            "last_updated_date": "2021-01-01 01:01:01"
        },
        "geolocation_fields": {
            "location": {
                "latitude": 0,
                "longitude": 0
            }
        },
        "embedding_fields": {
            "embedding1_name": [
                1,
                2,
                3,
                4,
                5
            ],
            "embedding2_name": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20
            ],
            "all_zeros_embedding_field": [0, 0]
        },
        "extra_information": {
            "image_urls": [
                "http://localhost:80/image1.pg",
                "http://localhost:80/image2.pg"
            ]
        }
    }
    index_item = sync_client.index_item(pytest_index_name, SearchItem.parse_obj(item1), refresh="true")

    item2 = {
        "id": "2",
        "text_fields": {
            "title": "item 2 title",
            "description": "description",
            "source_name": "source_name"
        },
        "categorical_fields": {
            "tags": [
                "tag1",
                "tag2",
                "tag3"
            ],
            "authors": ["author1", "author2"]
        },
        "numeric_fields": {
            "popularity": 1000000,
            "quality_signal": 4.9
        },
        "date_fields": {
            "published_date": "2021-09-04T19:50:42",
            "last_updated_date": datetime.utcnow()
        },
        "geolocation_fields": {
            "location": {
                "latitude": 0,
                "longitude": 0
            }
        },
        "embedding_fields": {
            "embedding1_name": [1, 1, 1, 1, 1],
        },
        "extra_information": {
            "image_urls": [
                "http://localhost:80/image1.pg",
                "http://localhost:80/image2.pg"
            ]
        }
    }
    index_item = sync_client.index_item(pytest_index_name, SearchItem.parse_obj(item2), refresh="true")
    assert type(index_item) == dict
    delete_item = sync_client.delete_item(pytest_index_name, "2", refresh="true")
    assert type(delete_item) == dict
    index_item = sync_client.index_item(pytest_index_name, SearchItem.parse_obj(item2), refresh="true")
    assert type(index_item) == dict

    item3 = {
        # "id": "3",
        "text_fields": {
            "title": "item with autogenerated id title",
            "description": "description",
            "source_name": "source_name"
        },
        "categorical_fields": {
            "tags": [
                "tag4",
                "tag5",
                "tag6"
            ],
            "authors": ["author3", "author4"]
        },
        "numeric_fields": {
            "popularity": 1000000,
            "quality_signal": 4.9
        },
        "date_fields": {
            "published_date": datetime.utcnow(),
            "last_updated_date": datetime.utcnow()
        },
        "geolocation_fields": {
            "location": {
                "latitude": 0,
                "longitude": 0
            }
        },
        "embedding_fields": {
            "embedding1_name": [-1, -1, -1, -1, -1],
        },
        "extra_information": {
            "image_urls": [
                "http://localhost:80/image1.pg",
                "http://localhost:80/image2.pg"
            ]
        }
    }
    index_item = sync_client.index_item(pytest_index_name, SearchItem.parse_obj(item3), refresh="true")
    assert type(index_item) == dict

    retrieved_item_1 = sync_client.get_item(pytest_index_name, "1")
    retrieved_item_1_dict = retrieved_item_1.json_serializable_dict()
    print("retrieved_item_1", retrieved_item_1)
    print("retrieved_item_1_dict", retrieved_item_1_dict)
    assert retrieved_item_1_dict["text_fields"] == item1["text_fields"], retrieved_item_1_dict["text_fields"]
    assert retrieved_item_1_dict["categorical_fields"] == item1["categorical_fields"], retrieved_item_1_dict["categorical_fields"]
    print('retrieved_item_1_dict["date_fields"]', retrieved_item_1_dict["date_fields"])
    print('item1["date_fields"]', item1["date_fields"])
    error_str = f"{retrieved_item_1_dict['date_fields']} vs {item1['date_fields']}"
    assert str(retrieved_item_1_dict["date_fields"]["last_updated_date"]) == item1["date_fields"]["last_updated_date"], error_str
    assert retrieved_item_1_dict["date_fields"]["published_date"] == item1["date_fields"]["published_date"], error_str
    assert retrieved_item_1_dict["date_fields"] == item1["date_fields"], error_str
    print("retrieved_item_1_dict", retrieved_item_1_dict["date_fields"])
    print("item1", item1["date_fields"])
    assert retrieved_item_1_dict["categorical_fields"] == item1["categorical_fields"], error_str
    assert retrieved_item_1_dict["numeric_fields"] == item1["numeric_fields"], error_str
    assert retrieved_item_1_dict["date_fields"] == item1["date_fields"], error_str
    assert retrieved_item_1_dict["geolocation_fields"] == item1["geolocation_fields"], error_str
    assert retrieved_item_1_dict["text_fields"] == item1["text_fields"], error_str
    assert retrieved_item_1_dict["extra_information"] == item1["extra_information"], error_str
    assert retrieved_item_1_dict == item1

    item1["numeric_fields"]["popularity"] = 2000000
    update_resp = \
        sync_client.update_item(
            pytest_index_name,
            SearchItem.parse_obj({
                "id": "1",
                "numeric_fields": {
                    "popularity": 2000000
                }
            }),
            refresh="true")
    assert type(update_resp) == dict
    retrieved_item_1 = sync_client.get_item(pytest_index_name, "1")
    retrieved_item_1_dict = retrieved_item_1.json_serializable_dict()
    assert retrieved_item_1_dict == item1, retrieved_item_1_dict

    # assert False, update_resp


def filtering_testing(sync_client, index_name):
    # Categorical variable filtering
    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "categorical_matching": [
                {
                    "categorical_field_name": "authors",
                    "values_list": [
                        "author1"
                    ],
                    "score_multiplier": 2,
                    "minimum_should_match": 1,
                    "required": True
                }
            ]
        }))
    assert type(search_response) == SearchResults
    assert search_response.results[0].id == "1"
    assert search_response.results[0].score == 2

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "categorical_matching": [
                {
                    "categorical_field_name": "authors",
                    "values_list": [
                        "author2"
                    ],
                    "score_multiplier": 0,
                    "minimum_should_match": 1,
                    "required": True
                }
            ]
        }))
    assert type(search_response) == SearchResults
    assert len(search_response.results) == 0

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "categorical_matching": [
                {
                    "categorical_field_name": "authors",
                    "values_list": [
                        "author2"
                    ],
                    "score_multiplier": 0,
                    "minimum_should_match": 1,
                    "required": False
                }
            ]
        }))
    assert type(search_response) == SearchResults
    assert search_response.results[0].id == "1"
    assert search_response.results[0].score == 1

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "categorical_matching": [
                {
                    "categorical_field_name": "tags",
                    "values_list": [
                        "tag1"
                    ],
                    "score_multiplier": 0,
                    "minimum_should_match": 1,
                    "required": True
                }
            ]
        }))
    assert type(search_response) == SearchResults
    assert search_response.results[0].id == "1"

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "categorical_matching": [
                {
                    "categorical_field_name": "tags",
                    "values_list": [
                        "tag1",
                        "tag2"
                    ],
                    "score_multiplier": 0,
                    "minimum_should_match": 2,
                    "required": True
                }
            ]
        }))
    assert type(search_response) == SearchResults
    assert search_response.results[0].id == "1"

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "categorical_matching": [
                {
                    "categorical_field_name": "tags",
                    "values_list": [
                        "tag0",
                        "tag2"
                    ],
                    "score_multiplier": 0,
                    "minimum_should_match": 2,
                    "required": True
                }
            ]
        }))
    assert type(search_response) == SearchResults
    assert len(search_response.results) == 0

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "categorical_matching": [
                {
                    "categorical_field_name": "tags",
                    "values_list": [
                        "tag0",
                        "tag2"
                    ],
                    "score_multiplier": 0,
                    "minimum_should_match": 1,
                    "required": True
                }
            ]
        }))
    assert type(search_response) == SearchResults
    assert search_response.results[0].id == "1"


    # Numeric Filtering
    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_filtering": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0
                    },
                    "minimum_value": 2000000,
                    "maximum_value": 2000000,
                    "strictly_greater_than": 0,
                    "strictly_less_than": 2000001
                }
            ]
        }))
    assert type(search_response) == SearchResults
    assert search_response.results[0].id == "1"

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_filtering": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0
                    },
                    "minimum_value": 0,
                    "maximum_value": 2000000,
                    "strictly_greater_than": 0,
                    "strictly_less_than": 2000000
                }
            ]
        }))
    assert type(search_response) == SearchResults
    assert len(search_response.results) == 0

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_filtering": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0
                    },
                    "minimum_value": 2000001,
                    "maximum_value": 2000001,
                    "strictly_greater_than": 0,
                    "strictly_less_than": 2000001
                }
            ]
        }))
    assert type(search_response) == SearchResults
    assert len(search_response.results) == 0

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_filtering": [
                {
                    "input_value_config": {
                        "field_name": "popularity___",
                        "default_field_value_if_missing": 0
                    },
                    "minimum_value": 2000000,
                }
            ]
        }))
    assert type(search_response) == SearchResults
    assert len(search_response.results) == 0

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_filtering": [
                {
                    "input_value_config": {
                        "field_name": "popularity___",
                        "default_field_value_if_missing": 2000000
                    },
                    "minimum_value": 2000000,
                }
            ]
        }))
    assert type(search_response) == SearchResults
    assert search_response.results[0].id == "1"

    # Date Filtering
    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "date_fields_filtering": [
                {
                    "input_value_config": {
                        "field_name": "published_date",
                        "default_field_value_if_missing": "2020-01-01 01:01:01"
                    },
                    "minimum_value": "2020-01-01 01:01:00Z",
                    "maximum_value": "2020-01-01 01:01:02Z",
                    "strictly_greater_than": "2020-01-01 01:01:00Z",
                    "strictly_less_than": "2022-01-01 01:01:02Z"
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "date_fields_filtering": [
                {
                    "input_value_config": {
                        "field_name": "published_date_",
                        "default_field_value_if_missing": "2020-01-01 01:01:01Z"
                    },
                    "minimum_value": "2020-01-01 01:01:00Z",
                    "maximum_value": "2020-01-01 01:01:02Z",
                    "strictly_greater_than": "2020-01-01 01:01:00Z",
                    "strictly_less_than": "2022-01-01 01:01:02Z"
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "date_fields_filtering": [
                {
                    "input_value_config": {
                        "field_name": "published_date",
                        "default_field_value_if_missing": "2022-09-17T03:16:34.296Z"
                    },
                    "minimum_value": "2020-01-01 01:01:01Z",
                    "maximum_value": "2020-01-01 01:01:01Z",
                    "strictly_greater_than": "2020-01-01 01:01:00Z",
                    "strictly_less_than": "2020-01-01 01:01:00Z"
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print(search_response)
    assert len(search_response.results) == 0

    # Geo Field Filtering

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "geolocation_fields_filtering": [
                {
                    "input_value_config": {
                        "field_name": "location",
                        "target_geolocation": {
                            "latitude": 40.729662,
                            "longitude": -73.999414
                        },
                        "default_distance_if_missing": 0,
                        "distance_unit": "kilometers"
                    },
                    "minimum_value": 8000,
                    "maximum_value": 9000,
                    "strictly_greater_than": 8000,
                    "strictly_less_than": 9000
                }
            ],

        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    assert search_response.results[0].score == 1


    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "geolocation_fields_filtering": [
                {
                    "input_value_config": {
                        "field_name": "location",
                        "target_geolocation": {
                            "latitude": 40.729662,
                            "longitude": -73.999414
                        },
                        "default_distance_if_missing": 0,
                        "distance_unit": "kilometers"
                    },
                    "minimum_value": 8000,
                    "maximum_value": 8500,
                    "strictly_greater_than": 8000,
                    "strictly_less_than": 9000
                }
            ],

        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert len(search_response.results) == 0

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "geolocation_fields_filtering": [
                {
                    "input_value_config": {
                        "field_name": "location_",
                        "target_geolocation": {
                            "latitude": 40.729662,
                            "longitude": -73.999414
                        },
                        "default_distance_if_missing": 0,
                        "distance_unit": "kilometers"
                    },
                    "minimum_value": 8000,
                    "maximum_value": 9000,
                    "strictly_greater_than": 8000,
                    "strictly_less_than": 9000
                }
            ],

        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert len(search_response.results) == 0


def search_testing(sync_client, index_name):
    search_response = sync_client.search(index_name, SearchConfig.parse_obj({}))
    assert type(search_response) == SearchResults

    # Numeric Scoring
    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 2,
                        "boost_function_increasing_function_type": "log1p"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 5,
                        "score_multiplier_variable_boost_weight": 1,
                        "score_multiplier_minimum_value": 0,
                        "score_multiplier_maximum_value": 100
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    popularity_value = 2000000
    expected_score = 5 + math.log(1 + 2 * popularity_value, 10)
    assert abs(search_response.results[0].score - expected_score) < 0.00001

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 2,
                        "boost_function_increasing_function_type": "log2p"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 5,
                        "score_multiplier_variable_boost_weight": 1,
                        "score_multiplier_minimum_value": 0,
                        "score_multiplier_maximum_value": 100
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    popularity_value = 2000000
    expected_score = 5 + math.log(2 + 2 * popularity_value, 10)
    assert abs(search_response.results[0].score - expected_score) < 0.00001

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 2,
                        "boost_function_increasing_function_type": "log1p"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 3,
                        "score_multiplier_variable_boost_weight": 1,
                        "score_multiplier_minimum_value": 0,
                        "score_multiplier_maximum_value": 5
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 5
    assert abs(search_response.results[0].score - expected_score) < 0.00001

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity_",
                        "default_field_value_if_missing": 0
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 2,
                        "boost_function_increasing_function_type": "log1p"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 3,
                        "score_multiplier_variable_boost_weight": 1,
                        "score_multiplier_minimum_value": 0,
                        "score_multiplier_maximum_value": 5
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 3
    assert abs(search_response.results[0].score - expected_score) < 0.00001

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity_",
                        "default_field_value_if_missing": 0
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 2,
                        "boost_function_increasing_function_type": "log1p"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1,
                        "score_multiplier_minimum_value": 0.5,
                        "score_multiplier_maximum_value": 5
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 0.5
    assert abs(search_response.results[0].score - expected_score) < 0.00001

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 1,
                        "boost_function_increasing_function_type": "none"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1,
                        "score_multiplier_minimum_value": 0,
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    popularity_value = 2000000
    assert abs(search_response.results[0].score - popularity_value) < 0.00001

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 0.001,
                        "boost_function_increasing_function_type": "square"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1,
                        "score_multiplier_minimum_value": 0,
                        # "score_multiplier_maximum_value": 5
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    popularity_value = 2000000
    assert abs(search_response.results[0].score - (0.001 * popularity_value)**2) < 0.00001

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 1,
                        "boost_function_increasing_function_type": "sqrt"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1,
                        "score_multiplier_minimum_value": 0,
                        # "score_multiplier_maximum_value": 5
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    popularity_value = 2000000
    assert abs(search_response.results[0].score - popularity_value**0.5) < 0.0001

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0,
                        "target_value": 2000000,
                    },
                    "boost_function_config": {
                        "decay_boost_type": "gaussian",
                        "decay_boost_offset": 100,
                        "decay_boost_decay_rate": 0.25,
                        "decay_boost_decay_scale": 1000,
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    popularity_value = 2000000
    expected_score = 1
    assert abs(search_response.results[0].score - expected_score) < 0.00001

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0,
                        "target_value": 2000100,
                    },
                    "boost_function_config": {
                        "decay_boost_type": "gaussian",
                        "decay_boost_offset": 100,
                        "decay_boost_decay_rate": 0.25,
                        "decay_boost_decay_scale": 1000,
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    popularity_value = 2000000
    expected_score = 1
    assert abs(search_response.results[0].score - expected_score) < 0.00001

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0,
                        "target_value": 2001000,
                    },
                    "boost_function_config": {
                        "decay_boost_type": "exponential",
                        "decay_boost_offset": 0,
                        "decay_boost_decay_rate": 0.25,
                        "decay_boost_decay_scale": 1000,
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    popularity_value = 2000000
    expected_score = 0.25
    assert abs(search_response.results[0].score - expected_score) < 0.00001

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0,
                        "target_value": 2000000 - 1100,
                    },
                    "boost_function_config": {
                        "decay_boost_type": "linear",
                        "decay_boost_offset": 100,
                        "decay_boost_decay_rate": 0.75,
                        "decay_boost_decay_scale": 1000,
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    popularity_value = 2000000
    expected_score = 0.75
    assert abs(search_response.results[0].score - expected_score) < 0.00001

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity_",
                        "default_field_value_if_missing": 2000000,
                        "target_value": 2000000 - 1100,
                    },
                    "boost_function_config": {
                        "decay_boost_type": "linear",
                        "decay_boost_offset": 100,
                        "decay_boost_decay_rate": 0.75,
                        "decay_boost_decay_scale": 1000,
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 0.75
    assert abs(search_response.results[0].score - expected_score) < 0.00001

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0,
                        "target_value": 2000000 - 1100 + 1,
                    },
                    "boost_function_config": {
                        "decay_boost_type": "exponential",
                        "decay_boost_offset": 100,
                        "decay_boost_decay_rate": 0.75,
                        "decay_boost_decay_scale": 1000,
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    popularity_value = 2000000
    expected_score = 0.75
    assert search_response.results[0].score > expected_score

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0
                    },
                    "boost_function_config": {
                        "minimum_value": 2000000,
                        "maximum_value": 2000000,
                        "strictly_greater_than": 2000000 - 1,
                        "strictly_less_than": 2000000 + 1,
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 1
    assert search_response.results[0].score == expected_score

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0
                    },
                    "boost_function_config": {
                        "minimum_value": 2000000,
                        "maximum_value": 2000000,
                        "strictly_greater_than": 2000000 - 1,
                        "strictly_less_than": 2000000,
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 0
    assert search_response.results[0].score == expected_score

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0
                    },
                    "boost_function_config": {
                        "minimum_value": 2000000,
                        "maximum_value": 2000000,
                        "strictly_greater_than": 2000000,
                        "strictly_less_than": 2000000 + 1,
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 0
    assert search_response.results[0].score == expected_score

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0
                    },
                    "boost_function_config": {
                        "minimum_value": 2000000 + 1,
                        "maximum_value": 2000000,
                        "strictly_greater_than": 2000000 - 1,
                        "strictly_less_than": 2000000 + 1,
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 0
    assert search_response.results[0].score == expected_score

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0
                    },
                    "boost_function_config": {
                        "minimum_value": 2000000,
                        "maximum_value": 2000000 - 1,
                        "strictly_greater_than": 2000000 - 1,
                        "strictly_less_than": 2000000 + 1,
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 0
    assert search_response.results[0].score == expected_score

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity__",
                        "default_field_value_if_missing": 2000000
                    },
                    "boost_function_config": {
                        "minimum_value": 2000000,
                        "maximum_value": 2000000,
                        "strictly_greater_than": 2000000 - 1,
                        "strictly_less_than": 2000000 + 1,
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 1
    assert search_response.results[0].score == expected_score

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity__",
                        "default_field_value_if_missing": 2000000 + 1
                    },
                    "boost_function_config": {
                        "minimum_value": 2000000,
                        "maximum_value": 2000000,
                        "strictly_greater_than": 2000000 - 1,
                        "strictly_less_than": 2000000 + 1,
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 0
    assert search_response.results[0].score == expected_score

    # Date Scoring
    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "date_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "last_updated_date",
                        "default_field_value_if_missing": "2021-01-01 01:01:01Z",
                        "target_date": "2021-01-01 01:01:01Z",
                        "time_units": "hours"
                    },
                    "boost_function_config": {
                        "decay_boost_type": "linear",
                        "decay_boost_offset": 6,
                        "decay_boost_decay_rate": 0.5,
                        "decay_boost_decay_scale": 6,
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
    }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 1
    assert search_response.results[0].score == expected_score

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "date_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "last_updated_date",
                        "default_field_value_if_missing": "2021-01-01 01:01:01Z",
                        "target_date": "2021-01-01 7:01:01Z",
                        "time_units": "hours"
                    },
                    "boost_function_config": {
                        "decay_boost_type": "linear",
                        "decay_boost_offset": 6,
                        "decay_boost_decay_rate": 0.5,
                        "decay_boost_decay_scale": 6,
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
    }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 1
    assert search_response.results[0].score == expected_score

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "date_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "last_updated_date",
                        "default_field_value_if_missing": "2021-01-01 13:01:01Z",
                        "target_date": "2021-01-01 13:01:01Z",
                        "time_units": "hours"
                    },
                    "boost_function_config": {
                        "decay_boost_type": "linear",
                        "decay_boost_offset": 6,
                        "decay_boost_decay_rate": 0.5,
                        "decay_boost_decay_scale": 6,
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
    }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 0.5
    assert search_response.results[0].score == expected_score

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "date_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "last_updated_date",
                        "default_field_value_if_missing": "2021-01-02 00:01:01Z",
                        "target_date": "2021-01-02 00:01:01Z",
                        "time_units": "hours"
                    },
                    "boost_function_config": {
                        "decay_boost_type": "linear",
                        "decay_boost_offset": 6,
                        "decay_boost_decay_rate": 0.5,
                        "decay_boost_decay_scale": 6,
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
    }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 0
    assert search_response.results[0].score == expected_score

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "date_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "last_updated_date___",
                        "default_field_value_if_missing": "2021-01-01 13:01:01Z",
                        "target_date": "2021-01-01 01:01:01Z",
                        "time_units": "hours"
                    },
                    "boost_function_config": {
                        "decay_boost_type": "linear",
                        "decay_boost_offset": 6,
                        "decay_boost_decay_rate": 0.5,
                        "decay_boost_decay_scale": 6,
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
    }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 0.5
    assert search_response.results[0].score == expected_score

    # Embedding Scoring
    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "embedding_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "embedding1_name",
                        "target_embedding": [1, 2, 3, 4, 5],
                        "default_embedding_comparison_value_if_missing": 0,
                        "embedding_comparison_metric": "cosine_similarity"
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 1,
                        "boost_function_increasing_function_type": "none"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
    }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 2
    assert search_response.results[0].score == expected_score

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "embedding_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "embedding1_name",
                        "target_embedding": [0, 0, 0, 0, 0],
                        "default_embedding_comparison_value_if_missing": 0,
                        "embedding_comparison_metric": "cosine_similarity"
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 1,
                        "boost_function_increasing_function_type": "none"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
    }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 0
    assert search_response.results[0].score == expected_score

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "embedding_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "embedding1_name",
                        "target_embedding": [0, 0, 0, 0, 0],
                        "default_embedding_comparison_value_if_missing": 0,
                        "embedding_comparison_metric": "dot_product"
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 1,
                        "boost_function_increasing_function_type": "none"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
    }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 0
    assert search_response.results[0].score == expected_score

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "embedding_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "embedding1_name",
                        "target_embedding": [0, 0, 0, 0, 2],
                        "default_embedding_comparison_value_if_missing": 0,
                        "embedding_comparison_metric": "dot_product"
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 1,
                        "boost_function_increasing_function_type": "none"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
    }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 10
    assert search_response.results[0].score == expected_score

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "embedding_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "embedding1_name",
                        "target_embedding": [1, 2, 3, 3, 3],
                        "default_embedding_comparison_value_if_missing": 0,
                        "embedding_comparison_metric": "l1_norm"
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 1,
                        "boost_function_increasing_function_type": "none"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
    }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 3
    assert search_response.results[0].score == expected_score

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "embedding_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "embedding1_name",
                        "target_embedding": [1, 2, 3, 0, 0],
                        "default_embedding_comparison_value_if_missing": 0,
                        "embedding_comparison_metric": "l2_norm"
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 1,
                        "boost_function_increasing_function_type": "none"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = math.sqrt(4 * 4 + 5 * 5)
    assert abs(search_response.results[0].score - expected_score) < 0.00001


    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "embedding_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "embedding1_name__",
                        "target_embedding": [1, 2, 3, 0, 0],
                        "default_embedding_comparison_value_if_missing": 0,
                        "embedding_comparison_metric": "l2_norm"
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 1,
                        "boost_function_increasing_function_type": "none"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 0
    assert abs(search_response.results[0].score - expected_score) < 0.00001

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "embedding_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "all_zeros_embedding_field",
                        "target_embedding": [1, 1],
                        "default_embedding_comparison_value_if_missing": 1,
                        "embedding_comparison_metric": "dot_product"
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 1,
                        "boost_function_increasing_function_type": "none"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 0
    assert abs(search_response.results[0].score - expected_score) < 0.00001

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "embedding_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "all_zeros_embedding_field",
                        "target_embedding": [1, 1],
                        "default_embedding_comparison_value_if_missing": 1,
                        "embedding_comparison_metric": "cosine_similarity"
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 1,
                        "boost_function_increasing_function_type": "none"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 1  # zero norm counts as missing
    assert abs(search_response.results[0].score - expected_score) < 0.00001

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "embedding_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "all_zeros_embedding_field",
                        "target_embedding": [1, 1],
                        "default_embedding_comparison_value_if_missing": 1,
                        "embedding_comparison_metric": "l1_norm"
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 1,
                        "boost_function_increasing_function_type": "none"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 2
    assert abs(search_response.results[0].score - expected_score) < 0.00001

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "embedding_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "all_zeros_embedding_field",
                        "target_embedding": [1, 1],
                        "default_embedding_comparison_value_if_missing": 1,
                        "embedding_comparison_metric": "l2_norm"
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 1,
                        "boost_function_increasing_function_type": "none"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = math.sqrt(2)
    assert abs(search_response.results[0].score - expected_score) < 0.00001

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "geolocation_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "location",
                        "target_geolocation": {
                            "latitude": 0,
                            "longitude": -0.55
                        },
                        "default_distance_if_missing": 0,
                        "distance_unit": "meters"
                    },
                    "boost_function_config": {
                        "minimum_value": 60 * 1000,
                        "maximum_value": 65 * 1000

                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 3,
                        "score_multiplier_minimum_value": 0
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 3
    assert abs(search_response.results[0].score - expected_score) < 0.00001

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "geolocation_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "location",
                        "target_geolocation": {
                            "latitude": 0,
                            "longitude": -0.55
                        },
                        "default_distance_if_missing": 0,
                        "distance_unit": "meters"
                    },
                    "boost_function_config": {
                        "minimum_value": 30 * 1000,
                        "maximum_value": 40 * 1000

                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 3,
                        "score_multiplier_minimum_value": 0
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].id == "1"
    expected_score = 0
    assert abs(search_response.results[0].score - expected_score) < 0.00001


def alias_testing(sync_client, alias_name, index_name):
    aliases = sync_client.list_all_indexes_and_aliases()
    print("aliases", aliases)
    assert type(aliases) == dict
    create_alias_resp = sync_client.create_or_override_alias(alias_name, index_name)
    assert type(create_alias_resp) == dict


def bulk_indexing_and_updating_and_deleting_testing(sync_client, index_name):
    items = [
        SearchItem.parse_obj({
            "id": "10",
            "text_fields": {
                "title": "item 2 title",
                "description": "description",
                "source_name": "source_name"
            },
        }),
        SearchItem.parse_obj({
            "id": "20",
            "text_fields": {
                "title": "item 2 title",
                "description": "description",
                "source_name": "source_name"
            },
        }),
        SearchItem.parse_obj({
            "id": "30",
            "text_fields": {
                "title": "item 2 title",
                "description": "description",
                "source_name": "source_name"
            },
        })
    ]
    resp = sync_client.bulk_index_items(index_name, items, refresh="true")
    assert type(resp) == dict

    items_update = [
        SearchItem.parse_obj({
            "id": "10",
            "text_fields": {
                "description": "updated description",
            },
        }),
        SearchItem.parse_obj({
            "id": "20",
            "text_fields": {
                "description": "updated description",
            },
        }),
        SearchItem.parse_obj({
            "id": "30",
            "text_fields": {
                "description": "updated description",
            },
        })
    ]
    resp = sync_client.bulk_update_items(index_name, items_update)
    assert type(resp) == dict

    resp = sync_client.bulk_delete_items(index_name, ["10", "20", "30"], refresh="true")
    assert type(resp) == dict


def approximate_nearest_neighbors_testing(sync_client, index_name):
    # knn search without filters
    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "approximate_embedding_nearest_neighbor_filter": {
                "embedding_field_name": "embedding1_name",
                "embedding": [1, 1, 1, 1, 1],
                "k": 2,
                "num_candidates": 50,
                "apply_filters_in_ann_search": False
            },
            "embedding_fields_to_return": [
                "embedding1_name"
            ],
            "text_fields_to_return": [],
            "categorical_fields_to_return": [],
            "numeric_fields_to_return": [],
            "date_fields_to_return": [],
            "geolocation_fields_to_return": [],
            "return_extra_information": False
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)

    assert search_response.results[0].id in ["1", "2"]
    assert search_response.results[1].id in ["1", "2"]

    # knn search with filters
    if sync_client.settings.database_type == "elasticsearch":
        search_response = \
            sync_client.search(index_name, SearchConfig.parse_obj({
                "id_exclude_list": ["2"],
                "approximate_embedding_nearest_neighbor_filter": {
                    "embedding_field_name": "embedding1_name",
                    "embedding": [1, 1, 1, 1, 1],
                    "k": 2,
                    "num_candidates": 50,
                    "apply_filters_in_ann_search": True
                },
                "embedding_fields_to_return": [
                    "embedding1_name"
                ],
                "text_fields_to_return": [],
                "categorical_fields_to_return": [],
                "numeric_fields_to_return": [],
                "date_fields_to_return": [],
                "geolocation_fields_to_return": [],
                "return_extra_information": False
            }))
        assert type(search_response) == SearchResults
        print("search_response", search_response)

        assert search_response.results[0].id != "2"
        assert search_response.results[1].id != "2"

    # Approx knn search followed by standard search
    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "approximate_embedding_nearest_neighbor_filter": {
                "embedding_field_name": "embedding1_name",
                "embedding": [1, 1, 1, 1, 1],
                "k": 5,
                "num_candidates": 50,
                "apply_filters_in_ann_search": False
            },
            "text_matching": [
                {
                    "query": "test",
                    "text_fields_names_and_weights_dict": {
                        "title": 1,
                        "description": 0.1
                    },
                    "use_ngram_fields_if_available": True,
                    "required": False,
                    "minimum_should_match": 1
                }
            ],
            "categorical_matching": [
                {
                    "categorical_field_name": "authors",
                    "values_list": [
                        "string"
                    ],
                    "score_multiplier": 1,
                    "minimum_should_match": 1,
                    "required": False
                }
            ],
            "numeric_fields_filtering": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0
                    },
                    "minimum_value": 0,
                    "strictly_greater_than": 0,
                }
            ],
            "geolocation_fields_filtering": [
                {
                    "input_value_config": {
                        "field_name": "location",
                        "target_geolocation": {
                            "latitude": 0,
                            "longitude": 0
                        },
                        "default_distance_if_missing": 0,
                        "distance_unit": "meters"
                    },
                    "minimum_value": 0,
                    "maximum_value": 1e15,
                    "strictly_greater_than": -1,
                    "strictly_less_than": 1e15
                }
            ],
            "scoring_factor_combine_mode": "multiply",
            "numeric_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "popularity",
                        "default_field_value_if_missing": 0
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 1,
                        "boost_function_increasing_function_type": "none"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 1,
                        "score_multiplier_variable_boost_weight": 1,
                        "score_multiplier_minimum_value": 0,
                        "score_multiplier_maximum_value": 100
                    }
                }
            ],
            "date_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "published_date",
                        "default_field_value_if_missing": datetime.utcnow(),
                        "target_date": datetime.utcnow(),
                        "time_units": "seconds"
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 1,
                        "boost_function_increasing_function_type": "none"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 1,
                        "score_multiplier_variable_boost_weight": 1,
                        "score_multiplier_minimum_value": 0,
                        "score_multiplier_maximum_value": 100
                    }
                }
            ],
            "embedding_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "embedding1_name",
                        "target_embedding": [1, 0, 0, 0, 0],
                        "default_embedding_comparison_value_if_missing": 0,
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
                        "score_multiplier_maximum_value": 100
                    }
                }
            ],
            "geolocation_fields_boosting": [
                {
                    "input_value_config": {
                        "field_name": "location",
                        "target_geolocation": {
                            "latitude": 0,
                            "longitude": 0
                        },
                        "default_distance_if_missing": 0,
                        "distance_unit": "meters"
                    },
                    "boost_function_config": {
                        "boost_function_input_value_rescaling_factor": 1,
                        "boost_function_increasing_function_type": "none"
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 1,
                        "score_multiplier_variable_boost_weight": 1,
                        "score_multiplier_minimum_value": 0,
                        "score_multiplier_maximum_value": 100
                    }
                }
            ]
        }))
    assert type(search_response) == SearchResults
    print("full search_response", search_response)

    assert len(search_response.results) == 3


def returned_fields_testing(sync_client, index_name):
    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "return_item_data": False
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert search_response.results[0].item is None

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "categorical_fields_to_return": ["tags"]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert "tags" in search_response.results[0].item["categorical_fields"]

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "embedding_fields_to_return": [],
            "categorical_fields_to_return": ["tags"]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert "tags" in search_response.results[0].item["categorical_fields"]
    assert "text_fields" in search_response.results[0].item
    assert "embedding_fields" not in search_response.results[0].item

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "embedding_fields_to_return": ["embedding2_name"],
            "categorical_fields_to_return": ["tags"]
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert "tags" in search_response.results[0].item["categorical_fields"]
    assert "text_fields" in search_response.results[0].item
    assert "embedding1_name" not in search_response.results[0].item["embedding_fields"]
    assert "embedding2_name" in search_response.results[0].item["embedding_fields"]
    assert "extra_information" in search_response.results[0].item

    search_response = \
        sync_client.search(index_name, SearchConfig.parse_obj({
            "id_list": [1],
            "embedding_fields_to_return": ["embedding2_name"],
            "categorical_fields_to_return": ["tags"],
            "return_extra_information": False
        }))
    assert type(search_response) == SearchResults
    print("search_response", search_response)
    assert "tags" in search_response.results[0].item["categorical_fields"]
    assert "text_fields" in search_response.results[0].item
    assert "embedding1_name" not in search_response.results[0].item["embedding_fields"]
    assert "embedding2_name" in search_response.results[0].item["embedding_fields"]
    assert "extra_information" not in search_response.results[0].item


@pytest.mark.parametrize("client_settings", [
    opensearch_settings,
    elasticsearch_settings
])
def test_index_functionality(client_settings):
    print("client_settings", client_settings)

    sync_client = ElasticSearchRetrievalClient(client_settings)

    setup_index(sync_client, pytest_index_name)
    index_and_update_items(sync_client, pytest_index_name)
    alias_testing(sync_client, pytest_alias_name, pytest_index_name)
    bulk_indexing_and_updating_and_deleting_testing(sync_client, pytest_index_name)
    approximate_nearest_neighbors_testing(sync_client, pytest_index_name)
    filtering_testing(sync_client, pytest_index_name)
    search_testing(sync_client, pytest_index_name)
    returned_fields_testing(sync_client, pytest_index_name)