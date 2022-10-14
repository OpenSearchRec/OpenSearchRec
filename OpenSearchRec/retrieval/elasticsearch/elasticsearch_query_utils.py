from datetime import datetime
import numpy as np
from typing import Dict

from OpenSearchRec.retrieval.models.common import (
    DatabaseType,
    TextMatchType,
    FieldTypeNamesAndPrefixes,
    BoostDecayFunctions,
    TimeUnits,
    DistanceUnit,
    EmbeddingComparisonMetric
)

from OpenSearchRec.retrieval.models.index import (
    IndexConfig,
    TextMatchingType
)

from OpenSearchRec.retrieval.models.search import (
    SearchConfig,
    NumericFieldValueConfig,
    NumericDistanceValueConfig,
    DateFieldValueConfig,
    DateTemporalDistanceValueConfig,
    ValueIncreasingBoostingConfig,
    ValueMinMaxBoostingConfig,
    ValueDecayingBoostConfig,
    EmbeddingComparisonValueConfig,
    GeolocationDistanceValueConfig
)


def get_index_mapping(index_config: IndexConfig, database_type: DatabaseType):
    mapping_propertes = {}

    if index_config.text_fields is not None:
        if index_config.text_matching_config.settings.enable_ngram_tokenizer:
            text_field_mapping = {
                "type": "text",
                "analyzer": index_config.text_matching_config.settings.analyzer,
                "fields": {
                    "ngrams": {
                        "type": "text",
                        "analyzer": "ngram_analyzer"
                    }
                }
            }
        else:
            text_field_mapping = {
                "type": "text"
            }
        for text_field_name in index_config.text_fields:
            mapping_field_name = \
                FieldTypeNamesAndPrefixes.text_fields.value + text_field_name
            mapping_propertes[mapping_field_name] = text_field_mapping

    if index_config.categorical_fields is not None:
        for categorical_field_name in index_config.categorical_fields:
            mapping_field_name = \
                FieldTypeNamesAndPrefixes.categorical_fields.value + categorical_field_name
            mapping_propertes[mapping_field_name] = {"type": "keyword"}

    if index_config.date_fields is not None:
        for date_field_name in index_config.date_fields:
            mapping_field_name = \
                FieldTypeNamesAndPrefixes.date_fields.value + date_field_name
            mapping_propertes[mapping_field_name] = {"type": "date"}

    if index_config.embedding_fields is not None:
        for embedding_field_name, embedding_config in index_config.embedding_fields.items():
            embedding_field_name = \
                FieldTypeNamesAndPrefixes.embedding_fields.value + embedding_field_name

            if database_type == DatabaseType.elasticsearch:
                mapping_properties = {
                    "type": "dense_vector",
                    "dims": embedding_config.embedding_dimension
                }
                if embedding_config.enable_approximate_nearest_embedding_search:
                    mapping_properties["index"] = True
                    if embedding_config.approximate_nearest_embedding_search_metric == EmbeddingComparisonMetric.cosine_similarity:
                        mapping_properties["similarity"] = "cosine"
                    elif embedding_config.approximate_nearest_embedding_search_metric == EmbeddingComparisonMetric.dot_product:
                        mapping_properties["similarity"] = "dot_product"
                    elif embedding_config.approximate_nearest_embedding_search_metric == EmbeddingComparisonMetric.l2_norm:
                        mapping_properties["similarity"] = "l2_norm"
                    else:
                        ValueError(f"EmbeddingComparisonMetric value not supported: {embedding_config.approximate_nearest_embedding_search_metric}")
            elif database_type == DatabaseType.opensearch:
                mapping_properties = {
                    "type": "knn_vector",
                    "dimension": embedding_config.embedding_dimension
                }
                if embedding_config.enable_approximate_nearest_embedding_search:
                    config_space_type = embedding_config.approximate_nearest_embedding_search_metric
                    if config_space_type == EmbeddingComparisonMetric.cosine_similarity:
                        space_type = "cosinesimil"
                    elif config_space_type == EmbeddingComparisonMetric.dot_product:
                        space_type = "innerproduct"
                    elif config_space_type == EmbeddingComparisonMetric.l2_norm:
                        space_type = "l2"
                    else:
                        ValueError(f"EmbeddingComparisonMetric value not supported: {config_space_type}")
                    mapping_properties["index"] = True
                    mapping_properties["method"] = {
                        "name": "hnsw",
                        "space_type": space_type,
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 128,
                            "m": 24
                        }
                    }

            mapping_propertes[embedding_field_name] = mapping_properties

    if index_config.geolocation_fields is not None:
        for geolocation_field_name in index_config.geolocation_fields:
            mapping_field_name = \
                FieldTypeNamesAndPrefixes.geolocation_fields.value + geolocation_field_name
            mapping_propertes[mapping_field_name] = {"type": "geo_point"}

    return {"properties": mapping_propertes}


def get_index_settings(index_config: IndexConfig, database_type: DatabaseType):
    settings = {
        "index": {},
        "index.max_result_window": "100000",
        "max_ngram_diff": "3",
        "number_of_shards": index_config.number_of_shards,
        "number_of_replicas": index_config.number_of_replicas,
    }

    if index_config.refresh_interval is not None:
        settings["index"]["refresh_interval"] = index_config.refresh_interval

    if index_config.text_matching_config.text_matching_type == TextMatchingType.bm25:
        settings_dictionary = index_config.text_matching_config.settings
        settings["index"] = {
            "similarity": {
                "default": {
                    "type": "BM25",
                    "b": settings_dictionary.bm25_b,
                    "k1": settings_dictionary.bm25_k1
                }
            }
        }
    elif index_config.text_matching_config.text_matching_type == TextMatchingType.boolean_matching:
        print("boolean matching")
        settings["index"] = {
            "similarity": {
                "default": {
                    "type": "boolean"
                }
            }
        }

    settings["index"]["analysis"] = {
        "analyzer": {
            "ngram_analyzer": {
                "tokenizer": "ngram_tokenizer",
                "filter": [
                    "lowercase"
                ]
            }
        },
        "tokenizer": {
            "ngram_tokenizer": {
                "type": "ngram",
                "min_gram": index_config.text_matching_config.settings.ngram_tokenizer_min_gram,
                "max_gram": index_config.text_matching_config.settings.ngram_tokenizer_max_gram,
                "token_chars": [
                    "letter",
                    "digit"
                ]
            }
        }
    }

    has_vector_field_with_approx_knn = False
    has_vector_field_without_approx_knn = False
    if database_type == DatabaseType.opensearch:
        if index_config.embedding_fields is not None:
            for embedding_field_name, embedding_field_config in index_config.embedding_fields.items():
                if embedding_field_config.enable_approximate_nearest_embedding_search:
                    has_vector_field_with_approx_knn = True
                else:
                    has_vector_field_without_approx_knn = True

    if has_vector_field_with_approx_knn and has_vector_field_without_approx_knn:
        raise Exception("When database_type == DatabaseType.opensearch, "
                        "approximate knn search has to be enable either for all embedding fields "
                        "or none of them")
    elif has_vector_field_with_approx_knn and not has_vector_field_without_approx_knn:
        settings["index"]["knn"] = True

    return settings


def get_input_value_step_script_and_params_dict(
        input_value_config,
        return_value_if_missing_and_no_default,
        database_type: DatabaseType):
    if type(input_value_config) == NumericFieldValueConfig:
        field_name = FieldTypeNamesAndPrefixes.numeric_fields.value + input_value_config.field_name
        input_value_step_script = """
            double boost_input_value;
            if (!doc.containsKey(params.field_name) || doc[params.field_name].empty) {
                if (params.default_field_value_if_missing == null) {
                    return params.return_value_if_missing_and_no_default;
                } else {
                    boost_input_value = params.default_field_value_if_missing;
                }
            } else {
                boost_input_value = doc[params.field_name].value;
            }
            """
        input_value_step_script_parameters = {
            "field_name": field_name,
            "default_field_value_if_missing":
                input_value_config.default_field_value_if_missing,
            "return_value_if_missing_and_no_default": return_value_if_missing_and_no_default
        }
    elif type(input_value_config) == NumericDistanceValueConfig:
        print("NumericDistanceValueConfig")
        field_name = FieldTypeNamesAndPrefixes.numeric_fields.value + input_value_config.field_name
        input_value_step_script = """
            double boost_input_value;
            if (!doc.containsKey(params.field_name) || doc[params.field_name].empty) {
                if (params.default_field_value_if_missing == null) {
                    return params.return_value_if_missing_and_no_default;
                } else {
                    boost_input_value = params.default_field_value_if_missing;
                }
            } else {
                boost_input_value = doc[params.field_name].value;
            }
            boost_input_value = Math.abs(boost_input_value - params.target_value);
            """

        input_value_step_script_parameters = {
            "field_name": field_name,
            "default_field_value_if_missing":
                input_value_config.default_field_value_if_missing,
            "target_value": input_value_config.target_value,
            "return_value_if_missing_and_no_default": return_value_if_missing_and_no_default
        }

    elif type(input_value_config) == DateFieldValueConfig:
        field_name = FieldTypeNamesAndPrefixes.date_fields.value + input_value_config.field_name

        input_value_step_script = """
            double boost_input_value;
            if (!doc.containsKey(params.field_name) || doc[params.field_name].empty) {
                if (params.default_timestamp_value_if_missing == null) {
                    return params.return_value_if_missing_and_no_default;
                } else {
                    boost_input_value = params.default_timestamp_value_if_missing;
                }
            } else {
                boost_input_value = doc[params.field_name].value.millis / 1000;
            }
            """

        if input_value_config.default_field_value_if_missing is not None:
            default_field_value_if_missing = input_value_config.default_field_value_if_missing.timestamp()
        else:
            default_field_value_if_missing = None

        input_value_step_script_parameters = {
            "field_name": field_name,
            "default_timestamp_value_if_missing": default_field_value_if_missing,
            "return_value_if_missing_and_no_default": return_value_if_missing_and_no_default
        }

    elif type(input_value_config) == DateTemporalDistanceValueConfig:
        field_name = FieldTypeNamesAndPrefixes.date_fields.value + input_value_config.field_name

        input_value_step_script = """
            double boost_input_value;
            if (!doc.containsKey(params.field_name) || doc[params.field_name].empty) {
                if (params.default_timestamp_value_if_missing == null) {
                    return params.return_value_if_missing_and_no_default;
                } else {
                    boost_input_value = params.default_timestamp_value_if_missing;
                }
            } else {
                boost_input_value = doc[params.field_name].value.millis / 1000.0;
            }
            boost_input_value = Math.abs(boost_input_value - params.target_timestamp_value);
            """
        # input_value_step_script = "double boost_input_value = 0;"

        if input_value_config.time_units == TimeUnits.seconds:
            pass  # already in seconds
        elif input_value_config.time_units == TimeUnits.minutes:
            input_value_step_script += "boost_input_value = boost_input_value / 60;"
        elif input_value_config.time_units == TimeUnits.hours:
            input_value_step_script += "boost_input_value = boost_input_value / 3600;"
        elif input_value_config.time_units == TimeUnits.days:
            input_value_step_script += "boost_input_value = boost_input_value / 86400;"
        else:
            raise Exception(f"Invalid TimeUnit = {input_value_config.time_units}")

        if input_value_config.default_field_value_if_missing is not None:
            default_field_value_if_missing = input_value_config.default_field_value_if_missing.timestamp()
        else:
            default_field_value_if_missing = None

        input_value_step_script_parameters = {
            "field_name": field_name,
            "default_timestamp_value_if_missing": default_field_value_if_missing,
            "target_timestamp_value": input_value_config.target_date.timestamp(),
            "time_units": input_value_config.time_units,
            "return_value_if_missing_and_no_default": return_value_if_missing_and_no_default
        }

    elif type(input_value_config) == EmbeddingComparisonValueConfig:
        field_name = FieldTypeNamesAndPrefixes.embedding_fields.value + input_value_config.field_name

        other_vector_conditions_for_missing_value = ""
        if input_value_config.embedding_comparison_metric == EmbeddingComparisonMetric.cosine_similarity:
            if database_type == DatabaseType.elasticsearch:
                other_vector_conditions_for_missing_value = "|| l2norm(params.zero_vector, params.field_name) == 0"
            elif database_type == DatabaseType.opensearch:
                other_vector_conditions_for_missing_value = "|| l2Squared(params.zero_vector, doc[params.field_name]) == 0"

            if database_type == DatabaseType.elasticsearch:
                compute_boost_input_value_line = f"boost_input_value = 1.0 + cosineSimilarity(params.target_embedding, params.field_name);"
            elif database_type == DatabaseType.opensearch:
                compute_boost_input_value_line = f"boost_input_value = 1.0 + cosineSimilarity(params.target_embedding, doc[params.field_name]);"
            else:
                raise Exception(f"Invalid database type = {database_type}")

        elif input_value_config.embedding_comparison_metric == EmbeddingComparisonMetric.dot_product:
            if database_type == DatabaseType.elasticsearch:
                compute_boost_input_value_line = "boost_input_value = dotProduct(params.target_embedding, params.field_name);"
            elif database_type == DatabaseType.opensearch:
                target_embedding_norm = np.linalg.norm(input_value_config.target_embedding)
                compute_boost_input_value_line = """
                    if (l2Squared(params.zero_vector, doc[params.field_name]) == 0) {
                        boost_input_value = 0;
                    } else {
                        boost_input_value = cosineSimilarity(params.target_embedding, doc[params.field_name]);
                        boost_input_value = boost_input_value * Math.sqrt(l2Squared(params.zero_vector, doc[params.field_name]));
                        boost_input_value = boost_input_value * """ + str(target_embedding_norm) + """;
                    }
                """
            else:
                raise Exception(f"Invalid database type = {database_type}")

        elif input_value_config.embedding_comparison_metric == EmbeddingComparisonMetric.l1_norm:
            if database_type == DatabaseType.elasticsearch:
                compute_boost_input_value_line = "boost_input_value = l1norm(params.target_embedding, params.field_name);"
            elif database_type == DatabaseType.opensearch:
                compute_boost_input_value_line = "boost_input_value = l1Norm(params.target_embedding, doc[params.field_name]);"
            else:
                raise Exception(f"Invalid database type = {database_type}")
        elif input_value_config.embedding_comparison_metric == EmbeddingComparisonMetric.l2_norm:

            if database_type == DatabaseType.elasticsearch:
                compute_boost_input_value_line = "boost_input_value = l2norm(params.target_embedding, params.field_name);"
            elif database_type == DatabaseType.opensearch:
                compute_boost_input_value_line = "boost_input_value = Math.sqrt(l2Squared(params.target_embedding, doc[params.field_name]));"
            else:
                raise Exception(f"Invalid database type = {database_type}")
        else:
            raise Exception(f"Invalid embedding_comparison_metric = {input_value_config.embedding_comparison_metric}")

        if input_value_config.embedding_comparison_metric == EmbeddingComparisonMetric.cosine_similarity:
            input_value_step_script = """
                double boost_input_value;
                if (!doc.containsKey(params.field_name) || doc[params.field_name].empty""" + other_vector_conditions_for_missing_value + """) {
                    if (params.default_embedding_comparison_value_if_missing == null) {
                        return params.return_value_if_missing_and_no_default;
                    } else {
                        boost_input_value = params.default_embedding_comparison_value_if_missing;
                    }
                } else {
                    """ + compute_boost_input_value_line + """
                }
                """
        else:
            input_value_step_script = """
                double boost_input_value;
                if (!doc.containsKey(params.field_name) || doc[params.field_name].empty) {
                    if (params.default_embedding_comparison_value_if_missing == null) {
                        return params.return_value_if_missing_and_no_default;
                    } else {
                        boost_input_value = params.default_embedding_comparison_value_if_missing;
                    }
                } else {
                    """ + compute_boost_input_value_line + """
                }
                """

        if np.linalg.norm(input_value_config.target_embedding) == 0:
            input_value_step_script = """
                double boost_input_value;
                if (params.default_embedding_comparison_value_if_missing == null) {
                    return params.return_value_if_missing_and_no_default;
                } else {
                    boost_input_value = params.default_embedding_comparison_value_if_missing;
                }
                """

        input_value_step_script_parameters = {
            "field_name": field_name,
            "target_embedding":
                input_value_config.target_embedding,
            "zero_vector": [0 for _ in input_value_config.target_embedding],
            "default_embedding_comparison_value_if_missing":
                input_value_config.default_embedding_comparison_value_if_missing,
            "return_value_if_missing_and_no_default": return_value_if_missing_and_no_default
        }

    elif type(input_value_config) == GeolocationDistanceValueConfig:
        field_name = FieldTypeNamesAndPrefixes.geolocation_fields.value + input_value_config.field_name

        input_value_step_script = """
            double boost_input_value;
            if (!doc.containsKey(params.field_name) || doc[params.field_name].empty) {
                if (params.default_distance_if_missing == null) {
                    return params.return_value_if_missing_and_no_default;
                } else {
                    boost_input_value = params.default_distance_if_missing;
                }
            } else {
                boost_input_value = doc[params.field_name].arcDistance(params.latitude,params.longitude);
            }
            """

        if input_value_config.distance_unit == DistanceUnit.kilometers:
            input_value_step_script += "boost_input_value = boost_input_value / 1000.0;"

        input_value_step_script_parameters = {
            "field_name": field_name,
            "latitude":
                input_value_config.target_geolocation.latitude,
            "longitude":
                input_value_config.target_geolocation.longitude,
            "default_distance_if_missing":
                input_value_config.default_distance_if_missing,
            "return_value_if_missing_and_no_default": return_value_if_missing_and_no_default
        }

    else:
        raise Exception(f"Invalid input_value_config = {input_value_config}, type = {type(input_value_config)}")

    return input_value_step_script, input_value_step_script_parameters


def get_filtering_script(filter_config, database_type: DatabaseType):
    return_value_if_missing_and_no_default = False  # filter out
    input_value_step_script, input_value_step_script_parameters = \
        get_input_value_step_script_and_params_dict(
            filter_config.input_value_config, return_value_if_missing_and_no_default, database_type)

    filter_script = ""
    filter_script_params = {}

    if filter_config.minimum_value is not None:
        filter_script += "if (boost_input_value < params.minimum_value) { return false; }"
        filter_script_params["minimum_value"] = filter_config.minimum_value

    if filter_config.maximum_value is not None:
        filter_script += "if (boost_input_value > params.maximum_value) { return false; }"
        filter_script_params["maximum_value"] = filter_config.maximum_value

    if filter_config.strictly_greater_than is not None:
        filter_script += "if (boost_input_value <= params.strictly_greater_than) { return false; }"
        filter_script_params["strictly_greater_than"] = filter_config.strictly_greater_than

    if filter_config.strictly_less_than is not None:
        filter_script += "if (boost_input_value >= params.strictly_less_than) { return false; }"
        filter_script_params["strictly_less_than"] = filter_config.strictly_less_than

    filter_script += "return true;"

    for param_name, param_value in filter_script_params.items():
        if type(param_value) == datetime:
            filter_script_params[param_name] = param_value.timestamp()

    script_source = input_value_step_script + filter_script
    filter_script = {
        "script": {
            "script": {
                "source": script_source,
                "params": {
                    **input_value_step_script_parameters,
                    **filter_script_params
                }
            }
        }
    }

    return filter_script


def get_boost_script(boost_config, database_type: DatabaseType):
    """
        Generally 3 steps:
            1. Value: Get value to boost by (or default).
                    - Depends on field type.
                    - It doesn't return, creates boost_input_value variable for use in later stages
            2. Variable Boosting Function: Apply boosting function.
                    - Depends on boosting type.
                    - Creates variable_boost_function_value variable
            3. Score Multiplier: Apply constant score, weight and min/max. Same everywhere.
    """

    # Boost input value step
    input_value_step_script, input_value_step_script_parameters = \
        get_input_value_step_script_and_params_dict(
            boost_config.input_value_config,
            boost_config.scoring_factor_config.score_multiplier_constant_weight,
            database_type)

    # Boost function step
    if type(boost_config.boost_function_config) == ValueMinMaxBoostingConfig:
        boost_function_script = """
            double variable_boost_function_value = 1;
            if (params.minimum_value != null) {
                if (boost_input_value < params.minimum_value) {
                    variable_boost_function_value = 0;
                }
            }
            if (params.maximum_value != null) {
                if (boost_input_value > params.maximum_value) {
                    variable_boost_function_value = 0;
                }
            }
            if (params.strictly_greater_than != null) {
                if (boost_input_value <= params.strictly_greater_than) {
                    variable_boost_function_value = 0;
                }
            }
            if (params.strictly_less_than != null) {
                if (boost_input_value >= params.strictly_less_than) {
                    variable_boost_function_value = 0;
                }
            }
            """
        boost_function_parameters = {
            "minimum_value": boost_config.boost_function_config.minimum_value,
            "maximum_value": boost_config.boost_function_config.maximum_value,
            "strictly_greater_than": boost_config.boost_function_config.strictly_greater_than,
            "strictly_less_than": boost_config.boost_function_config.strictly_less_than,
        }

    elif type(boost_config.boost_function_config) == ValueIncreasingBoostingConfig:
        modifier_func_name = boost_config.boost_function_config.boost_function_increasing_function_type
        if modifier_func_name == "none":
            variable_boost_function_value = "variable_boost_function_input"
        elif modifier_func_name == "log1p":
            variable_boost_function_value = "Math.log10(1 + variable_boost_function_input)"
        elif modifier_func_name == "log2p":
            variable_boost_function_value = "Math.log10(2 + variable_boost_function_input)"
        elif modifier_func_name == "sqrt":
            variable_boost_function_value = "Math.sqrt(variable_boost_function_input)"
        elif modifier_func_name == "square":
            variable_boost_function_value = "Math.pow(variable_boost_function_input, 2)"
        else:
            raise Exception(f"Invalid boost_function_increasing_function_type: {modifier_func_name}")

        boost_function_script = """
            double variable_boost_function_input = \
                params.boost_function_input_value_rescaling_factor * boost_input_value;
            double variable_boost_function_value = """ + variable_boost_function_value + """;
            """
        boost_function_parameters = {
            "boost_function_input_value_rescaling_factor":
                boost_config.boost_function_config.boost_function_input_value_rescaling_factor,
            "boost_function_increasing_function_type":
                boost_config.boost_function_config.boost_function_increasing_function_type,
        }

    elif type(boost_config.boost_function_config) == ValueDecayingBoostConfig:
        if boost_config.boost_function_config.decay_boost_type == BoostDecayFunctions.exponential:
            boost_function_script = """
                double lambda = Math.log(params.decay_boost_decay_rate) / params.decay_boost_decay_scale;
                double variable_boost_function_value = \
                    Math.exp(lambda * Math.max(0, Math.abs(boost_input_value) - params.decay_boost_offset));
            """
        elif boost_config.boost_function_config.decay_boost_type == BoostDecayFunctions.linear:
            boost_function_script = """
                double s = params.decay_boost_decay_scale / (1 - params.decay_boost_decay_rate);
                double variable_boost_function_value = \
                    Math.max(0, (s - Math.max(0, Math.abs(boost_input_value) - params.decay_boost_offset)) / s);
            """
        elif boost_config.boost_function_config.decay_boost_type == BoostDecayFunctions.gaussian:
            boost_function_script = """
                double variance = -Math.pow(params.decay_boost_decay_scale, 2) / (2.0 * Math.log(params.decay_boost_decay_rate));
                double variable_boost_function_value = \
                    Math.exp(-Math.pow(Math.max(0, Math.abs(boost_input_value) - params.decay_boost_offset), 2) / (2 * variance) );
            """

        else:
            raise Exception(f"Invalid decay_boost_type: {boost_config.boost_function_config.decay_boost_type}")

        boost_function_parameters = {
            "decay_boost_type": boost_config.boost_function_config.decay_boost_type,
            "decay_boost_offset": boost_config.boost_function_config.decay_boost_offset,
            "decay_boost_decay_rate": boost_config.boost_function_config.decay_boost_decay_rate,
            "decay_boost_decay_scale": boost_config.boost_function_config.decay_boost_decay_scale
        }

    score_multiplier_score_script = """
        double variable_boost = \
            params.score_multiplier_variable_boost_weight * variable_boost_function_value;
        double score_multiplier = params.score_multiplier_constant_weight + variable_boost;
        if (score_multiplier < params.score_multiplier_minimum_value) {
            score_multiplier = params.score_multiplier_minimum_value;
        }
        if (params.score_multiplier_maximum_value != null) {
            if (score_multiplier > params.score_multiplier_maximum_value) {
                score_multiplier = params.score_multiplier_maximum_value;
            }
        }
        return score_multiplier;
    """

    scoring_factor_script_config = {
        "score_multiplier_constant_weight":
            boost_config.scoring_factor_config.score_multiplier_constant_weight,
        "score_multiplier_variable_boost_weight":
            boost_config.scoring_factor_config.score_multiplier_variable_boost_weight,
        "score_multiplier_minimum_value":
            boost_config.scoring_factor_config.score_multiplier_minimum_value,
        "score_multiplier_maximum_value":
            boost_config.scoring_factor_config.score_multiplier_maximum_value,
    }

    source = \
        input_value_step_script + \
        boost_function_script + \
        score_multiplier_score_script

    boost_script_score = {
        "script_score": {
            "script": {
                "params": {
                    **input_value_step_script_parameters,
                    **boost_function_parameters,
                    **scoring_factor_script_config
                },
                "source": source
            }
        }
    }
    return boost_script_score


def get_query_components(search_config: SearchConfig, database_type: DatabaseType):
    query_must = []
    query_must_not = []
    query_should = []
    query_filter = []

    boosting_stage_query_functions = [{"weight": 1}]

    # Filtering
    if search_config.id_list is not None:
        query_filter.append({"terms": {"_id": search_config.id_list}})

    if search_config.id_exclude_list is not None:
        query_must_not.append({"terms": {"_id": search_config.id_exclude_list}})

    if search_config.numeric_fields_filtering is not None:
        for filter_config in search_config.numeric_fields_filtering:
            query_filter.append(get_filtering_script(filter_config, database_type))

    if search_config.date_fields_filtering is not None:
        for filter_config in search_config.date_fields_filtering:
            query_filter.append(get_filtering_script(filter_config, database_type))

    if search_config.geolocation_fields_filtering is not None:
        for filter_config in search_config.geolocation_fields_filtering:
            query_filter.append(get_filtering_script(filter_config, database_type))

    # Terms Matching
    if search_config.text_matching is not None:
        for text_match_config in search_config.text_matching:
            search_query_match_fields_list = []
            for field_name, field_weight in text_match_config.text_fields_names_and_weights_dict.items():
                if text_match_config.use_ngram_fields_if_available:
                    search_query_match_fields_list.append(
                        FieldTypeNamesAndPrefixes.text_fields.value + field_name + "*^" + str(field_weight))
                else:
                    search_query_match_fields_list.append(
                        FieldTypeNamesAndPrefixes.text_fields.value + field_name + "^" + str(field_weight))

            if text_match_config.minimum_should_match is not None:
                minimum_should_match = text_match_config.minimum_should_match
            else:
                minimum_should_match = 0
            text_match_query = {
                "multi_match": {
                    "query": text_match_config.query,
                    "fields": search_query_match_fields_list,
                    "type": text_match_config.match_type,
                    "minimum_should_match": minimum_should_match
                }
            }

            if text_match_config.fuzziness is not None:
                if text_match_config.match_type in [TextMatchType.cross_fields]:
                    raise ValueError(f"{text_match_config.fuzziness} for supported for match_type = {text_match_config.match_type}")
                text_match_query["multi_match"]["fuzziness"] = text_match_config.fuzziness

            if text_match_config.required:
                query_must.append(text_match_query)
            else:
                query_should.append(text_match_query)

    if len(query_must) == 0:
        query_must.append({"match_all": {}})

    # Categorical Matching
    if search_config.categorical_matching is not None:
        for categorical_match_config in search_config.categorical_matching:
            field_name = FieldTypeNamesAndPrefixes.categorical_fields.value + categorical_match_config.categorical_field_name
            match_query = {
                "bool": {
                    "should": [
                        {
                            "term": {field_name: tag}
                        } for tag in categorical_match_config.values_list
                    ],
                    "minimum_should_match": categorical_match_config.minimum_should_match
                }
            }
            if categorical_match_config.required:
                query_filter.append(match_query)
            if categorical_match_config.score_multiplier is not None and categorical_match_config.score_multiplier > 0:
                boosting_stage_query_functions.append({
                    "filter": match_query,
                    "weight": categorical_match_config.score_multiplier
                })

    # Boosting
    if search_config.numeric_fields_boosting is not None:
        for boost_config in search_config.numeric_fields_boosting:
            boosting_stage_query_functions.append(get_boost_script(boost_config, database_type))
    if search_config.date_fields_boosting is not None:
        for boost_config in search_config.date_fields_boosting:
            boosting_stage_query_functions.append(get_boost_script(boost_config, database_type))
    if search_config.embedding_fields_boosting is not None:
        for boost_config in search_config.embedding_fields_boosting:
            boosting_stage_query_functions.append(get_boost_script(boost_config, database_type))
    if search_config.geolocation_fields_boosting is not None:
        for boost_config in search_config.geolocation_fields_boosting:
            boosting_stage_query_functions.append(get_boost_script(boost_config, database_type))

    if search_config.random_boost is not None:
        assert search_config.random_boost.random_multiplier_max_value >= search_config.random_boost.random_multiplier_min_value
        boosting_stage_query_functions.append({
            "script_score": {
                "script": {
                    "params": {
                        "constant_value": search_config.random_boost.constant_value,
                        "random_multiplier_min_value": search_config.random_boost.random_multiplier_min_value,
                        "random_multiplier_max_value": search_config.random_boost.random_multiplier_max_value
                    },
                    "source": "params.constant_value + params.random_multiplier_min_value + (params.random_multiplier_max_value - params.random_multiplier_min_value) * Math.random()"
                }
            }
        })

    return {
        "query_must": query_must,
        "query_must_not": query_must_not,
        "query_should": query_should,
        "query_filter": query_filter,
        "boosting_stage_query_functions": boosting_stage_query_functions
    }


def generate_knn_query(search_config: SearchConfig, query_components: Dict, database_type: DatabaseType):
    embedding_field_name = \
        "embedding_" + search_config.approximate_embedding_nearest_neighbor_filter.embedding_field_name
    query_vector = search_config.approximate_embedding_nearest_neighbor_filter.embedding
    if database_type == DatabaseType.elasticsearch:
        if search_config.approximate_embedding_nearest_neighbor_filter is not None:
            knn_query = {
                "knn": {
                    "field": embedding_field_name,
                    "query_vector": query_vector,
                    "k": search_config.approximate_embedding_nearest_neighbor_filter.k,
                    "num_candidates": search_config.approximate_embedding_nearest_neighbor_filter.num_candidates
                }
            }

            if search_config.approximate_embedding_nearest_neighbor_filter.apply_filters_in_ann_search:
                knn_query["knn"]["filter"] = {
                    "bool": {
                        "must": query_components["query_must"],
                        "filter": query_components["query_filter"],
                        "must_not": query_components["query_must_not"],
                    }
                }
        else:
            knn_query = None
    elif database_type == DatabaseType.opensearch:
        if search_config.approximate_embedding_nearest_neighbor_filter is not None:
            knn_query = {
                "query": {
                    "knn": {
                        embedding_field_name: {
                            "vector": query_vector,
                            "k": search_config.approximate_embedding_nearest_neighbor_filter.k
                        }
                    }
                },
                "size": search_config.approximate_embedding_nearest_neighbor_filter.k
            }

            if search_config.approximate_embedding_nearest_neighbor_filter.apply_filters_in_ann_search:
                raise Exception("apply_filters_in_ann_search not available with database_type == DatabaseType.opensearch")

        else:
            knn_query = None

    return knn_query


def generate_search_query(search_config: SearchConfig, query_components: Dict):
    query_must = query_components["query_must"]
    query_must_not = query_components["query_must_not"]
    query_should = query_components["query_should"]
    query_filter = query_components["query_filter"]
    boosting_stage_query_functions = query_components["boosting_stage_query_functions"]

    boolean_query = {
        "bool": {
            "must": query_must,
            "should": query_should,
            "filter": query_filter,
            "must_not": query_must_not,
            "minimum_should_match": 0
        }
    }

    search_query = {
        "function_score": {
            "query": boolean_query,
            "functions": boosting_stage_query_functions,
            "score_mode": search_config.scoring_factor_combine_mode.value,
            "boost_mode": "multiply"
        }
    }

    return search_query
