import datetime
from enum import Enum
from pydantic import BaseModel, Field, Extra, validator
from pydantic.generics import GenericModel
from typing import List, Optional, Union, Dict, Any, Type, Generic, TypeVar

from OpenSearchRec.retrieval.models.common import (
    TextMatchType,
    TimeUnits,
    DistanceUnit,
    GeolocationField,
    EmbeddingComparisonMetric,
    ScalarBoostModifierFunctions,
    BoostDecayFunctions,
    raise_exception_if_field_name_not_valid
)


class TextMatch(BaseModel, extra=Extra.forbid):
    query: str
    text_fields_names_and_weights_dict: Dict[str, float] = \
        Field(None, description="Dictionary where the keys are the fields "
                                "searched and the mapped values are the boost "
                                "weight for a match.")
    match_type: TextMatchType = TextMatchType.cross_fields
    fuzziness: Optional[Union[int, str]] = None
    use_ngram_fields_if_available: bool = Field(True)
    required: bool = \
        Field(True, description="If set to true, items that don't have "
                                "minimum_should_match terms from the query "
                                "will be filtered out (in the fields specified "
                                "in the item_fields_weights dictionary)")
    minimum_should_match: Optional[Union[int, str]] = \
        Field(None, description="Minimum number of terms that need to be found "
                                "for it to be considered a match.", ge=1)

    @validator("text_fields_names_and_weights_dict", pre=True)
    def validate_text_fields_names_and_weights_dict(cls, value, values, config, field):
        for field_name in value:
            raise_exception_if_field_name_not_valid(field_name)
        return value


class CategoricalMatch(BaseModel, extra=Extra.forbid):
    """
        - Boost items that have minimum_should_match
          tags from tags_list by boost_weight
        - Filter out items that don't have minimum_should_match
          tags from tags_list (if required is set to true)
    """
    categorical_field_name: str
    values_list: List[str]
    score_multiplier: Optional[float] = \
        Field(None, description="Score multiplier if there is a match."
                                "When scoring_factor_combine_mode=multiply, will be set to 1 when there isn't a match."
                                "When scoring_factor_combine_mode=sum, will be set to 0 when there isn't a match."
                                "This distinction based on the scoring_factor_combine_mode depends on default "
                                "ElasticSearch behaviour."
                                "Set to None to not boost based on matches.", ge=0)
    minimum_should_match: Union[int, str] = \
        Field(1, description="Minimum number of tags that need to be found for "
                             "it to be considered a match. "
                             "Can be an integer or a percentage string ie. 50%", ge=1)
    required: bool = \
        Field(False, description="Does there need to be a match for an item to "
                                 "be returned")

    @validator("categorical_field_name", pre=True)
    def validate_text_fields_names_and_weights_dict(cls, value, values, config, field):
        raise_exception_if_field_name_not_valid(value)
        return value


################################################################################
#                                                                              #
#  Distance Inputs                                                             #
#                                                                              #
################################################################################


class NumericDistance(BaseModel):
    target_value: float


class DateTemporalDistance(BaseModel):
    target_date: List[float]
    time_units: TimeUnits


class EmbeddingComparisonConfig(BaseModel):
    target_embedding: List[float] = Field([0,0,0,0,0])
    embedding_comparison_metric: EmbeddingComparisonMetric


class GeolocationDistanceConfig(BaseModel):
    target_geolocation: GeolocationField
    distance_unit: DistanceUnit


################################################################################
#                                                                              #
#  Boosting Input Value Configuration                                          #
#                                                                              #
################################################################################


class InputValueConfig(BaseModel, extra=Extra.forbid):
    field_name: str

    @validator("field_name", pre=True)
    def validate_embedding_configs(cls, value, values, config, field):
        raise_exception_if_field_name_not_valid(value)
        return value


class NumericFieldValueConfig(InputValueConfig):
    default_field_value_if_missing: Optional[float]


class NumericDistanceValueConfig(NumericFieldValueConfig):
    target_value: float


class DateFieldValueConfig(InputValueConfig):
    default_field_value_if_missing: Optional[datetime.datetime]


class DateTemporalDistanceValueConfig(DateFieldValueConfig):
    target_date: datetime.datetime
    time_units: TimeUnits


class EmbeddingComparisonValueConfig(InputValueConfig):
    target_embedding: List[float] = Field([0,0,0,0,0])
    default_embedding_comparison_value_if_missing: Optional[float]
    embedding_comparison_metric: EmbeddingComparisonMetric


class GeolocationDistanceValueConfig(InputValueConfig):
    target_geolocation: GeolocationField
    default_distance_if_missing: float
    distance_unit: DistanceUnit


################################################################################
#                                                                              #
#  Generic Types                                                               #
#                                                                              #
################################################################################


Field_Value_Type = TypeVar('T')
Input_Value_Config = TypeVar('T')

################################################################################
#                                                                              #
#  Filter Configs                                                              #
#                                                                              #
################################################################################


class MinMaxValueFilter(GenericModel, Generic[Field_Value_Type, Input_Value_Config], extra=Extra.forbid):
    input_value_config: Input_Value_Config
    minimum_value: Optional[Field_Value_Type]
    maximum_value: Optional[Field_Value_Type]
    strictly_greater_than: Optional[Field_Value_Type]
    strictly_less_than: Optional[Field_Value_Type]


NumericFieldMinMaxFilter = MinMaxValueFilter[float, NumericFieldValueConfig]
DateMinMaxValueFilter = MinMaxValueFilter[datetime.datetime, DateFieldValueConfig]
EmbeddingFieldMinMaxFilter = MinMaxValueFilter[float, EmbeddingComparisonValueConfig]
GeolocationFieldMinMaxFilter = MinMaxValueFilter[float, GeolocationDistanceValueConfig]


################################################################################
#                                                                              #
#  Variable Boosting Function Config                                           #
#                                                                              #
################################################################################


class ValueMinMaxBoostingConfig(BaseModel, extra=Extra.forbid):
    minimum_value: Optional[float]
    maximum_value: Optional[float]
    strictly_greater_than: Optional[float]
    strictly_less_than: Optional[float]


class ValueIncreasingBoostingConfig(BaseModel, extra=Extra.forbid):
    boost_function_input_value_rescaling_factor: Optional[float] = 1
    boost_function_increasing_function_type: Optional[ScalarBoostModifierFunctions] = \
        Field(ScalarBoostModifierFunctions.none)


class ValueDecayingBoostConfig(BaseModel, extra=Extra.forbid):
    decay_boost_type: BoostDecayFunctions = Field(BoostDecayFunctions.gaussian)
    decay_boost_offset: int = Field(0, description="")
    decay_boost_decay_rate: float = Field(0.5, description="")
    decay_boost_decay_scale: int = Field(60 * 60, description="")


################################################################################
#                                                                              #
#  Score Factor Configuration                                                  #
#                                                                              #
################################################################################


class ScoringFactorConfig(BaseModel, extra=Extra.forbid):
    score_multiplier_constant_weight: float = 1
    score_multiplier_variable_boost_weight: float = 1
    score_multiplier_minimum_value: float = 0
    score_multiplier_maximum_value: Optional[float] = \
        Field(None,
              description="Maximum value for the score multiplier, "
                          "incluiding both the constant and variable components")


################################################################################
#                                                                              #
#  Boosting Configs                                                            #
#                                                                              #
################################################################################


class BoostConfig(GenericModel, Generic[Input_Value_Config]):
    input_value_config: Input_Value_Config
    boost_function_config: Union[ValueIncreasingBoostingConfig, ValueMinMaxBoostingConfig, ValueDecayingBoostConfig]
    scoring_factor_config: ScoringFactorConfig


NumericFieldBoostConfig = BoostConfig[Union[NumericFieldValueConfig, NumericDistanceValueConfig]]
DateFieldBoostConfig = BoostConfig[DateTemporalDistanceValueConfig]
EmbeddingFieldBoostConfig = BoostConfig[EmbeddingComparisonValueConfig]
GeolocationFieldBoostConfig = BoostConfig[GeolocationDistanceValueConfig]


class RandomBoostConfig(BaseModel, extra=Extra.forbid):
    constant_value: float = 1
    random_multiplier_min_value: float = 0
    random_multiplier_max_value: Optional[float] = 1


class ScoringFactorCombineMode(str, Enum):
    multiply = "multiply"
    sum = "sum"


class EmbeddingApproximateEmbeddingSearchConfig(BaseModel):
    embedding_field_name: str
    embedding: List[float]
    k: int = Field(10, description="Approximate nearest neighbor search will return approximate top k items.")
    num_candidates: int = Field(50, description="The number of nearest neighbor candidates to consider per shard.")
    apply_filters_in_ann_search: bool = \
        Field(False, description="Whether to apply the query filters during the ANN search.")


class SearchConfig(BaseModel):
    id_list: Optional[List[str]]
    id_exclude_list: Optional[List[str]]

    text_matching: Optional[List[TextMatch]]

    categorical_matching: Optional[List[CategoricalMatch]]

    numeric_fields_filtering: Optional[List[NumericFieldMinMaxFilter]]

    date_fields_filtering: Optional[List[DateMinMaxValueFilter]]

    geolocation_fields_filtering: Optional[List[GeolocationFieldMinMaxFilter]]

    numeric_fields_boosting: Optional[List[NumericFieldBoostConfig]]
    date_fields_boosting: Optional[List[DateFieldBoostConfig]]
    embedding_fields_boosting: Optional[List[EmbeddingFieldBoostConfig]]
    geolocation_fields_boosting: Optional[List[GeolocationFieldBoostConfig]]

    random_boost: Optional[RandomBoostConfig]

    approximate_embedding_nearest_neighbor_filter: Optional[EmbeddingApproximateEmbeddingSearchConfig] = \
        Field(description="Perform an approximate_embedding_nearest_neighbor search as an initial step, "
                          "and use the remaining filters, matching and boost configurations to rescore and filter"
                          "only the documents returned by the approximate embedding nearest neighbor search.")

    scoring_factor_combine_mode: ScoringFactorCombineMode = ScoringFactorCombineMode.multiply

    return_item_data: bool = \
        Field(True, description="Return the item data if true, return only the item_id otherwise")

    text_fields_to_return: Optional[List[str]]
    categorical_fields_to_return: Optional[List[str]]
    numeric_fields_to_return: Optional[List[str]]
    date_fields_to_return: Optional[List[str]]
    embedding_fields_to_return: Optional[List[str]]
    geolocation_fields_to_return: Optional[List[str]]

    return_extra_information: bool = \
        Field(True, description="Whether to return the item extra information")

    limit: int = Field(10, lte=10000)
    start: int = Field(0)

    @validator("text_fields_to_return",
               "categorical_fields_to_return",
               "numeric_fields_to_return",
               "date_fields_to_return",
               "embedding_fields_to_return",
               "geolocation_fields_to_return",
               pre=True)
    def validate_field_names_to_return(cls, value, values, config, field):
        for field_name in value:
            raise_exception_if_field_name_not_valid(field_name)
        return value

    class Config:
        extra = Extra.forbid

        @staticmethod
        def schema_extra(schema: Dict[str, Any], model: Type['SearchConfig']) -> None:
            schema["properties"]["text_matching"]["default"] = [
                {
                    "query": "test",
                    "text_fields_names_and_weights_dict": {
                        "title": 1,
                        "description": 0.1,
                    },
                    "use_ngram_fields_if_available": False,
                    "required": True,
                    "minimum_should_match": 1
                }
            ]


class SearchResult(BaseModel):
    id: str
    score: float
    item: Optional[Any]


class SearchResults(BaseModel):
    total: Dict[str, Union[str, int]]
    shards: Dict[str, Union[str, int]]
    results: List[SearchResult]
