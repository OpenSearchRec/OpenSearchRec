from enum import Enum
from pydantic import BaseModel, Field, validator, Extra
from typing import List, Optional, Dict, Any, Type

from OpenSearchRec.retrieval.models.common import (
    raise_exception_if_field_name_not_valid,
    FieldTypeNamesAndPrefixes,
    EmbeddingComparisonMetric
)


class TextMatchingType(str, Enum):
    bm25 = "bm25"
    boolean_matching = "boolean_matching"


class TextMatchingConfigSettings(BaseModel):
    bm25_k1: Optional[float] = \
        Field(1.2, description="For BM25 matching only: Controls non-linear term frequency normalization (saturation).")
    bm25_b: Optional[float] = \
        Field(0.75, description="For BM25 matching only: Controls to what degree "
                                "document length normalizes term frequency values.")
    enable_ngram_tokenizer: bool = \
        Field(True,
              description="Use ngram tokenizer, creates additional ngram fields (overhead). Allows for fuzzy search.")
    analyzer: str = "standard"
    ngram_tokenizer_min_gram: int = 2
    ngram_tokenizer_max_gram: int = 4


class TextMatchingConfig(BaseModel):
    text_matching_type: TextMatchingType = \
        Field(TextMatchingType.bm25, description="BM25 or Boolean Matching")
    settings: TextMatchingConfigSettings = TextMatchingConfigSettings()


class EmbeddingConfig(BaseModel):
    embedding_dimension: int = Field(..., gt=1)
    enable_approximate_nearest_embedding_search: int = \
        Field(False, description="Index embeddings for approximate knn search (overhead)")
    approximate_nearest_embedding_search_metric: EmbeddingComparisonMetric = \
        Field(EmbeddingComparisonMetric.cosine_similarity,
              description="Similarity metric to be used for the approximate nearest neighbor search")


class IndexConfig(BaseModel):
    text_matching_config: TextMatchingConfig = Field(TextMatchingConfig())

    text_fields: Optional[List[str]] = \
        Field(None, description="")

    categorical_fields: Optional[List[str]] = \
        Field(None, description="")

    numeric_fields: Optional[List[str]] = \
        Field(None, description="")

    date_fields: Optional[List[str]] = \
        Field(None, description="")

    geolocation_fields: Optional[List[str]] = \
        Field(None, description="")

    embedding_fields: Optional[Dict[str, EmbeddingConfig]] = \
        Field(None, description="Dimension of SearchItem embeddings.")

    number_of_shards: int = \
        Field(1, description="Number of shards for the index")

    number_of_replicas: int = \
        Field(1, description="Number of replicas for the index")

    refresh_interval: Optional[int] = Field(1)

    @validator(
        *[n.name for n in FieldTypeNamesAndPrefixes],
        pre=True
    )
    def validate_embedding_configs(cls, value, values, config, field):
        for field_name in value:
            raise_exception_if_field_name_not_valid(field_name)
        return value

    class Config:
        extra = Extra.forbid

        @staticmethod
        def schema_extra(schema: Dict[str, Any], model: Type['IndexConfig']) -> None:
            print(schema)
            schema["properties"]["text_fields"]["default"] = [
                "title",
                "description",
                "source_name"
            ]
            schema["properties"]["categorical_fields"]["default"] = [
                "tags",
                "authors"
            ]
            schema["properties"]["numeric_fields"]["default"] = [
                "popularity",
                "quality_signal"
            ]
            schema["properties"]["date_fields"]["default"] = [
                "published_date",
                "last_updated_date"
            ]
            schema["properties"]["geolocation_fields"]["default"] = [
                "location"
            ]

            schema["properties"]["embedding_fields"]["default"] = {
                "embedding1_name": {
                    "embedding_dimension": 5,
                    "enable_approximate_nearest_embedding_search": True,
                    "approximate_nearest_embedding_search_metric": "cosine_similarity"
                },
                "embedding2_name": {
                    "embedding_dimension": 20,
                    "enable_approximate_nearest_embedding_search": True,
                    "approximate_nearest_embedding_search_metric": "cosine_similarity"
                },
                "embedding3_name": {
                    "embedding_dimension": 768,
                    "enable_approximate_nearest_embedding_search": True,
                    "approximate_nearest_embedding_search_metric": "cosine_similarity"
                }
            }
