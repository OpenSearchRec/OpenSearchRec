import pytest

from OpenSearchRec.retrieval.models.item import SearchItem


def test_create_SearchItem():
    SearchItem(**{
        "id": "id",
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
            "authors": "author"
        },
        "numeric_fields": {
            "popularity": 1000000,
            "quality_signal": 4.9
        },
        "date_fields": {
            "published_date": "2021-08-11T22:15:06",
            "last_updated_date": "2022-08-11T22:15:06.388811"
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
            ]
        },
        "extra_information": {
            "image_urls": [
                "http://localhost:80/image1.pg",
                "http://localhost:80/image2.pg"
            ]
        }
    })


@pytest.mark.parametrize("flat_dict_item", [
    {
        "id": "1",
        "text_title": "title",
        "text_description": "description",
        "embedding_als": [1,2,3,4,5,6,7,8]
    },
    {
        # "id": None,
        "text_title": "title",
        "numeric_popularity": 1,
        "embedding_als": [1,2,3,4,5,6,7,8]
    },

])
def test_mapping_helper_functions(flat_dict_item):
    search_item = SearchItem.from_flat_mapping_dict(flat_dict_item)
    flat_dict_item_from_search_item = search_item.to_flat_mapping_dict()
    assert set(flat_dict_item.keys()) == set(flat_dict_item_from_search_item.keys())
    assert flat_dict_item == flat_dict_item_from_search_item
    search_item_from_regenerated_flat_dict = \
        SearchItem.from_flat_mapping_dict(flat_dict_item_from_search_item)
    assert search_item == search_item_from_regenerated_flat_dict


def test_field_name_validation():
    with pytest.raises(Exception) as exception:
        SearchItem(text_fields={"text*": "text"})
    with pytest.raises(Exception) as exception:
        SearchItem(embedding_field={"": [1]})
    with pytest.raises(Exception) as exception:
        SearchItem.from_flat_mapping_dict({
            "id": 1,
            "embedding_field*": [1]
        })
    SearchItem.from_flat_mapping_dict({
        "id": 1,
        "embedding_field": [1]
    })


if __name__ == "__main__":
    test_mapping_helper_functions()
