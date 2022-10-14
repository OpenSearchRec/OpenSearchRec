import json
from pydantic import BaseSettings, Field
from typing import List, Union
import httpx
from enum import Enum

from OpenSearchRec.utils.AsyncClassToSyncClassMetaClass import SyncClassGenerationMetaClass
from OpenSearchRec.utils.string_sanitization import sanitize_url_string_component

from OpenSearchRec.retrieval.models.search import (
    SearchConfig,
    SearchResult,
    SearchResults
)

from OpenSearchRec.retrieval.models.index import (
    IndexConfig
)

from OpenSearchRec.retrieval.models.item import (
    SearchItem
)

from OpenSearchRec.retrieval.models.common import (
    DatabaseType,
    OperationRefreshConfig,
    FieldTypeNamesAndPrefixes,
    raise_exception_if_field_name_not_valid
)


from OpenSearchRec.retrieval.elasticsearch.elasticsearch_query_utils import (
    get_query_components,
    generate_knn_query,
    generate_search_query
)

from OpenSearchRec.retrieval.elasticsearch.elasticsearch_query_utils import get_index_mapping, get_index_settings


class ElasticSearchRetrievalClientSettings(BaseSettings):
    database_type: DatabaseType = "elasticsearch"
    elasticsearch_host: Union[str, List[str]] = Field("http://localhost:9200")
    elasticsearch_index_prefix: Union[str, List[str]] = Field("opensearchrec_index_prefix_")
    elasticsearch_alias_prefix: Union[str, List[str]] = Field("opensearchrec_alias_prefix_")
    elasticsearch_username: str = "elastic"  # Field("elastic")
    elasticsearch_password: str = "admin"  #  Field("change_this_elasticsearch_password")
    elasticsearch_verify_certificates: bool = Field(False, description="Verify Elasticsearch SSL certificates.")


class RequestType(str, Enum):
    get = "get"
    post = "post"
    put = "put"
    delete = "delete"


async def execute_async_es_request(
        request_type: RequestType,
        request_url_path,
        request_params,
        settings,
        allowed_status_codes,
        timeout=15):
    base_url = settings.elasticsearch_host
    if base_url[-1] == "/" and request_url_path[0] == "/":
        request_url = base_url[:-1] + request_url_path
    elif base_url[-1] != "/" and request_url_path[0] != "/":
        request_url = base_url + "/" + request_url_path
    else:
        request_url = base_url + request_url_path

    async with httpx.AsyncClient(verify=settings.elasticsearch_verify_certificates) as client:
        if request_type == RequestType.get:
            resp = \
                await client.get(
                    request_url, **request_params,
                    auth=(settings.elasticsearch_username, settings.elasticsearch_password),
                    timeout=timeout,
                    follow_redirects=False)
        elif request_type == RequestType.post:
            resp = \
                await client.post(
                    request_url, **request_params,
                    auth=(settings.elasticsearch_username, settings.elasticsearch_password),
                    timeout=timeout,
                    follow_redirects=False)
        elif request_type == RequestType.put:
            resp = \
                await client.put(
                    request_url, **request_params,
                    auth=(settings.elasticsearch_username, settings.elasticsearch_password),
                    timeout=timeout,
                    follow_redirects=False)
        elif request_type == RequestType.delete:
            resp = \
                await client.delete(
                    request_url, **request_params,
                    auth=(settings.elasticsearch_username, settings.elasticsearch_password),
                    timeout=timeout,
                    follow_redirects=False)

    message = f"resp.status_code  = {resp.status_code}, " \
              f"resp.text  = {resp.text}, " \
              f"allowed_status_codes = {allowed_status_codes}, " \
              f"request_type = {request_type}, " \
              f"request_url_path = {request_url_path}, " \
              f"request_url = {request_url}, " \
              f"request_params = {request_params}"
    assert resp.status_code in allowed_status_codes, message

    return resp


class ElasticSearchRetrievalClientAsync:
    def __init__(self, settings: ElasticSearchRetrievalClientSettings):
        self.settings = settings
        self.auth = (self.settings.elasticsearch_username, self.settings.elasticsearch_password)
        self.verify = settings.elasticsearch_verify_certificates

    #############################################################################
    #                                                                           #
    #   Health Check Function                                                   #
    #                                                                           #
    #############################################################################
    async def health_check(self):
        health = \
            await execute_async_es_request(RequestType.get, "/_cluster/health?format=json", {}, self.settings, [200])
        return {"opensearchrec_api_server": "OK!", "elasticsearch": health.json()}

    #############################################################################
    #                                                                           #
    #   Index Functions                                                         #
    #                                                                           #
    #############################################################################

    async def list_all_indexes(self):
        list_indexes_resp = \
            await execute_async_es_request(
                RequestType.get,
                f"/_cat/indices/{sanitize_url_string_component(self.settings.elasticsearch_index_prefix)}*?format=json",
                request_params={},
                settings=self.settings,
                allowed_status_codes=[200])

        response_json = [
            index_info["index"][len(self.settings.elasticsearch_index_prefix):]
            for index_info in list_indexes_resp.json()
        ]
        return response_json

    async def get_index(self, index_name: str):
        resp = \
            await execute_async_es_request(
                RequestType.get,
                f"{sanitize_url_string_component(self.settings.elasticsearch_index_prefix + index_name)}?format=json",
                request_params={},
                settings=self.settings,
                allowed_status_codes=[200])

        return {"index_name": index_name, "elasticsearch_index_info": resp.json()}

    async def create_index(self, index_name: str, index_config: IndexConfig):
        index_settings = get_index_settings(index_config, self.settings.database_type)
        index_mapping = get_index_mapping(index_config, self.settings.database_type)

        index_json = {
            "settings": index_settings,
            "mappings": index_mapping
        }
        create_index_resp = \
            await execute_async_es_request(
                RequestType.put,
                f"/{sanitize_url_string_component(self.settings.elasticsearch_index_prefix + index_name)}",
                {"json": index_json}, self.settings, [200])

        response_json = {
            "elasticsearch_response": create_index_resp.json(),
            "index_name": index_name,
            "index_mapping": index_mapping,
            "index_settings": index_settings
        }
        return response_json

    async def delete_index(self, index_name: str):
        delete_index_resp = \
            await execute_async_es_request(
                RequestType.delete, "/" + sanitize_url_string_component(self.settings.elasticsearch_index_prefix + index_name),
                {}, self.settings, [200])

        return {"elasticsearch_response": delete_index_resp.text}

    #############################################################################
    #                                                                           #
    #   Item Functions                                                          #
    #                                                                           #
    #############################################################################

    async def get_item(self, index_name: str, item_id: str, refresh: OperationRefreshConfig = OperationRefreshConfig.false):
        raise_exception_if_field_name_not_valid(index_name)
        raise_exception_if_field_name_not_valid(item_id)
        result = \
            await execute_async_es_request(
                RequestType.get,
                f"/{sanitize_url_string_component(self.settings.elasticsearch_index_prefix + index_name)}/_doc/{sanitize_url_string_component(item_id)}?refresh={sanitize_url_string_component(refresh)}",
                {}, self.settings, [200])
        item_json = result.json()["_source"]
        search_item = SearchItem.from_flat_mapping_dict(item_json)
        return search_item

    async def index_item(self, index_name: str, item: SearchItem, refresh: OperationRefreshConfig = OperationRefreshConfig.false):
        if item.id is not None:
            resp = \
                await execute_async_es_request(
                    RequestType.post,
                    f"/{sanitize_url_string_component(self.settings.elasticsearch_index_prefix + index_name)}/_doc/{sanitize_url_string_component(item.id)}?refresh={sanitize_url_string_component(refresh)}",
                    {"json": item.to_flat_mapping_dict(convert_datetimes_to_string=True)},
                    self.settings, [200, 201])
            return resp.json()
        else:
            resp = \
                await execute_async_es_request(
                    RequestType.post,
                    f"/{sanitize_url_string_component(self.settings.elasticsearch_index_prefix + index_name)}/_doc?refresh={sanitize_url_string_component(refresh)}",
                    {"json": item.to_flat_mapping_dict(convert_datetimes_to_string=True)},
                    self.settings, [201])
            return resp.json()

    async def update_item(self, index_name: str, item: SearchItem, refresh: OperationRefreshConfig = OperationRefreshConfig.false):
        raise_exception_if_field_name_not_valid(item.id)
        resp = \
            await execute_async_es_request(
                RequestType.post,
                f"/{sanitize_url_string_component(self.settings.elasticsearch_index_prefix + index_name)}/_update/{sanitize_url_string_component(item.id)}?refresh={sanitize_url_string_component(refresh)}",
                {"json": {"doc": item.to_flat_mapping_dict(convert_datetimes_to_string=True)}},
                self.settings, [200])
        return resp.json()

    async def bulk_index_items(self, index_name: str, items: List[SearchItem], refresh: OperationRefreshConfig = OperationRefreshConfig.false):
        actions = []
        for item in items:
            item_action = {
                "index": {
                    "_index": self.settings.elasticsearch_index_prefix + index_name,
                }
            }
            if item.id is not None:
                item_action["index"]["_id"] = item.id
            actions.append(json.dumps(item_action))
            actions.append(json.dumps(item.to_flat_mapping_dict(convert_datetimes_to_string=True)))
        resp = \
            await execute_async_es_request(
                RequestType.post,
                f"/_bulk?refresh={sanitize_url_string_component(refresh)}",
                {
                    "content": "\n".join(actions) + "\n",
                    "headers": {'content-type': 'application/json'}
                },
                self.settings, [200])
        return resp.json()

    async def bulk_delete_items(self, index_name: str, item_ids: List[str], refresh: OperationRefreshConfig = OperationRefreshConfig.false):
        actions = []
        for item_id in item_ids:
            raise_exception_if_field_name_not_valid(item_id)
            item_action = {
                "delete": {
                    "_index": self.settings.elasticsearch_index_prefix + index_name,
                    "_id": item_id
                }
            }
            actions.append(json.dumps(item_action))
        resp = \
            await execute_async_es_request(
                RequestType.post,
                f"/_bulk?refresh={sanitize_url_string_component(refresh)}",
                {
                    "content": "\n".join(actions) + "\n",
                    "headers": {'content-type': 'application/json'}
                },
                self.settings, [200])
        return resp.json()

    async def bulk_update_items(self, index_name: str, items: List[SearchItem], refresh: OperationRefreshConfig = OperationRefreshConfig.false):
        actions = []
        for item in items:
            if item.id is None:
                raise Exception("Every item needs to have an id.")
            item_action = {
                "update": {
                    "_index": self.settings.elasticsearch_index_prefix + index_name,
                    "_id": item.id
                }
            }
            actions.append(json.dumps(item_action))
            actions.append(json.dumps({"doc": item.to_flat_mapping_dict(convert_datetimes_to_string=True)}))
        resp = \
            await execute_async_es_request(
                RequestType.post,
                f"/_bulk?refresh={sanitize_url_string_component(refresh)}",
                {
                    "content": "\n".join(actions) + "\n",
                    "headers": {'content-type': 'application/json'}  # , 'charset':'UTF-8'}
                },
                self.settings, [200])
        return resp.json()

    async def delete_item(self, index_name: str, item_id: str, refresh: OperationRefreshConfig = OperationRefreshConfig.false):
        raise_exception_if_field_name_not_valid(item_id)
        resp = \
            await execute_async_es_request(
                RequestType.delete,
                f"/{sanitize_url_string_component(self.settings.elasticsearch_index_prefix + index_name)}/_doc/{sanitize_url_string_component(item_id)}?refresh={sanitize_url_string_component(refresh)}",
                {}, self.settings, [200])
        return resp.json()

    #############################################################################
    #                                                                           #
    #   Search Functions                                                        #
    #                                                                           #
    #############################################################################

    async def _search(self, es_index_or_alias_name: str, search_config: SearchConfig):
        query_components = get_query_components(search_config, self.settings.database_type)

        # knn search query
        if search_config.approximate_embedding_nearest_neighbor_filter is not None:
            knn_query = \
                generate_knn_query(search_config, query_components, self.settings.database_type)

            knn_results = \
                await execute_async_es_request(
                    RequestType.post,
                    f"/{sanitize_url_string_component(es_index_or_alias_name)}/_search",
                    {
                        "json": knn_query
                    },
                    self.settings, [200])
            knn_results = knn_results.json()

            knn_result_ids = [r["_id"] for r in knn_results["hits"].get("hits", [])]
        else: 
            knn_result_ids = None


        if knn_result_ids is not None:
            query_components["query_filter"].append({"terms": {"_id": knn_result_ids}})

        search_query = generate_search_query(search_config, query_components)

        # search query
        if not search_config.return_item_data:
            source = False
        else:
            source = True
            source_include = ["id"]
            source_exclude = []
            if search_config.text_fields_to_return is None:
                source_include.append(FieldTypeNamesAndPrefixes.text_fields.value + "*")
            else:
                for field_name in search_config.text_fields_to_return:
                    source_include.append(FieldTypeNamesAndPrefixes.text_fields.value + field_name)

            if search_config.categorical_fields_to_return is None:
                source_include.append(FieldTypeNamesAndPrefixes.categorical_fields.value + "*")
            else:
                for field_name in search_config.categorical_fields_to_return:
                    source_include.append(FieldTypeNamesAndPrefixes.categorical_fields.value + field_name)

            if search_config.numeric_fields_to_return is None:
                source_include.append(FieldTypeNamesAndPrefixes.numeric_fields.value + "*")
            else:
                for field_name in search_config.numeric_fields_to_return:
                    source_include.append(FieldTypeNamesAndPrefixes.numeric_fields.value + field_name)

            if search_config.date_fields_to_return is None:
                source_include.append(FieldTypeNamesAndPrefixes.date_fields.value + "*")
            else:
                for field_name in search_config.date_fields_to_return:
                    source_include.append(FieldTypeNamesAndPrefixes.date_fields.value + field_name)

            if search_config.embedding_fields_to_return is None:
                source_include.append(FieldTypeNamesAndPrefixes.embedding_fields.value + "*")
            else:
                for field_name in search_config.embedding_fields_to_return:
                    source_include.append(FieldTypeNamesAndPrefixes.embedding_fields.value + field_name)

            if search_config.geolocation_fields_to_return is None:
                source_include.append(FieldTypeNamesAndPrefixes.geolocation_fields.value + "*")
            else:
                for field_name in search_config.geolocation_fields_to_return:
                    source_include.append(FieldTypeNamesAndPrefixes.geolocation_fields.value + field_name)

            if not search_config.return_extra_information:
                source_exclude.append("extra_information*")
            else:
                source_include.append("extra_information*")

            source = {
                "include": source_include,
                "exclude": source_exclude
            }

        search_resp = \
            await execute_async_es_request(
                RequestType.post,
                f"/{sanitize_url_string_component(es_index_or_alias_name)}/_search",
                {
                    "json": {
                        "query": search_query,
                        "from": search_config.start,
                        "size": search_config.limit,
                        "_source": source,
                    }
                },
                self.settings, [200])
        search_resp_json = search_resp.json()

        hits = search_resp_json["hits"].get("hits", [])
        search_results = []
        for hit in hits:
            if search_config.return_item_data:
                if "id" not in hit["_source"]:
                    hit["_source"]["id"] = hit["_id"]
                item = SearchItem.from_flat_mapping_dict(hit["_source"])
            else:
                item = None
            search_results.append(
                SearchResult(
                    id=hit["_id"],
                    score=hit["_score"],
                    item=item,
                )
            )
        search_results = SearchResults(
            results=search_results,
            total=search_resp_json["hits"]["total"],
            shards=search_resp_json["_shards"]
        )

        return search_results

    async def search(self, index_name: str, search_config: SearchConfig):
        return await ElasticSearchRetrievalClientAsync._search(self, self.settings.elasticsearch_index_prefix + index_name, search_config)

    async def search_using_index_alias(self, alias_name: str, search_config: SearchConfig):
        return await ElasticSearchRetrievalClientAsync._search(self, self.settings.elasticsearch_alias_prefix + alias_name, search_config)

    #############################################################################
    #                                                                           #
    #   Index Alias Functions                                                   #
    #                                                                           #
    #############################################################################

    async def list_all_indexes_and_aliases(self):
        list_aliases_response = \
            await execute_async_es_request(
                RequestType.get,
                f"/_alias/{sanitize_url_string_component(self.settings.elasticsearch_alias_prefix)}*?format=json",
                {}, self.settings, [200])

        list_aliases_response_json = list_aliases_response.json()
        aliases_to_indexes_dict = {}
        for index_name_with_prefix, info_dict in list_aliases_response_json.items():
            index_name = index_name_with_prefix[len(self.settings.elasticsearch_index_prefix):]
            for alias_name_with_prefix in info_dict.get("aliases", {}):
                alias_name = alias_name_with_prefix[len(self.settings.elasticsearch_alias_prefix):]
                assert(alias_name not in aliases_to_indexes_dict)
                aliases_to_indexes_dict[alias_name] = index_name
        response = aliases_to_indexes_dict
        return response

    async def create_or_override_alias(self, index_alias_name: str, index_name: str):
        actions = {
            "actions": [
                {
                    "add": {
                        "index": self.settings.elasticsearch_index_prefix + index_name,
                        "alias": self.settings.elasticsearch_alias_prefix + index_alias_name
                    }
                }
            ]
        }

        aliases = await ElasticSearchRetrievalClientAsync.list_all_indexes_and_aliases(self)
        if index_alias_name in aliases:
            actions["actions"].append({
                "remove": {
                    "index": self.settings.elasticsearch_index_prefix + aliases[index_alias_name],
                    "alias": self.settings.elasticsearch_alias_prefix + index_alias_name
                }
            })

        create_alias_response = \
            await execute_async_es_request(
                RequestType.post, "/_aliases", {"json": actions}, self.settings, [200])

        create_alias_response_json = create_alias_response.json()

        return {
            "es_response": create_alias_response_json,
            "index_alias_name": index_alias_name,
            "index_name": index_name
        }

    async def delete_alias(self, index_alias_name: str):
        aliases = await ElasticSearchRetrievalClientAsync.list_all_indexes_and_aliases(self)

        actions = {
            "actions": [
                {
                    "remove": {
                        "alias": self.settings.elasticsearch_alias_prefix + index_alias_name,
                        "index": self.settings.elasticsearch_index_prefix + aliases[index_alias_name]
                    }
                }
            ]
        }
        delete_alias_response = \
            await execute_async_es_request(
                RequestType.post, "/_aliases", {"json": actions}, self.settings, [200])

        return delete_alias_response.json()


class ElasticSearchRetrievalClient(ElasticSearchRetrievalClientAsync, metaclass=SyncClassGenerationMetaClass):
    """
        Uses the SyncClassGenerationMetaClass metaclass to transform all the async
        methods of ElasticSearchRetrievalClientAsync into synchronous methods.
    """
    pass
