import traceback
from typing import List
from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse

from OpenSearchRec.retrieval.ElasticSearchRetrievalClient import (
    ElasticSearchRetrievalClientSettings,
    ElasticSearchRetrievalClientAsync
)

from OpenSearchRec.retrieval.models.common import (
    OperationRefreshConfig
)

from OpenSearchRec.retrieval.models.index import (
    IndexConfig
)

from OpenSearchRec.retrieval.models.item import (
    SearchItem
)

from OpenSearchRec.retrieval.models.search import (
    SearchConfig,
    SearchResults
)


api = FastAPI(title="OpenSearchRec: Open Source Search and Recommendations", description="")

opensearchrec_settings = ElasticSearchRetrievalClientSettings()


open_search_rec = ElasticSearchRetrievalClientAsync(opensearchrec_settings)


#############################################################################
#                                                                           #
#   Error Handling Utility                                                  #
#                                                                           #
#############################################################################


async def get_json_response(function, args):
    try:
        response_json = await function(**args)
        status_code = 200
    except Exception as e:
        response_json = {"error": str(e)}
        traceback.print_exc()
        status_code = 500
    return JSONResponse(response_json, status_code=status_code) 


#############################################################################
#                                                                           #
#   Index Router                                                            #
#                                                                           #
#############################################################################


index_router = APIRouter(
    prefix="/index",
    tags=["Indexes"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)


@index_router.get("/list")
async def list_all_indexes():
    return await get_json_response(open_search_rec.list_all_indexes, {})


@index_router.get("/{index_name}")
async def get_index(index_name: str):
    return await get_json_response(open_search_rec.get_index, {"index_name": index_name})


@index_router.post("/create_index/{index_name}")
async def create_index(index_name: str, index_config: IndexConfig):
    return await get_json_response(open_search_rec.create_index,
                                   {"index_config": index_config, "index_name": index_name})


@index_router.delete("/{index_name}")
async def delete_index(index_name: str):
    return await get_json_response(open_search_rec.delete_index, {"index_name": index_name})


#############################################################################
#                                                                           #
#   Item Router                                                             #
#                                                                           #
#############################################################################
item_router = APIRouter(
    prefix="/item",
    tags=["Items"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)


@item_router.get("/{index_name}/{item_id}", response_model=SearchItem)
async def get_item(index_name: str, item_id: str, refresh: OperationRefreshConfig = OperationRefreshConfig.false):
    try:
        response_json = await open_search_rec.get_item(index_name=index_name, item_id=item_id, refresh=refresh)
        response_json = response_json.json_serializable_dict()
        status_code = 200
    except Exception as e:
        response_json = {"error": str(e)}
        traceback.print_exc()
        status_code = 500
    return JSONResponse(response_json, status_code=status_code)


@item_router.post("/{index_name}/")
async def index_item(index_name: str, item: SearchItem, refresh: OperationRefreshConfig = OperationRefreshConfig.false):
    try:
        response_json = await open_search_rec.index_item(index_name=index_name, item=item, refresh=refresh)
        response_json = response_json
        status_code = 200
    except Exception as e:
        response_json = {"error": str(e)}
        traceback.print_exc()
        status_code = 500
    return JSONResponse(response_json, status_code=status_code)


@item_router.post("/update/{index_name}/{item_id}")
async def update_item(index_name: str, item: SearchItem, refresh: OperationRefreshConfig = OperationRefreshConfig.false):
    try:
        response_json = await open_search_rec.update_item(index_name=index_name, item=item, refresh=refresh)
        response_json = response_json.body
        status_code = 200
    except Exception as e:
        response_json = {"error": str(e)}
        status_code = 500
    return JSONResponse(response_json, status_code=status_code)


@item_router.delete("/{index_name}/{item_id}")
async def delete_item(index_name: str, item_id: str, refresh: OperationRefreshConfig = OperationRefreshConfig.false):
    try:
        response_json = await open_search_rec.delete_item(index_name=index_name, item_id=item_id, refresh=refresh)
        response_json = response_json.body
        status_code = 200
    except Exception as e:
        response_json = {"error": str(e)}
        status_code = 500
    return JSONResponse(response_json, status_code=status_code)


@item_router.post("/bulk_index/{index_name}")
async def index_item(index_name: str, items: List[SearchItem]):
    try:
        response_json = await open_search_rec.bulk_index_items(index_name=index_name, items=items)
        status_code = 200
    except Exception as e:
        traceback.print_exc()
        response_json = {"error": str(e)}
        status_code = 500
    return JSONResponse(response_json, status_code=status_code)


@item_router.post("/bulk_update/{index_name}")
async def index_item(index_name: str, items: List[SearchItem]):
    try:
        response_json = await open_search_rec.bulk_update_items(index_name=index_name, items=items)
        status_code = 200
    except Exception as e:
        traceback.print_exc()
        response_json = {"error": str(e)}
        status_code = 500
    return JSONResponse(response_json, status_code=status_code) 


#############################################################################
#                                                                           #
#   Search Router                                                            #
#                                                                           #
#############################################################################


search_router = APIRouter(
    prefix="/search",
    tags=["Search"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)


@search_router.post("/{index_name}", response_model=SearchResults)
async def search_items(index_name: str, search_config: SearchConfig):
    try:
        response_json = await open_search_rec.search(index_name=index_name, search_config=search_config)
        for idx in range(len(response_json.results)):
            response_json.results[idx].item = response_json.results[idx].item.json_serializable_dict()
        response_json = response_json.dict()
        status_code = 200
    except Exception as e:
        response_json = {"error": str(e)}
        status_code = 500
    return JSONResponse(response_json, status_code=status_code) 


#############################################################################
#                                                                           #
#   Index Alias Router                                                      #
#                                                                           #
#############################################################################


index_alias_router = APIRouter(
    prefix="/index_alias",
    tags=["Aliases"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)


@index_alias_router.get("/list")
async def list_all_indexes_and_aliases():
    try:
        response_json = await open_search_rec.list_all_indexes_and_aliases()
        status_code = 200
    except Exception as e:
        response_json = {"error": str(e)}
        status_code = 500
    return JSONResponse(response_json, status_code=status_code)


@index_alias_router.post("")
async def create_or_override_alias(index_alias_name: str, index_name: str):
    try:
        response_json = await open_search_rec.create_or_override_alias(index_alias_name, index_name)
        status_code = 200
    except Exception as e:
        response_json = {"error": str(e)}
        status_code = 500
    return JSONResponse(response_json, status_code=status_code) 


@index_alias_router.delete("/{alias_name}")
async def delete_alias(index_alias_name: str):
    try:
        response_json = await open_search_rec.delete_alias(index_alias_name)
        status_code = 200
    except Exception as e:
        response_json = {"error": str(e)}
        status_code = 500
    return JSONResponse(response_json, status_code=status_code) 


@index_alias_router.post("/search/{alias_name}", response_model=SearchResults)
async def search_items(alias_name: str, search_config: SearchConfig):
    try:
        response_json = await open_search_rec.search_using_index_alias(alias_name=alias_name, search_config=search_config)
        response_json = response_json.dict()
        status_code = 200
    except Exception as e:
        response_json = {"error": str(e)}
        status_code = 500
    return JSONResponse(response_json, status_code=status_code) 


#############################################################################
#                                                                           #
#   Other Router                                                            #
#                                                                           #
#############################################################################


other_router = APIRouter(
    tags=["Other"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)


@other_router.get("/", tags=["Other"])
async def healthcheck():
    try:
        response_json = await open_search_rec.health_check()
        status_code = 200
    except Exception as e:
        response_json = {"error": str(e)}
        status_code = 500
    return JSONResponse(response_json, status_code=status_code)


#############################################################################
#                                                                           #
#   Add Routers to App                                                      #
#                                                                           #
#############################################################################

api.include_router(index_router)
api.include_router(item_router)
api.include_router(search_router)
api.include_router(index_alias_router)
api.include_router(other_router)
