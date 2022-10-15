# OpenSearchRec: Open Source Search and Recommendations

OpenSearchRec is an open source search engine and recommendation system library. At the moment, in includes:
- An ElasticSearch or OpenSearch powered retrieval engine that can be used to create effective, high availability and high scalability search engines and recommendation systems. The retrieval client can be used to create simple end-to-end search and/or recommendation engines. Alternatively, OpenSearchRec can be used as the first step of a multistep search/recommendation pipeline. In this scenario, OpenSearchRec is used to generate a shortlist of search results of a more manageable size which is then reranked with additional algorithms/models which are less scalable and would be too slow to be applied to the entire search dataset. For example, OpenSearchRec could be used to generate a shortlist of the top 1000 results from a search index containing 1 million of document and then other models, which would be too slow to be applied to all of the 1 million documents, would be used to reorder the top 1000 documents returned by OpenSearchRec. Some key features include:
    - Filtering and boosting based on a variety of signals, including terms matching, tags, numeric values, dates, locations and embedding comparison metrics.
    - Uses a novel formula to combine multiple different boosting signals in a more balanced way compared to what is available in the Elasticsearch/OpenSearch query language (implemented using custom scripts as part of the queries).
    - High availability and scalability (based on ElasticSearch/OpenSearch, which can be deployed with replication and sharding).
    - It can be used as either a microservice API or as a python library.
    - The Python client is available in both synchronous and asynchronous versions.
    - The API is built using FastAPI and has interactive documentation.
- Other utility functions, such as for generating collaborative filtering embeddings and clustering search results.


## Example Use Cases:
- Semantic search engine that combines terms matching and embedding similarity based matching with filtering and boosting based on other factors such as quality, popularity, dates, locations and tags.
- Embedding based recommendations system with filtering and boosting based on other factors such as ratings, popularity, recency,  tags and location.


## Demo Applications:
- News Search Engine
    - Crawl news articles from multiple sources and make them available for search.
    - Using clustering to identify important stories in recent news (home page).
    - In examples/news subdirectory
- Goodreads Demo: Example of a content search and collaborative filtering embedding based recommendation engine.
  - Uses Goodreads dataset containing information on books, authors and books reviews by anonymized users (available [here](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home))
  - Features:
    - Searching for books and authors
    - Book recommendations on books pages based on embeddings produced using the alternating least squares algorithm (ie books you may also like)
    - Author recommendations on authors pages based on embeddings produced using the alternating least squares algorithm (ie authors you may also like)
    - Autosuggest
    - Uses OpenSearchRec via API calls
    - In examples/goodreads subdirectory


## Running the Elasticsearch Retrieval Engine as a FastAPI endpoint
- For looking at the interactive documentation or for development
- These instructions can be run from this directory
- Important to note that endpoint does not have built-in authentication and https

### Set environment variables  
- The example values here are for a dev setup
- The environment variables are picked up by the ElasticSearchRetrievalClientSettings object, which is used in the OpenSearchRecRetrievalAPI and defined in the OpenSearchRec/retrieval/ElasticSearchRetrievalClient.py file.
- The elasticsearch_username and elasticsearch_password values are also by docker-compose-dev-elasticsearch.yml
```
export database_type=elasticsearch
export elasticsearch_host=http://localhost:9200
export elasticsearch_username=elastic
export elasticsearch_password=elasticsearch_password
export elasticsearch_verify_certificates=false
```

### Start of ElasticSearch 
```
docker compose -f docker-compose-dev-elasticsearch.yml up -d
```
### Install OpenSearchRec and Dependencies
- You may want to use a python virtual environment for this
```
python3 -m venv env
. env/bin/activate
pip install --upgrade pip
pip install OpenSearchRec
```
### Run Endpoint
```
uvicorn OpenSearchRec.retrieval.OpenSearchRecRetrievalAPI:api 
```

### View the interactive documentation at http://127.0.0.1:8000/docs


## Retrieval Engine Boosting Formulas:
New boosting formulas were developed to address 2 weaknesses with the commonly used formulas available in ElasticSearch/OpenSearch:

1. Balancing the relative importance of boosting factors doesn't work very well:
   - If the boosting factors are combined by a weighted sum, it can be very difficult if not impossible to weight the different signals correctly given that they vary at different rates and on different scales.
   - If the boosting factors are combined via multiplication (which removes the need to balance their relative scales), a missing or low value can cause the score to collapse and be zero. For example, if a boost is done based on boost_factor = log(1 + number of views), then an item that has zero views will have a boosting factor equal to zero, which will cause its score to be zero since the factors are combined via multiplication. This could cause it to not be returned even if it is the only relevant result for a query. This issue can be addressed in the new formulas by adding the boost value to a constant value, for example:  boost_factor = 5 + log(1 + number of views). That way, the boost can be though of as a percentage increase from a base value and the score will not be zero when the number of views are 0.
2. Robustness to outlier values. This is addressed by allowing minimum and maximum values to be set for the boosting factors. For the boost by popularity example, a maximum could be set to prevent an unusually popular item from having an overly large boost, for example, a popularity boost factor would become: boost_factor = min(18, 9 + log(1 + number of views)). This would limit the boost to 18, which is 9 more than the base value of 9, meaning that the largest possible popularity boost would be 100%. In a search engine, this mechanism could be used to prevent items that are irrelevant to a search query from being returned because they get an overly large boost due to being much more popular than the relevant items.

For numeric, date, embedding and geolocation signals, the boosting factor are combined via multiplication. Each boosting factor is computed in 3 steps:
1. Value: An input value is generated for the boosting formula. A default value can be provided in case the field is missing. The way to get the value depends on the field type:
    - Numeric: Either the numeric value itself or the absolute value for the difference between the numeric value and another provided value.
    - Date: The time difference from a target date is used (in seconds, minutes, hours or days).
    - Embedding: An embedding comparison metric is used to compare the embedding with a provided embedding. The available metrics are cosine_similarity, dot_product, l2_norm and l1_norm.
    - Geolocation: The distance is used, either in meters or kilometers.
2. Variable Boosting Function: A boosting function is applied to modify the value. The available functions are:
    -  Min/Max boost: Apply a constant boost to values between the specified min and max.
    -  Increasing boost: An increasing function is applied to the value. Also the value can be rescaled using a multiplicative factor before the function is applied. The available functions are: "none", "log1p", "log2p", "sqrt", "square". The "none" function doesn't modify the value. Example use: boost a video based on the logarithm of its view count (numeric value).
    -  Decaying Function: A decaying function is used (larger values have a smaller boost.) The available decay functions are: exponential, linear, gaussian. Example use: boost a restaurant based on how close it is to the person making the search (geolocation based boost).
3. Score Factor: A constant number is added to the output of the variable boosting function. As a result, the output can be thought of as a percentage increase from a base value. Alternatively, the constant value can be set to zero to avoid this effect. Optionally, a minimum and maximum value can be set for the boosting factor to increase robustness to outliers.


## Running Unit and Integration Tests
- There are 2000+ lines of unit and integration test code
- These instructions can be run from this directory

### Start of ElasticSearch and OpenSearch Docker Containers for Testing
```
docker compose -f tests/docker-compose-testing.yml up -d
```
### Install OpenSearchRec and Dependencies
- You may want to use a python virtual environment for this
```
python3 -m venv env
. env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r models_requirements.txt
```
### Run Tests
```
pytest
```

### Shutdown the testing ElasticSearch and OpenSearch containers
```
docker compose -f tests/docker-compose-testing.yml down
```