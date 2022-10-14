# How to run 

## Open a terminal in this directory (OpenSearchRec/examples/news/)

## Start ElasticSearch Container
```
docker compose up -d
```

## Create a Virtual Environment and Install Python Dependencies
```
python3 -m venv env
. env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Crawl News
```
python crawl_news.py
```

## Start Web UI
```
python news_ui.py
```

## Go to http://127.0.0.1:5000

## In case of GPU/CUDA related errors, you can try:
```
CUDA_VISIBLE_DEVICES=-1 python crawl_news.py
```
and then
```
CUDA_VISIBLE_DEVICES=-1 python news_ui.py  # Go to http://127.0.0.1:5000
```
