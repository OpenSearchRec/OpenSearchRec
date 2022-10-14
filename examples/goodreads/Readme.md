# Overview
- Demo using a Goodreads dataset of books, authors and books reviews.
- OpenSearchRec retrieval client via the API
- Dataset: https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home

# How to run
1. Download book_id_map.csv, user_id_map.csv and goodreads_interactions.csv from https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home and put is data/ local directory
2. Install requirements in requirements.txt files
3. Run compute_authors_embeddings.py and compute_books_embeddings.py to generate the books and authors embeddings in the alternating_least_square_embeddings local directory
4. Start ElasticSearch and OpenSearchRec retrieval client locally
5. Run index_authors.py and index_books.py to index the data
6. Run demo_ui.py and go to http://127.0.0.1:5555/ to see the web application.