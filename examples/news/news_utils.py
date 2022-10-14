import datetime
import os
import traceback
import re
import newspaper


def remove_parameters(url):
    parameter_start_position = url.find('?')
    if parameter_start_position == -1:
        return url
    else:
        return url[:parameter_start_position]


def remove_anchor(url):
    anchor_start_position = url.find('#')
    if anchor_start_position == -1:
        return url
    else:
        return url[:anchor_start_position]


def remove_index_html(url):
    postfix = "/index.html"
    postfix_len = len(postfix)
    if url[-postfix_len:] == postfix:
        return url[:-postfix_len]
    return url


def remove_www(url):
    if url.startswith("https://www."):
        return "https://" + url[len("https://www."):]
    elif url.startswith("http://www."):
        return "http://" + url[len("http://www."):]
    else:
        return url


def get_normalized_url(url):
    url = remove_parameters(url)
    url = remove_anchor(url)
    url = remove_index_html(url)
    url = remove_www(url)    
    url = url.replace("https", "http")

    if url[-1] == "/":
        url = url[:-1]
    return url


def get_domain_from_url(url):
    domain = url.replace("https://", "")
    domain = domain.replace("http://", "")
    domain = domain.replace("www.", "")
    first_slash_location = domain.find("/")
    if first_slash_location != -1:
        domain = domain[:first_slash_location]
    return domain


def get_page_url_from_domain(domain_url, memoize_articles=False):
    paper = newspaper.build(domain_url, memoize_articles=memoize_articles)
    paper_urls = \
        list(map(lambda article: get_normalized_url(article.url), paper.articles))
    paper_urls_fitlered = \
        set(filter(lambda url: get_domain_from_url(url) == get_domain_from_url(domain_url), paper_urls))
    return list(paper_urls_fitlered)


def get_article_dict(article_url):
    article = newspaper.Article(article_url)
    article.download()
    article.parse()
    return {
        "title": article.title,
        "text": article.text,
        "top_image": article.top_image,
        "movies": article.movies,
        "authors": article.authors,
        "publish_date": article.publish_date,
        "url": article.url,
        "normalized_url": get_normalized_url(article.url)
    }


if __name__ == "__main__":
    domain_url = "https://www.cnn.com"
    #domain_url = "https://www.wsj.com"
    urls = get_page_url_from_domain(domain_url)
    print(urls)
    print(len(urls))
    print(get_article_dict(urls[0]))
