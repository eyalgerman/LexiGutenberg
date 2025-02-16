import pandas as pd

prefix = ""
prefix = "/sise/home/germane/BigData/"
metadata_file = "metadata.csv"
METADATA_PATH = prefix + "data/metadata.csv"
TEXTS_PATH = prefix + "gutenberg_books_train"
# Stopwords from "https://www.matthewjockers.net/2013/04/12/secret-recipe-for-topic-modeling-themes/"
STOPWORDS_PATH = prefix + "sw_jockers.txt"


def load_metadata():
    metadata = pd.read_csv(METADATA_PATH, names=['file', 'title', 'year', 'url'], header=None)
    return metadata


def get_metadata_by_year_range(start_year, end_year):
    metadata = load_metadata()
    mask = (metadata['year'] >= start_year) & (metadata['year'] <= end_year)
    return metadata[mask]


def get_metadata_by_file_id(file_id):
    metadata = load_metadata()
    return metadata[metadata['file'] == int(file_id)]


def get_metadata_by_title(title):
    metadata = load_metadata()
    return metadata[metadata['title'].str.contains(title, case=False)]




