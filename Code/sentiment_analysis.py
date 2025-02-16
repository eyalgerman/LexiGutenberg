import nltk
import matplotlib.pyplot as plt
import metadata_manager
import numpy as np
import os
import pandas as pd
import json
from metadata_manager import TEXTS_PATH
from tqdm import tqdm

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def nltk_sentiment_text(text):
    """
    Perform sentiment analysis on a text using the NLTK Vader sentiment analyzer.
    """
    sia = SentimentIntensityAnalyzer()
    sentences = text.split('\n')

    # We'll average the sentiment across all sentences
    scores = {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
    count = 0

    for sentence in sentences:
        if sentence.strip():
            sentiment = sia.polarity_scores(sentence)
            for k in scores:
                scores[k] += sentiment[k]
            count += 1

    # Compute average
    for k in scores:
        scores[k] = scores[k] / max(count, 1)  # avoid division by zero

    return scores['compound'], scores['pos'], scores['neu'], scores['neg']


def text_blob_sentiment_analysis_text(text):
    """
    Perform sentiment analysis on a text using the TextBlob library.
    """
    from textblob import TextBlob

    blob = TextBlob(text)
    polarity_sum = 0.0
    subjectivity_sum = 0.0
    sentence_count = 0

    for sentence in blob.sentences:
        polarity_sum += sentence.sentiment.polarity
        subjectivity_sum += sentence.sentiment.subjectivity
        sentence_count += 1

    avg_polarity = polarity_sum / max(sentence_count, 1)
    avg_subjectivity = subjectivity_sum / max(sentence_count, 1)
    return avg_polarity, avg_subjectivity


def find_sentiment_in_folder(folder_path, output_dir="saved_results", kind='nltk'):
    """
    Perform sentiment analysis on all text files in a folder and save the results to a CSV file.
    """
    # sentiments_by_year = {}
    sentiment_records = []

    for file in tqdm(os.listdir(folder_path), desc="Processing texts"):
        if not file.endswith('.txt'):
            continue

        file_id = file.split('.')[0]
        file_data = metadata_manager.get_metadata_by_file_id(file_id)
        file_year = file_data['year'].values[0] if not file_data.empty else None

        if file_year is None:
            continue

        file_year = str(int(file_year))

        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Perform sentiment analysis
        if kind == 'nltk':
            compound, pos, neu, neg = nltk_sentiment_text(text)
            sentiment_records.append({
                'book_id': file_id,
                'year': file_year,
                "compound": compound,
                "positive": pos,
                "neutral": neu,
                "negative": neg
            })
        elif kind == 'textblob':
            polarity, subjectivity = text_blob_sentiment_analysis_text(text)
            sentiment_records.append({
                'book_id': file_id,
                'year': file_year,
                "polarity": polarity,
                "subjectivity": subjectivity
            })

    # Save the results to a CSV file
    df_sentiment = pd.DataFrame(sentiment_records)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"sentiment_analysis_results_{kind}_{timestamp}.csv")
    df_sentiment.to_csv(output_file, index=False)
    return df_sentiment


def plot_sentiment_trends_over_years(df_sentiment, save_path=None):
    """
    Plots the trend of different sentiment features over the years using a line plot.
    Saves the plot to a file if save_path is provided.

    Parameters:
        df_sentiment (pd.DataFrame): DataFrame containing sentiment analysis results with a 'year' column.
        save_path (str, optional): Path to save the plot image. Defaults to None.
    """
    df_sentiment['year'] = df_sentiment['year'].astype(int)
    mean_sentiment_by_year = df_sentiment.drop(columns=['book_id']).groupby('year').mean()

    plt.figure(figsize=(12, 6))
    for column in mean_sentiment_by_year.columns:
        plt.plot(mean_sentiment_by_year.index, mean_sentiment_by_year[column], marker='o', label=column)

    plt.title("Sentiment Trends Over Years")
    plt.xlabel("Year")
    plt.ylabel("Sentiment Score")
    plt.legend(title="Sentiment Features")
    plt.grid(True)

    # Save plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to '{save_path}'")

    plt.show()

if __name__ == "__main__":
    folder_path = TEXTS_PATH
    output_dir = "saved_results"
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    kind = "nltk"
    df_sentiment = find_sentiment_in_folder(folder_path, kind=kind, output_dir=output_dir)
    # path = "/sise/home/germane/BigData/saved_results/sentiment_analysis_results_nltk_20250129_134503.csv"
    # df_sentiment = pd.read_csv(path)
    plot_file = os.path.join(output_dir, f"sentiment_trends_{kind}_{timestamp}.png")
    plot_sentiment_trends_over_years(df_sentiment, save_path=plot_file)

    kind = "textblob"
    df_sentiment = find_sentiment_in_folder(folder_path, kind=kind, output_dir=output_dir)
    plot_file = os.path.join(output_dir, f"sentiment_trends_{kind}_{timestamp}.png")
    plot_sentiment_trends_over_years(df_sentiment, save_path=plot_file)





