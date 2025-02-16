import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import os
from tqdm import tqdm
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import preprocess_files
from metadata_manager import get_metadata_by_file_id, TEXTS_PATH

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def preprocess_text(documents):
    """
    Preprocess a list of documents by tokenizing, lowercasing, removing stopwords, and lemmatizing
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    processed_docs = []
    for doc in tqdm(documents, desc="Preprocessing text"):
        doc = doc.lower()
        doc = preprocess_files.eliminate_words_from_text(doc)
        tokens = word_tokenize(doc.lower())  # Convert to lowercase and tokenize
        filtered_tokens = [
            lemmatizer.lemmatize(word)
            for word in tokens
            if word.isalpha() and word not in stop_words
        ]
        processed_docs.append(filtered_tokens)
    return processed_docs


def create_word_cloud(lda_model, num_topics):
    """
    Function to create a word cloud for each topic
    """
    for i in range(num_topics):
        # Extract words and their weights for the topic
        words = lda_model.show_topic(i, topn=50)  # Top 50 words for the topic
        word_freq = {word: weight for word, weight in words}

        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

        # Plot the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Topic {i + 1}")
        plt.show()


def generate_wordclouds_by_year(folder_path, output_before_1900="wordcloud_before_1900.png", output_after_1900="wordcloud_after_1900.png"):
    """
    Generate two word clouds from text files in a directory based on year metadata:
    - One for texts up to 1900
    - One for texts after 1900

    :param folder_path: str, path to the folder containing text files
    :param output_before_1900: str, file name to save the word cloud for texts before 1900
    :param output_after_1900: str, file name to save the word cloud for texts after 1900
    """
    text_before_1900 = []
    text_after_1900 = []

    # Process all files in the directory
    for file in tqdm(os.listdir(folder_path), desc="Loading texts"):
        if not file.endswith('.txt'):
            continue



        file_id = file.split('.')[0]
        file_data = get_metadata_by_file_id(file_id)
        file_year = file_data['year'].values[0] if not file_data.empty else None

        if file_year is None:
            continue

        file_year = int(file_year)

        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        if file_year <= 1900:
            text_before_1900.append(file_content)
        else:
            text_after_1900.append(file_content)

    # Preprocess text
    text_before_1900 = preprocess_text(text_before_1900)
    text_after_1900 = preprocess_text(text_after_1900)

    # Create dictionary and corpus for LDA
    dictionary_before_1900 = corpora.Dictionary(text_before_1900)
    corpus_before_1900 = [dictionary_before_1900.doc2bow(text) for text in text_before_1900]

    dictionary_after_1900 = corpora.Dictionary(text_after_1900)
    corpus_after_1900 = [dictionary_after_1900.doc2bow(text) for text in text_after_1900]

    # Train LDA models
    print("Starting LDA training...")
    lda_before_1900 = LdaModel(corpus=corpus_before_1900, id2word=dictionary_before_1900, num_topics=5, passes=10)
    lda_after_1900 = LdaModel(corpus=corpus_after_1900, id2word=dictionary_after_1900, num_topics=5, passes=10)

    # Extract topics and generate word clouds
    topics_before_1900 = lda_before_1900.show_topics(num_topics=5, num_words=50, formatted=False)
    topics_after_1900 = lda_after_1900.show_topics(num_topics=5, num_words=50, formatted=False)

    words_before_1900 = {word: weight for topic in topics_before_1900 for word, weight in topic[1]}
    words_after_1900 = {word: weight for topic in topics_after_1900 for word, weight in topic[1]}

    wordcloud_before_1900 = WordCloud(width=800, height=400, background_color='white', margin=0).generate_from_frequencies(
        words_before_1900)
    wordcloud_after_1900 = WordCloud(width=800, height=400, background_color='white', margin=0).generate_from_frequencies(
        words_after_1900)

    # Plot and save word clouds
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_before_1900, interpolation='bilinear')
    plt.axis("off")
    # plt.title("Word Cloud for Texts Before 1900")
    plt.savefig(output_before_1900, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_after_1900, interpolation='bilinear')
    plt.axis("off")
    # plt.title("Word Cloud for Texts After 1900")
    plt.savefig(output_after_1900, bbox_inches='tight')
    plt.show()

# Example usage
if __name__ == "__main__":
    folder_path = TEXTS_PATH
    folder_result = "saved_results"
    output_before_1900 = folder_result + "/wordcloud_before_1900.png"
    output_after_1900 = folder_result + "/wordcloud_after_1900.png"
    print(folder_path)
    generate_wordclouds_by_year(folder_path, output_before_1900=output_before_1900, output_after_1900=output_after_1900)


