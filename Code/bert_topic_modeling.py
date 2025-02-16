import os
from datetime import datetime
from bertopic import BERTopic
from collections import defaultdict
import json
from sentence_transformers import SentenceTransformer
import metadata_manager
from metadata_manager import TEXTS_PATH
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def load_texts_from_folder(folder_path):
    """
    Load all text files from a folder.
    """
    texts = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                texts.append(f.read())
    return texts


def fit_bertopic_model(texts, language="english", embedding_model="all-MiniLM-L6-v2"):
    """
    Initialize and fit a BERTopic model to the texts.
    """
    topic_model = BERTopic(language=language, embedding_model=embedding_model, calculate_probabilities=True, verbose=True)
    topics, probs = topic_model.fit_transform(texts)
    return topic_model, topics, probs


def save_document_topics(texts, topics, output_file="document_topics.json"):
    """
    Save the mapping of documents to their topics.
    """
    document_topics = [{"text": text, "topic": topic} for text, topic in zip(texts, topics)]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(document_topics, f, ensure_ascii=False, indent=4)


def group_topics_by_metadata_decade(folder_path, topics):
    """
    Group topics by metadata
    """
    topics_by_decade = defaultdict(list)

    for file, topic in zip(os.listdir(folder_path), topics):
        if not file.endswith(".txt"):
            continue

        file_id = file.split('.')[0]
        file_data = metadata_manager.get_metadata_by_file_id(file_id)
        file_year = file_data['year'].values[0] if not file_data.empty else None
        decade = (file_year // 10 * 10) if file_year else "Unknown"
        # decade = (file_year // 10 + 1) if file_year else "Unknown"
        topics_by_decade[decade].append(topic)

    return topics_by_decade


def display_grouped_topics(topics_by_century):
    """
    Display topics grouped by metadata.
    """
    for century, century_topics in topics_by_century.items():
        print(f"Century: {century}")
        print(f"Topics: {set(century_topics)}")


def visualize_topics(topic_model, save_path=None):
    """
    Visualize the topics using BERTopic's built-in tools and optionally save the plot.
    """
    fig = topic_model.visualize_topics()
    if save_path:
        fig.write_html(f"{save_path}/topics_visualization.html")
    return fig


def visualize_hierarchy(topic_model, save_path=None):
    """
    Visualize the hierarchical structure of topics and optionally save the plot.
    """
    fig = topic_model.visualize_hierarchy()
    if save_path:
        fig.write_html(f"{save_path}/hierarchy_visualization.html")
    return fig


def visualize_distribution(topic_model, probs, document_index=0, save_path=None):
    """
    Visualize the topic probabilities for a specific document and optionally save the plot.
    """
    fig = topic_model.visualize_distribution(probs[document_index])
    if save_path:
        fig.write_html(f"{save_path}/distribution_visualization_doc_{document_index}.html")
    return fig



def save_model(topic_model, path="bertopic_model"):
    """
    Save the BERTopic model to disk.
    """
    topic_model.save(path)


def load_model(path="bertopic_model"):
    """
    Load a BERTopic model from disk.
    """
    return BERTopic.load(path)


def compute_topic_matrix(topics_by_decade):
    """
    Create a topic distribution matrix where rows represent decades
    and columns represent topic frequencies.
    """
    all_topics = sorted(set(topic for topics in topics_by_decade.values() for topic in topics))
    decades = sorted(topics_by_decade.keys())

    # Create a matrix where each row corresponds to a decade
    topic_matrix = []
    for decade in decades:
        row = [topics_by_decade[decade].count(topic) for topic in all_topics]
        topic_matrix.append(row)

    return np.array(topic_matrix), decades, all_topics


def plot_distance_clustering_map(topic_matrix, decades, save_dir=None):
    """
    Perform hierarchical clustering and visualize the distance map using t-SNE.
    """
    # Perform clustering
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.0, linkage="ward")
    labels = clustering_model.fit_predict(topic_matrix)

    # Perform dimensionality reduction (t-SNE)
    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    reduced_data = tsne.fit_transform(topic_matrix)

    # Plot the results
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        x=reduced_data[:, 0],
        y=reduced_data[:, 1],
        hue=labels,
        palette="tab10",
        s=100,
        alpha=0.8,
        # legend="full"
        legend=False
    )

    # Annotate points with decade labels
    for i, decade in enumerate(decades):
        plt.text(reduced_data[i, 0], reduced_data[i, 1], str(decade), fontsize=9, ha='right')

    # plt.title("Distance Clustering Map of Topics Across Decades")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    # plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    if save_dir:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"distance_clustering_map_{date_str}.png")
        plt.savefig(save_path)
    plt.show()



def run_bertopic_workflow(folder_path, output_folder="saved_results"):
    """
    Run the entire BERTopic workflow.
    """
    texts = load_texts_from_folder(folder_path)
    model = "sentence-transformers/all-distilroberta-v1"
    print(f"Embedding model: {model}")
    sentence_model = SentenceTransformer(model)
    topic_model, topics, probs = fit_bertopic_model(texts, embedding_model=SentenceTransformer)
    save_document_topics(texts, topics)
    topics_by_decade = group_topics_by_metadata_decade(folder_path, topics)
    # Compute topic matrix
    topic_matrix, decades, all_topics = compute_topic_matrix(topics_by_decade )
    # Plot distance clustering map
    plot_distance_clustering_map(topic_matrix, decades, save_dir=output_folder)
    display_grouped_topics(topics_by_decade)
    # Visualize and save
    visualize_topics(topic_model, save_path=output_folder)
    visualize_hierarchy(topic_model, save_path=output_folder)
    visualize_distribution(topic_model, probs, document_index=0, save_path=output_folder)
    # visualize_topics(topic_model)
    save_model(topic_model)


if __name__ == "__main__":
    # text_folder = "../gutenberg_books_train"
    folder_path = "saved_results"
    os.makedirs(folder_path, exist_ok=True)
    run_bertopic_workflow(TEXTS_PATH, folder_path)
