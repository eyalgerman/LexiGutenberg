from transformers import pipeline
import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import metadata_manager
import matplotlib.pyplot as plt
from collections import Counter
import torch

from metadata_manager import TEXTS_PATH


def classify_genre_in_document(text, model_name="classla/xlm-roberta-base-multilingual-text-genre-classifier"):
    """
    Classify the genre of a given text using a pretrained model.
    """
    # Check for GPU availability
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model=model_name, device=device)
    model = pipe.model
    # id2label = model.config.id2label  # Get the mapping from label IDs to human-readable labels
    result = pipe(text, truncation=True, max_length=512)  # Truncate text if too long
    predicted_label = result[0]['label']
    # print(f"Predicted label: {predicted_label}")
    # Convert the label to the actual genre name using id2label
    # label_txt = id2label[int(predicted_label.split("_")[-1])]
    return predicted_label


def genre_classification_on_folder(folder_path, folder_dir, output_file="genre_classification_by_year.json"):
    """
    Classify the genre of documents in a folder and save the results to a JSON file in the specified directory.
    """
    genres_by_year = {}

    # Ensure the output directory exists
    os.makedirs(folder_dir, exist_ok=True)

    output_file_path = os.path.join(folder_dir, output_file)

    for file in tqdm(os.listdir(folder_path), desc="Classifying genres"):
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

        genre = classify_genre_in_document(text)
        # print(f"File: {file}, Genre: {genre}")

        if file_year not in genres_by_year:
            genres_by_year[file_year] = []

        genres_by_year[file_year].append(genre)

    # Save the results to a JSON file in the output directory
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(genres_by_year, f, ensure_ascii=False, indent=4)

    return genres_by_year

# Define a consistent color mapping
COLOR_MAPPING = {
    "Male": "#1f77b4",   # Blue
    "Female": "#ff7f0e", # Orange
    "Forum": "#1f77b4",
    "Legal": "#ff7f0e",
    "News": "#2ca02c",
    "Other": "#d62728",
    "Promotion": "#9467bd",
    "Opinion/Argumentation": "#8c564b",
    "Information/Explanation": "#e377c2",
    "Prose/Lyrical": "#bcbd22",
    "Instruction": "#17becf"
}

def plot_genre_distribution_bar(genres_by_year, folder_dir, timestamp=""):
    """
    Plot the distribution of genres across all files using a bar chart and save it to the specified directory.
    """
    all_genres = []
    for year, genres in genres_by_year.items():
        all_genres.extend(genres)

    genre_counts = Counter(all_genres)
    genres = list(genre_counts.keys())
    counts = list(genre_counts.values())

    # Plotting bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(genres, counts)
    plt.xlabel("Genres")
    plt.ylabel("Number of Documents")
    plt.title("Genre Distribution Across All Files (Bar Chart)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot
    os.makedirs(folder_dir, exist_ok=True)
    bar_chart_path = os.path.join(folder_dir, f"genre_distribution_bar_{timestamp}.png")
    plt.savefig(bar_chart_path)
    plt.show()


def plot_genre_distribution_pie(genres_by_year, folder_dir, timestamp):
    """
    Plot the distribution of genres across all files using a pie chart and save it to the specified directory.
    """
    all_genres = []
    for year, genres in genres_by_year.items():
        all_genres.extend(genres)

    genre_counts = Counter(all_genres)
    genres = list(genre_counts.keys())
    counts = list(genre_counts.values())
    pie_colors = [COLOR_MAPPING[genre] for genre in genre_counts.keys()]

    # Plotting pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(counts, labels=genres, autopct='%1.1f%%', startangle=140, colors=pie_colors)
    # plt.title("Genre Distribution Across All Files (Pie Chart)")
    plt.tight_layout()

    # Save the plot
    os.makedirs(folder_dir, exist_ok=True)
    pie_chart_path = os.path.join(folder_dir, f"genre_distribution_pie_{timestamp}.png")
    plt.savefig(pie_chart_path)
    plt.show()


def plot_genre_distribution_by_decade(genres_by_year, folder_dir, timestamp):
    """
    Plot the distribution of genres for each decade using a stacked bar chart,
    with different colors for each genre, and save it to the specified directory.
    """
    # Helper function to calculate the decade from a year
    def get_decade(year):
        return (year // 10) * 10

    # Group genres by decade
    genres_by_decade = {}
    for year, genres in genres_by_year.items():
        decade = get_decade(int(year))  # Convert year to integer and find its decade
        if decade not in genres_by_decade:
            genres_by_decade[decade] = []
        genres_by_decade[decade].extend(genres)

    # Extract all unique genres and decades
    decades = sorted(genres_by_decade.keys())
    all_genres = set(genre for decade_genres in genres_by_decade.values() for genre in decade_genres)

    # Create a dictionary to count genres for each decade
    genre_counts_by_decade = {decade: Counter(genres_by_decade[decade]) for decade in decades}

    # Prepare data for the stacked bar chart
    genre_data = {genre: [genre_counts_by_decade[decade].get(genre, 0) for decade in decades] for genre in all_genres}

    # Plotting
    plt.figure(figsize=(10, 5))
    bar_width = 5
    bottom = np.zeros(len(decades))

    # Assign different colors for each genre
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_genres)))

    for genre, color in zip(genre_data.keys(), colors):
        plt.bar(decades, genre_data[genre], bottom=bottom, label=genre, color=color, width=bar_width, alpha=0.8)
        bottom += genre_data[genre]  # Update bottom for the next genre

    # Add labels and title
    plt.xlabel("Decade")
    plt.ylabel("Number of Documents")
    plt.title("Genre Distribution by Decade (Stacked Bar Chart)")
    plt.xticks(decades, [f"{decade}s" for decade in decades], rotation=45, ha='right')
    plt.legend(title="Genres", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the plot
    os.makedirs(folder_dir, exist_ok=True)
    stacked_bar_chart_path = os.path.join(folder_dir, f"genre_distribution_by_decade_{timestamp}.png")
    plt.savefig(stacked_bar_chart_path)
    plt.show()


def plot_genre_distribution_percentage_by_decade(genres_by_year, folder_dir, timestamp):
    """
    Plot the percentage distribution of genres for each decade using a stacked bar chart,
    with different colors for each genre, and save it to the specified directory.
    """
    # Helper function to calculate the decade from a year
    def get_decade(year):
        return (year // 10) * 10

    # Group genres by decade
    genres_by_decade = {}
    for year, genres in genres_by_year.items():
        decade = get_decade(int(year))  # Convert year to integer and find its decade
        if decade not in genres_by_decade:
            genres_by_decade[decade] = []
        genres_by_decade[decade].extend(genres)

    # Extract all unique genres and decades
    decades = sorted(genres_by_decade.keys())
    all_genres = set(genre for decade_genres in genres_by_decade.values() for genre in decade_genres)

    # Create a dictionary to count genres for each decade
    genre_counts_by_decade = {decade: Counter(genres_by_decade[decade]) for decade in decades}

    # Compute percentages per decade
    genre_percentage_by_decade = {}
    for decade in decades:
        total_docs = sum(genre_counts_by_decade[decade].values())
        genre_percentage_by_decade[decade] = {
            genre: (count / total_docs) * 100 if total_docs > 0 else 0
            for genre, count in genre_counts_by_decade[decade].items()
        }

    # Convert data into structured format for plotting
    genre_data = {
        genre: [genre_percentage_by_decade[decade].get(genre, 0) for decade in decades]
        for genre in all_genres
    }

    # Plotting
    plt.figure(figsize=(12, 5))
    bar_width = 8
    bottom = np.zeros(len(decades))

    # Assign different colors for each genre
    # colors = plt.cm.tab20(np.linspace(0, 1, len(all_genres)))
    colors = [COLOR_MAPPING[genre] for genre in all_genres]

    for genre, color in zip(genre_data.keys(), colors):
        plt.bar(decades, genre_data[genre], bottom=bottom, label=genre, color=color, width=bar_width, alpha=0.8)
        bottom += genre_data[genre]  # Update bottom for the next genre

    # Add labels and title
    plt.xlabel("Decade")
    plt.ylabel("Percentage of Documents (%)")
    plt.title("Genre Distribution by Decade (Stacked Bar Chart - Percentage)")
    plt.xticks(decades, [f"{decade}s" for decade in decades], rotation=45, ha='right')
    plt.ylim(0, 100)  # Ensure the Y-axis represents percentages (0% to 100%)
    plt.legend(title="Genres", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the plot
    os.makedirs(folder_dir, exist_ok=True)
    stacked_bar_chart_path = os.path.join(folder_dir, f"genre_distribution_percentage_by_decade_{timestamp}.png")
    plt.savefig(stacked_bar_chart_path)
    plt.show()

def main(tests_path, output_dir, use_existing_data=False):
    """
    Main function to execute the workflow.
    """
    output_file = "genre_classification_by_year.json"

    print("\nClassifying genres in documents...")
    if not use_existing_data:
        genres_by_year = genre_classification_on_folder(tests_path, output_dir, output_file)
    else:
        with open(os.path.join(output_dir, output_file), 'r', encoding='utf-8') as f:
            genres_by_year = json.load(f)

    # print("\nClassification completed! Results saved to:", os.path.join(output_dir, output_file))

    print("\nPlotting genre distribution...")
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    # plot_genre_distribution_bar(genres_by_year, output_dir, timestamp)
    plot_genre_distribution_pie(genres_by_year, output_dir, timestamp)
    # plot_genre_distribution_by_decade(genres_by_year, output_dir, timestamp)
    plot_genre_distribution_percentage_by_decade(genres_by_year, output_dir, timestamp)

    # print("\nPlots saved in:", output_dir)


# Run the main function
if __name__ == "__main__":
    output_dir = "saved_results"
    tests_path = TEXTS_PATH
    main(tests_path, output_dir, use_existing_data=True)
