import json
import os
from collections import defaultdict
from datetime import datetime

import spacy
from matplotlib import pyplot as plt
from tqdm import tqdm

import metadata_manager
from metadata_manager import TEXTS_PATH
# from named_entity_recognition import chunk_text, nlp
import en_core_web_lg

nlp = en_core_web_lg.load()

def chunk_text(text, chunk_size=100000):
    """
    Split text into smaller chunks for processing.
    """
    for i in range(0, len(text), chunk_size):
        yield text[i:i+chunk_size]


def count_gender_entities_in_document(text, chunk_size=1000000):
    """
    Analyzes a given text document to count the occurrences of male and female entities
    based on named entity recognition (NER) and surrounding gender-related keywords.

    Args:
        text (str): The input text document to be analyzed.
        chunk_size (int, optional): The size of text chunks for processing. Defaults to 1,000,000 characters.
                                    Splitting the text into chunks helps manage memory usage when processing large documents.

    Returns:
        dict: A dictionary containing:
            - "male" (int): The number of detected male entities.
            - "female" (int): The number of detected female entities.
            - "male_names" (list of str): The list of detected male entity names.
            - "female_names" (list of str): The list of detected female entity names.
    """
    gender_keywords_male = {"he", "him", "his", "man", "men", "boy", "gentleman", "gentlemen", "father", "brother", "husband", "son", "sir"}
    gender_keywords_female = {"she", "her", "hers", "woman", "women", "girl", "lady", "ladies", "mother", "sister", "wife", "daughter", "madam", "miss", "ms"}

    male_count = 0
    female_count = 0
    male_names = []
    female_names = []

    for chunk in chunk_text(text, chunk_size=chunk_size):
        doc = nlp(chunk)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Check context for gender-specific keywords
                offset = 5
                surrounding_text = doc[max(0, ent.start-offset):min(len(doc), ent.end+offset)].text.lower()
                if any(keyword in surrounding_text for keyword in gender_keywords_male) or ent.text.lower in male_names:
                    male_count += 1
                    male_names.append(ent.text)
                elif any(keyword in surrounding_text for keyword in gender_keywords_female) or ent.text.lower in female_names:
                    female_count += 1
                    female_names.append(ent.text)

    return {"male": male_count, "female": female_count, "male_names": male_names, "female_names": female_names}


def find_named_entities_in_folder(folder_path, output_file="gender_named_entities.json"):
    """
    Find named entities in text files in a folder and save the counts by decade to a JSON file.
    """
    entities_by_decade = defaultdict(lambda: {"male": 0, "female": 0})

    for file in tqdm(os.listdir(folder_path)):
        if not file.endswith('.txt'):
            continue

        file_id = file.split('.')[0]
        file_data = metadata_manager.get_metadata_by_file_id(file_id)
        file_year = file_data['year'].values[0] if not file_data.empty else None

        if file_year is None:
            continue

        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        gender_count_entities = count_gender_entities_in_document(text)
        # gender_count_entities = count_gender_entities_with_gender_spacy(text)

        # Determine the decade
        decade = (file_year // 10) * 10

        decade = str(int(decade))

        # Aggregate counts for the decade
        entities_by_decade[decade]["male"] += gender_count_entities["male"]
        entities_by_decade[decade]["female"] += gender_count_entities["female"]

    # Save the results to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(entities_by_decade, f, ensure_ascii=False, indent=4)

    return entities_by_decade


def plot_gender_entities_by_decade(entities_by_decade, save_folder=None):
    """
    Plot the counts of male and female entities by decade.
    """
    # Convert the keys to numeric values (assuming they represent decades as strings like "1990", "2000", etc.)
    decades = sorted(int(decade) for decade in entities_by_decade.keys())
    male_counts = [entities_by_decade[str(decade)]["male"] for decade in decades]
    female_counts = [entities_by_decade[str(decade)]["female"] for decade in decades]

    # decades = sorted(entities_by_decade.keys())
    # male_counts = [entities_by_decade[decade]["male"] for decade in decades]
    # female_counts = [entities_by_decade[decade]["female"] for decade in decades]

    plt.figure(figsize=(12, 6))
    width = 5  # Bar width

    # Plot male and female counts side by side
    plt.bar([d - width / 2 for d in decades], male_counts, width=width, label="Male", alpha=0.7)
    plt.bar([d + width / 2 for d in decades], female_counts, width=width, label="Female", alpha=0.7)

    # Add labels and title
    plt.xlabel("Decade")
    plt.ylabel("Entity Count")
    # plt.title("Gender Entity Counts by Decade")
    plt.xticks(decades, rotation=45)
    plt.legend()
    plt.tight_layout()
    if save_folder:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_folder, f"gender_entities_by_decade_{time_str}.png")
        plt.savefig(save_path)
    plt.show()


def plot_gender_entities_percentage_by_decade(entities_by_decade, save_folder=None):
    """
    Plot the percentages of male and female entities by decade.
    """
    decades = sorted(int(decade) for decade in entities_by_decade.keys())
    male_counts = [entities_by_decade[str(decade)]["male"] for decade in decades]
    female_counts = [entities_by_decade[str(decade)]["female"] for decade in decades]

    # decades = sorted(entities_by_decade.keys())
    # male_counts = [entities_by_decade[decade]["male"] for decade in decades]
    # female_counts = [entities_by_decade[decade]["female"] for decade in decades]

    # Calculate percentages
    total_counts = [male + female for male, female in zip(male_counts, female_counts)]
    male_percentages = [(male / total) * 100 if total > 0 else 0 for male, total in zip(male_counts, total_counts)]
    female_percentages = [(female / total) * 100 if total > 0 else 0 for female, total in zip(female_counts, total_counts)]

    plt.figure(figsize=(9, 3))
    width = 5  # Bar width

    # Plot male and female percentages side by side
    plt.bar([d - width / 2 for d in decades], male_percentages, width=width, label="Male (%)", alpha=0.7)
    plt.bar([d + width / 2 for d in decades], female_percentages, width=width, label="Female (%)", alpha=0.7)

    # Add labels and title
    plt.xlabel("Decade")
    plt.ylabel("Percentage (%)")
    # plt.title("Gender Entity Percentages by Decade")
    plt.xticks(decades, rotation=45)
    plt.ylim(0, 100)  # Ensure the y-axis goes from 0 to 100
    plt.legend()
    plt.tight_layout()

    if save_folder:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_folder, f"gender_entities_percentage_by_decade_{time_str}.png")
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    folder_path = "saved_results"
    os.makedirs(folder_path, exist_ok=True)
    filename = f"{folder_path}/gender_named_entities_gender_spacy.json"
    # entities_by_decade = find_named_entities_in_folder(TEXTS_PATH, filename)
    with open(filename, "r") as file:
        entities_by_decade = json.load(file)
    plot_gender_entities_by_decade(entities_by_decade, save_folder=folder_path)
    plot_gender_entities_percentage_by_decade(entities_by_decade, save_folder=folder_path)