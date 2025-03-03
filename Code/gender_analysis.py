import json
import os
from collections import defaultdict, Counter
from datetime import datetime
import re
import numpy as np
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


def count_gender_entities_in_document_keywords(text, chunk_size=1000000):
    """
    Optimized function to count occurrences of male and female gendered words in a document.

    Args:
        text (str): The input text document to be analyzed.
        chunk_size (int, optional): The size of text chunks for processing. Defaults to 1,000,000 characters.

    Returns:
        dict: A dictionary containing:
            - "male" (int): The total count of detected male-related words.
            - "female" (int): The total count of detected female-related words.
    """
    # Define gender-specific keywords for fallback and independent counting.
    gender_keywords_male = {"he", "him", "his", "man", "men", "boy", "gentleman", "gentlemen",
                            "father", "brother", "husband", "son", "sir"}
    gender_keywords_male.update({
        "himself", "lord", "lords", "prince", "king", "duke", "sir",
        "mister", "mr", "knight", "bachelor", "baron", "squire",
        "monk", "abbot", "patriarch", "emperor", "czar", "tsar",
        "groom", "nephew", "godfather", "stepfather", "grandfather",
        "stepbrother", "grandson", "uncle", "actor", "hero", "master",
        "chap", "lad", "fellow", "bloke", "gent"
    })
    gender_keywords_female = {"she", "her", "hers", "woman", "women", "girl", "lady", "ladies",
                              "mother", "sister", "wife", "daughter", "madam", "miss", "ms"}
    gender_keywords_female.update({
        "herself", "ladyship", "princess", "queen", "duchess", "mistress",
        "madame", "mademoiselle", "baroness", "nun", "abbess", "matriarch",
        "empress", "czarina", "tsarina", "bride", "niece", "godmother",
        "stepmother", "grandmother", "stepsister", "granddaughter", "aunt",
        "actress", "heroine", "maiden", "damsel", "lass", "belle", "girlhood",
        "spinster", "widow", "diva"
    })

    male_count = 0
    female_count = 0

    # Process the document in manageable chunks.
    for chunk in chunk_text(text, chunk_size=chunk_size):
        # Tokenize properly (handle punctuation, hyphens, etc.)
        words = re.findall(r"\b[\w'-]+\b", chunk.lower())

        # Normalize words (lemmatization)
        # words = [token.lemma_ for token in nlp(" ".join(words)) if token.is_alpha]

        # Use Counter for fast word frequency lookup
        word_counts = Counter(words)

        # Count occurrences of gender-related words
        male_count += sum(word_counts[word] for word in gender_keywords_male if word in word_counts)
        female_count += sum(word_counts[word] for word in gender_keywords_female if word in word_counts)

    return {"male": male_count, "female": female_count}


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

        gender_count_entities = count_gender_entities_in_document_keywords(text)
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


def count_gender_entities_bookMIA(output_file="gender_named_entities_bookMIA.json"):
    """
    Find named entities in text files in a folder and save the counts by decade to a JSON file.
    """
    from datasets import load_dataset
    data_sources = [load_dataset("swj0419/BookMIA", split=f"train")]
    gender_entities = {"male": 0, "female": 0}
    gender_entities_by_label = defaultdict(lambda: {"male": 0, "female": 0})

    for data_source in data_sources:
        for text in tqdm(data_source, desc="Processing texts"):
            gender_count_entities = count_gender_entities_in_document_keywords(text["snippet"])
            gender_entities["male"] += gender_count_entities["male"]
            gender_entities["female"] += gender_count_entities["female"]
            gender_entities_by_label[str(text["label"])]["male"] += gender_count_entities["male"]
            gender_entities_by_label[str(text["label"])]["female"] += gender_count_entities["female"]

    # Save the results to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(gender_entities, f, ensure_ascii=False, indent=4)

    with open(output_file.replace(".json", "_by_label.json"), 'w', encoding='utf-8') as f:
        json.dump(gender_entities_by_label, f, ensure_ascii=False, indent=4)

    return gender_entities, gender_entities_by_label


def plot_gender_entities_by_label_comparison(bookMIA_entities, entities_by_decade, save_folder=None,
                                             dataset1_name="BookMIA", dataset2_name="Gutenberg", from_decade=1900):
    """
    Plots a grouped bar chart comparing the overall gender percentages (male and female)
    between:
      - BookMIA member records (bookMIA_entities["1"])
      - BookMIA non-member records (bookMIA_entities["0"])
      - A second dataset aggregated by decade (entities_by_decade) filtered for decades >= from_decade

    Args:
        bookMIA_entities (dict): A dictionary with keys "1" and "0" representing member and non-member results.
                                 Each value is a dict with keys "male" and "female" representing counts.
        entities_by_decade (dict): A nested dictionary with decade keys (e.g., "1980s") mapping to counts:
                                   { "male": int, "female": int }.
        save_folder (str, optional): If provided, the plot is saved to this folder as a PNG file.
        dataset1_name (str, optional): Name for the first dataset (BookMIA). Defaults to "BookMIA".
        dataset2_name (str, optional): Name for the second dataset (e.g., Gutenberg). Defaults to "Gutenberg".
        from_decade (int, optional): The starting decade (e.g., 1900) from which to include counts from entities_by_decade.
    """
    # Split the results into member and non-member groups.
    member_results = bookMIA_entities["1"]
    non_member_results = bookMIA_entities["0"]

    # Compute overall percentages for BookMIA member records.
    member_total = member_results.get("male", 0) + member_results.get("female", 0)
    if member_total == 0:
        member_male_percent = member_female_percent = 0
    else:
        member_male_percent = (member_results.get("male", 0) / member_total) * 100
        member_female_percent = (member_results.get("female", 0) / member_total) * 100

    # Compute overall percentages for BookMIA non-member records.
    non_member_total = non_member_results.get("male", 0) + non_member_results.get("female", 0)
    if non_member_total == 0:
        non_member_male_percent = non_member_female_percent = 0
    else:
        non_member_male_percent = (non_member_results.get("male", 0) / non_member_total) * 100
        non_member_female_percent = (non_member_results.get("female", 0) / non_member_total) * 100

    # Compute overall percentages for the second dataset (entities_by_decade), filtering by from_decade.
    total_decade_male = 0
    total_decade_female = 0
    for decade in entities_by_decade:
        try:
            # Assuming the decade keys are strings like "1980s" (first 4 characters are the year)
            decade_year = int(decade[:4])
        except Exception:
            continue
        if decade_year >= from_decade:
            total_decade_male += entities_by_decade[decade].get("male", 0)
            total_decade_female += entities_by_decade[decade].get("female", 0)
    total_decade = total_decade_male + total_decade_female
    if total_decade == 0:
        decade_male_percent = decade_female_percent = 0
    else:
        decade_male_percent = (total_decade_male / total_decade) * 100
        decade_female_percent = (total_decade_female / total_decade) * 100

    # Prepare the data for plotting.
    # We use two categories: "Male" and "Female".
    # For each category, we have three bars corresponding to:
    # BookMIA Member, BookMIA Non-member, and the second dataset.
    categories = ["Male", "Female"]
    x = np.arange(len(categories))
    width = 0.25  # width of each bar

    # Percentages for each group.
    member_values = [member_male_percent, member_female_percent]
    non_member_values = [non_member_male_percent, non_member_female_percent]
    decade_values = [decade_male_percent, decade_female_percent]

    # Create the grouped bar chart.
    fig, ax = plt.subplots(figsize=(8, 6))
    bars_decade = ax.bar(x + width, decade_values, width, label=dataset2_name, color="lightgreen")
    bars_member = ax.bar(x - width, member_values, width, label=f"{dataset1_name} (Member)", color="steelblue")
    bars_non_member = ax.bar(x, non_member_values, width, label=f"{dataset1_name} (Non-member)", color="darkorange")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)")
    # ax.set_title("Gender Entities by Label Comparison")
    ax.legend()

    # Function to annotate the bars with percentage values.
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bars_member)
    autolabel(bars_non_member)
    autolabel(bars_decade)

    plt.tight_layout()

    # Save the plot if a folder is provided.
    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_folder, f"gender_entities_by_label_comparison_{time_str}.png")
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()

if __name__ == "__main__":
    folder_path = "saved_results"
    os.makedirs(folder_path, exist_ok=True)
    filename = f"{folder_path}/gender_named_entities_gender_keywords.json"
    # entities_by_decade = find_named_entities_in_folder(TEXTS_PATH, filename)
    with open(filename, "r") as file:
        entities_by_decade = json.load(file)
    plot_gender_entities_by_decade(entities_by_decade, save_folder=folder_path)
    plot_gender_entities_percentage_by_decade(entities_by_decade, save_folder=folder_path)

    # compare to BookMIA dataset
    output_boolMIA = f"{folder_path}/gender_named_entities_BookMIA.json"
    entities_bookMIA, gender_entities_by_label = count_gender_entities_bookMIA(output_boolMIA)
    # with open(output_boolMIA, "r") as file:
    #     entities_bookMIA = json.load(file)
    # with open(output_boolMIA.replace(".json", "_by_label.json"), "r") as file:
    #     gender_entities_by_label = json.load(file)
    plot_gender_entities_by_label_comparison(gender_entities_by_label, entities_by_decade, save_folder=folder_path)