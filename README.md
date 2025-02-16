# LexiGutenberg

LexiGutenberg is a comprehensive analysis project utilizing **Project Gutenberg** texts to examine literary trends, gender representation, sentiment evolution, and thematic shifts using **NLP** and **machine learning** techniques.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Jupyter Notebook](#jupyter-notebook)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

## Introduction
This project explores the evolution of literature using the **Project Gutenberg dataset**, employing methods like **topic modeling, gender analysis, sentiment classification, and genre identification**. The study reveals historical and thematic trends, including shifts in dominant themes and representation biases over time.

## Features
- **Metadata Processing:** Efficient extraction and filtering of book metadata.
- **Topic Modeling:** Uses **BERTopic** and **LDA** to analyze dominant themes across time.
- **Gender Analysis:** Extracts gender-related entities to track representation trends.
- **Genre Classification:** Applies pre-trained **XLM-RoBERTa** for genre labeling.
- **Sentiment Analysis:** Uses **VADER** and **TextBlob** to examine sentiment variations.
- **Visualization:** Generates word clouds, topic maps, and sentiment trends.

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/eyalgerman/LexiGutenberg.git
   cd LexiGutenberg
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download necessary NLP models:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('stopwords')
   nltk.download('vader_lexicon')
   ```

## Project Structure

```
LexiGutenberg/
├── data/                     # Project Gutenberg dataset (subset used)
│   ├── metadata.csv           # Metadata file containing book details
├── sw_jockers.txt             # Stopword file from "https://www.matthewjockers.net/2013/04/12/secret-recipe-for-topic-modeling-themes/"
├── results/                   # Generated outputs (graphs, CSVs, JSONs)
├── code/
│   ├── metadata_manager.py       # Handles metadata extraction & filtering
│   ├── lda_analysis.py           # LDA-based topic modeling & word clouds
│   ├── bert_topic_modeling.py    # BERTopic-based topic modeling & clustering
│   ├── gender_analysis.py        # Gender entity recognition & analysis
│   ├── sentiment_analysis.py     # Sentiment analysis using VADER & TextBlob
│   ├── main.ipynb                # Jupyter Notebook for interactive analysis
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

## Jupyter Notebook
The project also includes a **Jupyter Notebook** (`main.ipynb`) that provides an interactive environment for running analyses and visualizations. It allows users to:
- Load and preprocess the **Project Gutenberg dataset**.
- Perform **topic modeling** using **LDA and BERTopic**.
- Analyze **gender representation** over time.
- Classify books into genres with a pre-trained **XLM-RoBERTa model**.
- Conduct **sentiment analysis** using **VADER and TextBlob**.
- Generate visualizations such as **word clouds, topic maps, and sentiment trends**.
- Easily modify parameters and rerun analyses without modifying code files directly.


## Results
### Key Findings:
1. **Literary Evolution:** Pre-1900 texts focused on hierarchy, religion, and formal themes, while post-1900 literature shifted toward individualism and modern narratives.
2. **Gender Representation:** Male mentions overwhelmingly dominate historical texts, with female mentions rarely exceeding **10%**.
3. **Genre Trends:** Informational content has become the most dominant genre over time.
4. **Sentiment Analysis:** Most texts maintain a neutral tone, with occasional spikes reflecting major historical and literary shifts.

## Future Work
- Expanding multilingual comparisons.
- Enhancing **NER-based** gender recognition.
- Exploring **literary influence networks**.
- Incorporating **spatial analysis** for geographic trends in literature.

## Acknowledgments
Special thanks to **Michael Fire** for guidance. This project leverages **Project Gutenberg**, **spaCy**, **BERTopic**, **NLTK**, and **Hugging Face Transformers**.
