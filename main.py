"""
NLP Mini Project: Text Cleaner and Keyword Finder
Run: python main.py
Input: dataset.csv (column name: text)
Outputs: cleaned_texts.csv, top_10_common_words.png,
         top_10_tfidf_keywords.csv, top_10_tfidf_keywords.png
"""
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK stop-word list on the first run.
nltk.download("stopwords", quiet=True)

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "dataset.csv"
OUTPUT_CLEANED = BASE_DIR / "cleaned_texts.csv"

def clean_text(text, stop_words, stemmer):
    """Lowercase, remove punctuation/numbers, remove stop words, and stem."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # keep English letters and spaces
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE.name}. Keep dataset.csv beside main.py."
        )

    data = pd.read_csv(INPUT_FILE)
    if "text" not in data.columns:
        raise ValueError("dataset.csv must contain a column named 'text'.")

    data = data.dropna(subset=["text"]).copy()
    data["text"] = data["text"].astype(str)
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    data["cleaned_text"] = data["text"].apply(
        lambda value: clean_text(value, stop_words, stemmer)
    )
    data.to_csv(OUTPUT_CLEANED, index=False, encoding="utf-8")
    print(f"Loaded {len(data)} texts.")
    print("\nSample cleaned text:")
    for original, cleaned in zip(data["text"].head(5), data["cleaned_text"].head(5)):
        print(f"Original: {original}\nCleaned:  {cleaned}\n")

    # Count words across all cleaned documents.
    all_words = " ".join(data["cleaned_text"]).split()
    if not all_words:
        raise ValueError("No words remain after cleaning. Check the dataset.")
    word_counts = pd.Series(all_words).value_counts().head(10).sort_values()
    print("Top 10 most common words:")
    print(word_counts.sort_values(ascending=False).to_string())

    plt.figure(figsize=(10, 6))
    word_counts.plot(kind="barh")
    plt.title("Top 10 Most Common Words (After Cleaning)")
    plt.xlabel("Frequency")
    plt.ylabel("Stemmed word")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "top_10_common_words.png", dpi=200)
    plt.close()

    # TF-IDF scores per document; average each term's score across documents.
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(data["cleaned_text"])
    terms = vectorizer.get_feature_names_out()
    mean_scores = matrix.mean(axis=0).A1
    keywords = (
        pd.DataFrame({"keyword": terms, "mean_tfidf_score": mean_scores})
        .sort_values("mean_tfidf_score", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    keywords.to_csv(BASE_DIR / "top_10_tfidf_keywords.csv", index=False)
    print("\nTop 10 TF-IDF keywords (highest average score):")
    print(keywords.to_string(index=False, formatters={"mean_tfidf_score": "{:.4f}".format}))

    plot_data = keywords.sort_values("mean_tfidf_score")
    plt.figure(figsize=(10, 6))
    plt.barh(plot_data["keyword"], plot_data["mean_tfidf_score"])
    plt.title("Top 10 Keywords by Average TF-IDF Score")
    plt.xlabel("Mean TF-IDF score")
    plt.ylabel("Keyword")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "top_10_tfidf_keywords.png", dpi=200)
    plt.close()

    print("\nDone! Check the project folder for CSV files and charts.")

if __name__ == "__main__":
    main()
