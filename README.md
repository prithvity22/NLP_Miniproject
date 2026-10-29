# NLP Mini Project: Text Cleaner and Keyword Finder

## Objective
Clean a collection of short English texts and identify frequent words and important keywords using TF-IDF.

## Tools
Python, Pandas, NLTK, Scikit-learn, Matplotlib.

## Dataset
`dataset.csv` contains 139 short text samples in a column named `text`. You can replace these with 100–200 news headlines or SMS messages, keeping the same column name.

## How to run in Google Colab
1. Upload `dataset.csv` and `main.py` to the Colab session, or upload the whole ZIP and extract it.
2. Install dependencies in a code cell:
   `!pip install pandas matplotlib nltk scikit-learn`
3. Run the script:
   `!python main.py`
4. Download/view the generated CSV and PNG files from the Files panel.

## How to run in VS Code
1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. In the terminal, run:
   `python -m pip install -r requirements.txt`
4. Run:
   `python main.py`

Internet is needed on the first run so NLTK can download its English stop-word list.

## Project workflow
1. Load text data from CSV.
2. Convert text to lowercase.
3. Remove punctuation, digits, and non-English characters.
4. Remove English stop words (e.g., “the”, “is”, “and”).
5. Apply Porter stemming (e.g., “learning” becomes “learn”).
6. Count all remaining words and plot the top 10.
7. Calculate TF-IDF scores and rank the top 10 terms by average score across documents.

## Output files
- `cleaned_texts.csv`: original text and cleaned text.
- `top_10_common_words.png`: bar chart of the 10 most frequent cleaned/stemmed words.
- `top_10_tfidf_keywords.csv`: top 10 terms with mean TF-IDF scores.
- `top_10_tfidf_keywords.png`: chart of the top TF-IDF terms.

## Important interpretation
Word frequency and TF-IDF are different. Frequency shows which words appear most often in the whole collection. TF-IDF gives more weight to terms that are informative in a document and less weight to terms that occur across many documents. The script ranks terms by their mean TF-IDF score across all documents; this is a collection-level summary, not a separate keyword list for every document.

## Limitations
- The sample dataset is for demonstrating the workflow and is not a balanced or representative research corpus.
- Porter stemming can produce word stems that are not full dictionary words.
- The cleaner is designed for simple English text and is not intended for multilingual data or advanced language understanding.

