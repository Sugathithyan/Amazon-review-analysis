# AMAZON FAKE REVIEW ANALYSIS    
## Overview

The Fake Review Detection project uses natural language processing (NLP) and machine learning techniques to classify Amazon reviews as genuine or fake. This script preprocesses the text data, vectorizes it, and applies a Random Forest classifier to predict the authenticity of the reviews. The dataset is an Excel file containing Amazon reviews, and the script involves data cleaning, visualization, feature extraction, and model training.

## Libraries Used

- **Numpy**: For numerical operations.
- **NLTK**: For natural language processing, including tokenization, stemming, and stopwords.
- **String**: For string operations.
- **BeautifulSoup (bs4)**: For web scraping (though not used in the provided code).
- **re**: For regular expressions.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For plotting graphs.
- **Seaborn**: For statistical data visualization.
- **Scikit-learn**: For machine learning, including vectorization, model training, and evaluation.
- **Ipywidgets**: For creating interactive widgets in Jupyter notebooks.

## Features

- **Data Loading**: Reads Amazon review data from an Excel file.
- **Data Cleaning**: Removes unnecessary columns and handles missing data.
- **Visualization**: Uses Seaborn for heatmaps, count plots, and bar plots to visualize data distribution and missing values.
- **Text Preprocessing**: Tokenizes and stems review text, removes punctuation and stopwords.
- **Feature Extraction**: Converts text data into a Bag of Words (BoW) model.
- **Interactive Widgets**: Allows selection of product categories using a combobox.
- **Model Training**: Trains a Random Forest classifier to detect fake reviews.
- **Model Evaluation**: Uses confusion matrix and classification report to evaluate the model.

## Installation Steps

1. **Install Required Libraries**:

   ```bash
   pip install numpy nltk beautifulsoup4 pandas matplotlib seaborn scikit-learn ipywidgets

2. **Download NLTK Data**:
    ```bash
    import nltk
   nltk.download('punkt')
   nltk.download('stopwords')

3.**Prepare Data Files**:
-Ensure the amazon_reviews.xlsx file is located at the specified path.
-Include necessary corpora and tokenizer files in appropriate directories.


-Execute the script in a Jupyter notebook or a Python environment
