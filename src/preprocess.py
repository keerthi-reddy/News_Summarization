import pandas as pd
import re
import contractions
from keras.preprocessing.text import Tokenizer
import pickle

def load_data(train_path, test_path, validation_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(validation_path)
    return train_df, test_df, val_df

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def expand_contractions(text):
    return contractions.fix(text)

def preprocess_data(df):
    df['article'] = df['article'].apply(clean_text).apply(expand_contractions)
    df['highlights'] = df['highlights'].apply(clean_text).apply(expand_contractions)
    df['highlights'] = ['<start> ' + sentence + ' <end>' for sentence in df['highlights']]
    return df

def fit_tokenizer(train_articles, train_highlights):
    tok = Tokenizer()
    tok.fit_on_texts(train_articles + train_highlights)
    return tok

def save_tokenizer(tok, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(tok, f)

def load_tokenizer(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
