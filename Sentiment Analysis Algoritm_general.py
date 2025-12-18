import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from textblob import TextBlob
from textblob import Word

# Load the dataframe of tweets
df = pd.read_csv('Scrapped and Cleaned.csv')

# Define a function to preprocess the text of each tweet
def preprocess_text(text):
    # Create a TextBlob object from the tweet text
    blob = TextBlob(text)
    
    # Correct spelling errors using the TextBlob `correct` method
    corrected_blob = blob.correct()
    
    # Extract noun phrases using the TextBlob `noun_phrases` property
    noun_phrases = corrected_blob.noun_phrases
    
    # Tokenize the corrected text
    tokens = word_tokenize(str(corrected_blob))
    
    # Convert all letters to lowercase
    tokens = [token.lower() for token in tokens]
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stem or lemmatize the words
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Join the tokens back into a string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text, noun_phrases

# Apply the preprocessing function to each tweet in the dataframe
preprocessed_texts = []
noun_phrases_list = []
for tweet in tqdm(df['full_text']):
    preprocessed_text, noun_phrases = preprocess_text(tweet)
    preprocessed_texts.append(preprocessed_text)
    noun_phrases_list.append(noun_phrases)
df['preprocessed_text'] = preprocessed_texts
df['noun_phrases'] = noun_phrases_list

# Print the dataframe with the preprocessed text and noun phrases columns
print(df)
