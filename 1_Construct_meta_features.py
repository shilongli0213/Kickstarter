# -*- coding: utf-8 -*-
"""
@author: Shilong

"""

# parse the scraped contents and construct features

# import necessary libaries to parse HTML for each funding project
# and make meta features
import pandas as pd
import nltk
nltk.download('punkt')
from bs4 import BeautifulSoup
import numpy as np
import re

# read saved data files to be parsed
scraped_all = pd.read_pickle('scraped_all.pkl')
df = pd.read_csv('df_0.csv')

# -----------------------------------------------------------------------------
# Parse and preprocess content for each project

def parse(scraped_piece):
    """Utilize BeautifulSoup libary to parse scraped HTML for each project
    
    Args:
        scraped_piece: the response object to be parsed  
    Returns:
        soup object with parse HTML
    """
    
    return BeautifulSoup(scraped_piece.text, 'lxml')

def clean_up(messy_text):   
    """Clean the text by removing unnecessary content
    
    Args:
        messy_text: the raw text   
    Returns:
        cleaned string with context
    """
    
    # Remove line breaks, whitespaces
    clean_text = ' '.join(messy_text.split()).strip() 
    # Remove the HTML5 warning for videos
    return clean_text.replace(
        "You'll need an HTML5 capable browser to see this content. " + \
        "Play Replay with sound Play with sound 00:00 00:00",'')   
        
def get_campaign(soup):
    """Extract campaign content
    
    Args:
        soup: soup object containing a kickstarter project page   
    Returns:
        a dictionary with cleaned project content
    """
    
    try:
        section1 = soup.find('div',
            class_='full-description js-full-description responsive-media formatted-lists').get_text(' ')
    except AttributeError:
        section1 = 'section_not_found'
           
    return {'campaign': clean_up(section1)}

def normalize(text):
    """ Normalize some contents such as email address, hyperlinks, money amounts,
        percentages, phone number and number into tags
        
    Args:
        text: clean context for each project
    Returns:
        normalized text"""
        
    # Tag email addresses with emailaddr using regex
    normalized = re.sub(
        r'\b[\w\-.]+?@\w+?\.\w{2,4}\b',
        'emailaddr',text) 
    # Tag hyperlinks with httpaddr using regex
    normalized = re.sub(
        r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)',
        'httpaddr',normalized) 
    # Tag money amounts with dollramt using regex
    normalized = re.sub(r'\$\d+(\.\d+)?', 'dollramt', normalized)
    # Tag percentages with percntg using regex
    normalized = re.sub(r'\d+(\.\d+)?\%', 'percntg', normalized)
    # Tag phone numbers with phonenumbr using regex
    normalized = re.sub(
        r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        'phonenumbr',normalized)
    # Tag remaining numbers with numbr using regex
    return re.sub(r'\d+(\.\d+)?', 'numbr', normalized)

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# feature engineering to build meta features
    
def get_sentences(text):
    """Get sentences using sentence tokenizer of NLTK library 
    
    Args:
        text (str): cleaned and normalized text of each project
    Returns:
        a list containing sentences"""
        
    # Tokenize the text into sentences
    return nltk.sent_tokenize(text)

def remove_punc(text):
    """ Remove all the punctuation in the text
    
    Args:
        text (str) cleaned and normalized text of each project
    Returns:
        a string without punctations"""
        
    # Remove punctuation with regex
    return re.sub(r'[^\w\d\s]|\_', ' ', text)

def get_words(text):
    """Tokenize the text into words
    
    Args:
        text (str): cleaned and normalized text of each project
    Returns:
        a list containing words for each project"""
        
    # Remove punctuation and then tokenize the text into words
    return [word for word in nltk.word_tokenize(remove_punc(text))]

def identify_allcaps(text):
    """Find all capital letters in each project
    
    Args:
        text (str): cleaned and normalized text of each project
    Returns:
        a list containing all-caps words"""
        
    # Identify all-caps words with regex
    return re.findall(r'\b[A-Z]{2,}', text)

def count_exclamations(text):
    """Count the number of exclamation marks
    Args:
        text (str): cleaned and normalized text of each project
    Returns:
        an integer number of exclamation marks"""
        
    # Count the number of exclamation marks in the text
    return text.count('!')

def count_buzz_words(text):
    """Count the number of innovation words in the text of each project
    
    Args:
        text (str): cleaned and normalized text of each project
    Returns:
        an interger number of buzz words in the context"""
    
    # Define a set of buzz words
    buzz_words = frozenset(
        ['revolutionary', 'breakthrough', 'beautiful', 'magical', 
        'gorgeous', 'amazing', 'incredible', 'awesome','data','intelligence'])
    return sum(1 for word in get_words(text) if word in buzz_words)

def compute_avg_words(text):
    """Count average number of words in each sentence from each project
    
    Args:
        text (str): cleaned and normalized text of each project
    Returns:
        a float number of the average number of words"""
    
    # Compute the average number of words in each sentence
    return pd.Series(
        [len(get_words(sentence)) for sentence in get_sentences(text)]
    ).mean()
    
def count_paragraphs(soup, section):  
    """Count the number of paragraph for each project
    
    Args:
        soup (soup object): parsed HTML content for each project
        section (str): context section
    Returns:
        an integer of number of paragraphs in each project"""
    
    # Count the number of paragraphs
    if section == 'campaign':
        return len(soup.find('div',
            class_='full-description js-full-description responsive-media ' + \
                'formatted-lists').find_all('p')) 
        
def count_images(soup, section):  
    """Count the number of images in each project
    
    Args:
        soup (soup object): parsed HTML content for each project
        section (str): context section
    Returns:
        an integer of number of images in each project"""
    
    # Use tree parsing to identify all image tags 
    if section == 'campaign':
        return len(soup.find('div',
            class_='full-description js-full-description responsive-media ' + \
                'formatted-lists').find_all('img')) 
        
def count_videos(soup, section):
    """Count the number of videos in each project
    
    Args:
        soup (soup object): parsed HTML content for each project
        section (str): context section
    Returns:
        an integer of number of videos in each project"""
    
    # Count all videos
    youtube_count = 0
    non_youtube_count = 0
    if section == 'campaign':
        non_youtube_count = len(soup.find('div',
            class_='full-description js-full-description responsive-media ' + \
                'formatted-lists').find_all('video-player')) 

        youtube = soup.find('div',
            class_='full-description js-full-description responsive-media ' + \
                'formatted-lists').find_all('iframe')
        
    for iframe in youtube:
        try:
            if 'youtube' in iframe.get('src'):
                youtube_count += 1
        except TypeError:
            pass
    return youtube_count+non_youtube_count


def count_gifs(soup, section):    
    """Count the number of gif images in each project
    
    Args:
        soup (soup object): parsed HTML content for each project
        section (str): context section
    Returns:
        an integer of number of gif images in each project"""
    
    gif_count = 0
    # Use tree parsing to select all image tags depending on the section
    # requested
    if section == 'campaign':
        images = soup.find('div',
            class_='full-description js-full-description responsive-media ' + \
                'formatted-lists').find_all('img')
#            +
#        soup.find('div',
#            class_='mb3 mb10-sm mb3 js-risks').find_all('img'))   
    for image in images:
        # Catch any iframes that fail to include an image source link
        try: 
            if 'gif' in image.get('data-src'):
                gif_count += 1
        except TypeError:
            pass
    return gif_count

def count_hyperlinks(soup, section):    
    """Count the number of hyperlinks in each project
    
    Args:
        soup (soup object): parsed HTML content for each project
        section (str): context section
    Returns:
        an integer of number of hyperlinks in each project"""
        
    # Use tree parsing to compute number of hyperlink 
    if section == 'campaign':
        return  len(soup.find('div',
            class_='full-description js-full-description responsive-media ' + \
                'formatted-lists').find_all('a')) 

def count_bolded(soup, section): 
    """Count the number of bold tags in each project
    
    Args:
        soup (soup object): parsed HTML content for each project
        section (str): context section
    Returns:
        an integer of number of bold tags in each project"""
        
    # Use tree parsing to compute number of bolded text 
    if section == 'campaign':
        return  len(soup.find('div',
            class_='full-description js-full-description responsive-media ' + \
                'formatted-lists').find_all('b')) 

        
def preprocess_text(text):
    """Text preprocessing including removing punctuation, lowercasing all words, 
       removing stop words and stemming remaining words
       
    Args:
        text (str): cleaned and normalized text for each project
    Returns:
        a string containing preprocessed text"""
    
    # Access stop word dictionary
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # Initialize the Porter stemmer
    porter = nltk.PorterStemmer()    
    # Remove punctuation and lowercase each word
    text = remove_punc(text).lower()    
    # Remove stop words and stem each word
    return ' '.join(porter.stem(term )
        for term in text.split()
        if term not in set(stop_words))

# -----------------------------------------------------------------------------

def extract_features(soup, campaign, section):
    """Construct meta features from the context of each project
    
    Args:
        soup (soup object): parsed HTML content for each project
        campaign (dict): dictionary containing context of each project
        section (str): section to be accessed
    Returns:
        a tuple containing """
    
    # Compute the number of words in the requested section
    num_words = len(get_words(campaign[section]))    
    # If the section contains no words, assign NaN to 'num_words' to avoid
    # potential division by zero
    if num_words == 0:
        num_words = np.nan
    if campaign[section] == 'section_not_found':
        return([np.nan] * 17)
    else:
        return (
            len(get_sentences(campaign[section])),
            num_words,
            len(identify_allcaps(campaign[section])),
            len(identify_allcaps(campaign[section])) / num_words,
            count_exclamations(campaign[section]),
            count_exclamations(campaign[section]) / num_words,
            count_buzz_words(campaign[section]),
            count_buzz_words(campaign[section]) / num_words,
            compute_avg_words(campaign[section]),
            count_paragraphs(soup, section),
            count_images(soup, section),
            count_videos(soup, section),
            count_gifs(soup, section),
            count_hyperlinks(soup, section),
            count_bolded(soup, section),
            count_bolded(soup, section) / num_words,
            campaign[section])

# Initialize empty DataFrames of meta features
features = ['num_sents', 'num_words', 'num_all_caps', 'percent_all_caps',
            'num_exclms', 'percent_exclms', 'num_buzz_words',
            'percent_buzz_words', 'avg_words_per_sent', 'num_paragraphs',
            'num_images', 'num_videos', 'num_gifs',
            'num_hyperlinks', 'num_bolded', 'percent_bolded',
            'normalized_text']

section_df = pd.DataFrame(columns=features)

for index, row in scraped_all.iterrows():
    # Parse scraped HTML
    soup = parse(row[0])
    # Extract and normalize campaign sections
    campaign = get_campaign(soup)
    campaign['campaign'] = normalize(campaign['campaign'])
    # Extract meta features
    section_df.loc[index] = extract_features(soup, campaign, 'campaign')           

df_2 =  pd.merge(df,section_df, left_index=True, right_index=True)         
df_2.to_pickle('raw_meta_feature.pkl')
                