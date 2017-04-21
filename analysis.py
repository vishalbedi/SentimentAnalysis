import pandas
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

# Tweet file and dictionary
FILE_READ = {
    'en': './input/AFINN-1111',
    'emoticons': './input/AFINN-emoticons',
    'tweets': './input/Tweets.csv'
}

# If there is any negative word ahead of the sentiment word
# invert the sentiment
NEGATIVE_MULTIPLIER = -1

# Words that change the sentiment
NEGATIVE_WORDS = \
    ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
     "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
     "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
     "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
     "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
     "oughtnt", "shant", "shouldnt", "wasnt", "werent",
     "oughtn't", "shan't", "shouldn't", "wasn't", "weren't",
     "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

# Array of classifiers we run our data against.
Classifiers = [
    LogisticRegression(C=0.000000001, solver='liblinear', max_iter=200),
    KNeighborsClassifier(3),
    SVC(C=0.02, probability=False),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200),
    AdaBoostClassifier(),
    GaussianNB()]


def get_dict(filename: str = FILE_READ['en']) -> dict:
    """
    Created a dictionary from the lexicon with word as key ans sentiment as score
    :param filename: lexicon file
    :return: dict: a dictionary of words and their respective sentiment score
    """
    word_dict = {}
    with open(filename) as file:
        for line in file:
            word, score = line.strip().split('\t')
            word_dict[word] = int(score)
    return word_dict


# Create dict for global use
# avoids recreating dictionary
WORD_DICT = get_dict()


def read_file() -> pandas.DataFrame:
    """
    Reads the Tweet.csv fie and converts it to dataframe.
    Performs data cleaning over it.
    Adds a custom sentiment to the csv whose classification is achieved by considering negatives and 
    BUT statements within the tweet
    :return: pandas.DataFrame: CSV data-frame
    """
    tweet_collection = pandas.read_csv(FILE_READ['tweets'])
    tweet_collection['clean_tweets'] = tweet_collection['text'].apply(lambda tweet: clean_tweets(tweet))
    tweet_collection['custom_sentiment'] = tweet_collection['text'].apply(lambda tweet: custom_sentiment(tweet))
    tweet_collection['sentiment'] = tweet_collection['airline_sentiment'].apply(
        lambda sentiment: 0 if sentiment == 'negative' else 1)
    return tweet_collection


def clean_tweets(raw_tweet: str) -> str:
    """
    Reads a raw tweet, cleans it and converts it to the words that contribute to sentiment.
    :param raw_tweet: 
    :return: 
    """
    neg_words = get_negative_word(raw_tweet)
    letters_only = re.sub("[^a-zA-Z]", " ", raw_tweet)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stops]
    meaningful_neg_words = meaningful_words + neg_words
    return " ".join(meaningful_neg_words)


def get_negative_word(raw_tweet: str) -> list:
    """
    Get the negative words in the tweet
    :param raw_tweet: 
    :return: 
    """
    neg_words = []
    split_tweet = raw_tweet.split()
    for word in split_tweet:
        if word in NEGATIVE_WORDS or "n't" in word:
            neg_words.append(word)
    return neg_words


def get_regex(tokens_in_dict: list, boundary: bool = True) -> str:
    """"
    Convert dictionary into regex 
    :param tokens_in_dict: all the words contributing sentiments
    :param boundary: regex match with boundary or not
    :return: regex of dict
    """
    tokens = tokens_in_dict[:]
    tokens.sort(key = len, reverse=True)
    tokens = [re.escape(token) for token in tokens]

    regex = '(?:' + '|'.join(tokens) + ')'
    if boundary:
        regex = r'\b' + regex + r'\b'
    return regex


def get_sentiment_score(raw_tweet: str) -> list:
    """
    Generate sentiment score for given tweet
    :param raw_tweet: 
    :return: score list containing sentiment score of each word contributing sentiment
    """
    tweet = raw_tweet.lower().split(' ')
    word_list = WORD_DICT.keys()
    sentiment_score = [WORD_DICT[tweet[0]] if tweet[0] in word_list else 0]
    for i in range(1, len(tweet)):
        if 'but' in tweet:
            sentiment_score = handle_but(raw_tweet)
        elif tweet[i] in WORD_DICT.keys():
            if tweet[i - 1] in NEGATIVE_WORDS or "n't" in tweet[i - 1]:
                sentiment_score.append(WORD_DICT[tweet[i]] * NEGATIVE_MULTIPLIER)
            else:
                sentiment_score.append(WORD_DICT[tweet[i]])
    return sentiment_score


def handle_but(raw_tweet: str) -> list:
    """
    Handle scenarios in which the tweet contains 'but'
    :param raw_tweet: 
    :return: sentiment score for the tweet
    """
    tweets = raw_tweet.lower().split('but')
    score1 = get_sentiment_score(tweets[0].strip())
    score2 = get_sentiment_score(tweets[1].strip())
    return [float(sum(score1)) * 0.5, float(sum(score2)) * 1.5]


def custom_sentiment(raw_tweet: str) -> int:
    """
    Classify the tweets based on our own classification model
    :param raw_tweet: 
    :return: sentiment of the tweet
    """
    score = get_sentiment_score(raw_tweet)
    score = float(sum(score))
    if score < 0:
        return 0
    else:
        return 1


def analytics(tweet_df: pandas.DataFrame) -> dict:
    """
    Perform analytics on the data. 
    Create a model for a list of classifiers and calculate their accuracies
    :param tweet_df: dataframe of tweets
    :return accuracy_per_model: dictionary with accuracy per model
    """
    # split the data into 70 (training)-30 (testing)
    train, test = train_test_split(tweet_df, test_size=0.3, random_state=42)

    test_tweets = []
    train_tweets = []
    for tweet in train['clean_tweets']:
        train_tweets.append(tweet)
    for tweet in test['clean_tweets']:
        test_tweets.append(tweet)
    # Get the term frequency of words in each tweet
    cv = CountVectorizer(analyzer="word", min_df=1)
    train_features = cv.fit_transform(train_tweets)
    test_features = cv.transform(test_tweets)
    train_features_array = train_features.toarray()
    test_features_array = test_features.toarray()
    accuracy_per_model = {}
    for classifier in Classifiers:
        try:
            fit = classifier.fit(train_features, train['sentiment'])
            pred = fit.predict(test_features)
        except Exception:
            fit = classifier.fit(train_features_array, train['sentiment'])
            pred = fit.predict(test_features_array)
        accuracy = accuracy_score(pred, test['sentiment'])
        accuracy_per_model[classifier.__class__.__name__] = accuracy
        print('Accuracy of ' + classifier.__class__.__name__ + ' is ' + str(accuracy))
    return accuracy_per_model


if __name__ == '__main__':
    t = read_file()
    analytics(t)
