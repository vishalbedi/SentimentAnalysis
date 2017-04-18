import pandas
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

FILE_READ = {
    'en': './input/AFINN-1111',
    'emoticons': './input/AFINN-emoticons',
    'tweets': './input/Tweets.csv'
}

NEGATIVE_MULTIPLIER = -1

NEGATIVE_WORDS = \
    ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
     "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
     "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
     "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
     "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
     "oughtnt", "shant", "shouldnt", "wasnt", "werent",
     "oughtn't", "shan't", "shouldn't", "wasn't", "weren't",
     "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]


Classifiers = [
    LogisticRegression(C=0.000000001, solver='liblinear', max_iter=200),
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200),
    AdaBoostClassifier(),
    GaussianNB()]

def get_dict(filename=FILE_READ['en']):
    word_dict = {}
    with open(filename) as file:
        for line in file:
            word, score = line.strip().split('\t')
            word_dict[word] = int(score)
    return word_dict


WORD_DICT = [] #get_dict()


def read_file():
    tweet_collection = pandas.read_csv(FILE_READ['tweets'])
    tweet_collection['clean_tweets'] = tweet_collection['text'].apply(lambda tweet: clean_tweets(tweet))
   # tweet_collection['custom_sentiment'] = tweet_collection['text'].apply(lambda tweet: custom_sentiment(tweet))
    tweet_collection['sentiment'] = tweet_collection['airline_sentiment'].apply(lambda sentiment: 0 if sentiment == 'negative' else 1)
    return tweet_collection


def clean_tweets(raw_tweet):
    letters_only = re.sub("[^a-zA-Z]", " ", raw_tweet)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return (" ".join(meaningful_words))


def get_regex(tokens_in_dict, boundry=True):
    """"Convert dictionary into regex """
    tokens = tokens_in_dict[:]
    tokens.sort(key=len, reverse=True)
    tokens = [re.escape(token) for token in tokens]

    regex = '(?:' + '|'.join(tokens) + ')'
    if boundry:
        regex = r'\b' + regex + r'\b'
    return regex


def get_sentiment_score(raw_tweet):
    tweet = raw_tweet.lower().split(' ')
    word_list = WORD_DICT.keys()
    sentiment_score = [WORD_DICT[tweet[0]] if tweet[0] in word_list else 0]
    for i in range(1, len(tweet)):
        if 'but' in tweet:
            sentiment_score = handle_but(raw_tweet)
        elif tweet[i] in WORD_DICT.keys():
            if tweet[i-1] in NEGATIVE_WORDS or "n't" in tweet[i-1]:
                sentiment_score.append(WORD_DICT[tweet[i]] * NEGATIVE_MULTIPLIER)
            else:
                sentiment_score.append(WORD_DICT[tweet[i]])
    return sentiment_score


def handle_but (raw_tweet):
    tweets = raw_tweet.lower().split('but')
    score1 = get_sentiment_score(tweets[0])
    score2 = get_sentiment_score(tweets[1])
    return [float(sum(score1))*0.5, float(sum(score2))*1.5]


def custom_sentiment(raw_tweet):
    score = get_sentiment_score(raw_tweet)
    score = float(sum(score))
    if score < 0:
        return 0
    else:
        return 1

def analytics(tweet_df):
    train, test = train_test_split(tweet_df, test_size=0.2, random_state=42)

    test_tweets = []
    train_tweets = []
    for tweet in train['clean_tweets']:
        train_tweets.append(tweet)
    for tweet in test['clean_tweets']:
        test_tweets.append(tweet)

    cv = CountVectorizer(analyzer="word")
    train_features = cv.fit_transform(train_tweets)
    test_features = cv.transform(test_tweets)
    blah = cv.get_feature_names()
    print(len(blah))
    train_features_array = train_features.toarray()
    test_features_array = test_features.toarray()
    Accuracy = []
    Model = []
    for classifier in Classifiers:
        try:
            fit = classifier.fit(train_features, train['sentiment'])
            pred = fit.predict(test_features)
        except Exception:
            fit = classifier.fit(train_features_array, train['sentiment'])
            pred = fit.predict(test_features_array)
        accuracy = accuracy_score(pred, test['sentiment'])
        Accuracy.append(accuracy)
        Model.append(classifier.__class__.__name__)
        print('Accuracy of ' + classifier.__class__.__name__ + 'is ' + str(accuracy))

if __name__ == '__main__':
    t = read_file()
    analytics(t)
