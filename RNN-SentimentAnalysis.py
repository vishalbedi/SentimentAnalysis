import pandas as pd;
import numpy as np
import tflearn
from tflearn.data_utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import re

print("Reading dataset")
# Read The input dataset
filename = "./input/Tweets.csv"
data = pd.read_csv(filename, encoding="ISO-8859-1", dayfirst=False,usecols=['airline_sentiment', 'text'])

print("LSTM-GLOVE MODEL FOR SENTIMENT ANALYSIS")
test = input("Enter Y to train and N to generate accuracy - Training consumes heavy resource due to the huge size of the DNN")

if test == 'Y'or test == 'N':
    # Creating set of the input stopwords to improve computation - Reference Scikit learn documentation
    stops = set(stopwords.words("english"))
    airlines = set(['united', 'virginamerica', 'staralliance', 'southwestair', 'jetblue', 'usairways', 'americanair', ''])
    word_vectors = []

    # Removing data with neutral sentiment
    filtered = pd.DataFrame(data[data.airline_sentiment != 'neutral'])

    # Preprocessing input data
    # for i in range(len(filtered['text'])):
    for item in filtered['text'].iteritems():
        # Retaining only characters
        characters = re.sub("[^a-zA-Z]", " ", item[1])
        # characters = re.sub("[^a-zA-Z]", " ", filtered['text'][i])
        words = characters.lower().split(' ')
        words = [w for w in words if not w in stops and not w in airlines]
        word_vectors.append(" ".join(words))


    vectorize = CountVectorizer(preprocessor=None, max_features=5000)
    vec = vectorize.fit_transform(word_vectors)
    word_vectors = vec.toarray()

    # Assigning int value to be
    Y = (data['airline_sentiment'] == 'positive').astype(np.int_)

    records = len(filtered['text'])
    shuffle = np.arange(records)
    np.random.shuffle(shuffle)
    test_fraction = 0.8

    # Splits the data into training and testing samples
    train_split, test_split = shuffle[:int(records*test_fraction)], shuffle[int(records*test_fraction):]

    trainX, trainY = word_vectors[train_split,:], to_categorical(Y.values[train_split],2)
    testX, testY = word_vectors[test_split,:], to_categorical(Y.values[test_split],2)
    # # Network construction

    # The vector input is of represented using a vector of size 5000
    net = tflearn.input_data([None, 5000])
    # The Embedding layer represents the words in VSM on semantic similarity contributing to sentiment
    net = tflearn.embedding(net, input_dim=500, output_dim=128)
    # The lstm learn the relationship between the vectors
    net = tflearn.lstm(net, 128)
    # The Pooling layer
    net = tflearn.fully_connected(net, 2, activation='softmax')
    # performs gradient descent using adam optimizer
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy')

    # # Train the network
    model = tflearn.DNN(net, tensorboard_verbose=0)
    if(test == 'Y'):
        model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128)
    elif(test == 'N'):
        # Loads the pretrained model to the database
        print("Loading the trained model")
        model.load('model/sentiment.tfl')
    score = model.evaluate(testX, testY)
    print("Evaluation score: " + score);
else:
    print("Exiting.. wrong choice..")




