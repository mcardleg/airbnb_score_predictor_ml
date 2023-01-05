import pandas as pd
import numpy as np
from nltk import download
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import sent_tokenize
from nltk.data import load
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from googletrans import Translator
from bs4 import BeautifulSoup


def get_features_from_listings_text(tokenizers, min_dfs, ngram_min, ngram_max, ngrams_range_max):
    df = pd.read_csv("clean_listings.csv")
    textual_fields = ['name', 'description', 'neighborhood_overview', 'host_about']

    while ngram_max <= ngrams_range_max:
        for key in tokenizers.keys():
            for min_df in min_dfs:
                print("     with tokenizer=" + key + " and min_df=" + str(min_df)
                      + " and ngram range=(" + str(ngram_min) + "," + str(ngram_max) + ")")
                vectorizer = CountVectorizer(
                    tokenizer=tokenizers[key],
                    stop_words=stopwords.words('english'),
                    min_df=min_df,
                    ngram_range=(ngram_min, ngram_max)
                )

                features = pd.DataFrame()
                for field in textual_fields:
                    data = vectorizer.fit_transform(df[field]).toarray()
                    columns = vectorizer.get_feature_names_out()
                    field_features = pd.DataFrame(data=data, columns=[field + "_" + column for column in columns])
                    features = pd.concat([features, field_features], axis=1)

                features.to_csv(
                    "feature_engineering/textual_data/"
                    + key + "_" + str(min_df) + "_"
                    + str(ngram_min) + "-" + str(ngram_max)
                    + ".csv", index=False)
        ngram_max += 1


def vectorization_parameter_cross_validation(c_range, tokenizers, min_dfs, ngram_min, ngram_max, ngrams_range_max):
    y_df = pd.read_csv("../scores.csv")
    y = y_df['review_scores_rating']

    while ngram_max <= ngrams_range_max:
        for key in tokenizers.keys():
            for min_df in min_dfs:
                print("     with tokenizer=" + key + " and min_df=" + str(min_df)
                      + " and ngram range=(" + str(ngram_min) + "," + str(ngram_max) + ")")
                X = pd.read_csv(
                    "feature_engineering/textual_data/"
                    + key + "_" + str(min_df) + "_"
                    + str(ngram_min) + "-" + str(ngram_max)
                    + ".csv")
                mean_error = []
                std_error = []

                for C in c_range:
                    model = Lasso(alpha=1/(2*C), max_iter=100000)
                    model.fit(X, y)
                    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
                    mean_error.append(np.array(scores).mean())
                    std_error.append(np.array(scores).std())

                plt.errorbar(c_range, mean_error, yerr=std_error, linewidth=3,
                             label="tokenizer={} min_df={} ngram range = ({},{})"
                             .format(key, min_df, ngram_min, ngram_max))
        ngram_max += 1

    plt.rc('font', size=12)
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.title("Cross validation for tokenizer, min_kf and ngram range")
    plt.xlabel("C")
    plt.ylabel("Negative mean squared error")
    plt.legend()
    plt.show()


def translate_reviews(index):
    reviews = pd.read_csv("../original_data/reviews.csv")
    translator = Translator()

    # Split up translations because of timeouts
    english_reviews = []
    file = 542
    while index < len(reviews.index):
        english_reviews.append(translator.translate(str(reviews['comments'][index])).text)
        if index % 200 == 0:         # write at regular intervals because of timeouts
            pd.DataFrame(english_reviews)\
                .to_csv("feature_engineering/translations/" + str(file) + ".csv", index=False)
            english_reviews = []
            file += 1
        index += 1


def merge_translations(max_file_index):
    english_reviews = pd.DataFrame()
    file_index = 2

    while file_index <= max_file_index:
        english_reviews = pd.concat(
            [english_reviews, pd.read_csv("feature_engineering/translations/" + str(file_index) + ".csv")])
        file_index += 1

    english_reviews.columns = ['reviews']
    english_reviews.to_csv("feature_engineering/english_reviews.csv", index=False)


def clean_review(field):
    soup = BeautifulSoup(field, "lxml")
    field = soup.get_text(separator=" ")
    field = field.replace("nan", "")
    field = field.replace("â€™", "'")
    field = field.replace("\r", " ")
    field = field.replace("\t", " ")
    return field


def replace_hostname(df):
    name = df['host_name']
    print(name)
    review = df['reviews']
    return review


def map_reviews_to_listings():
    english_reviews = pd.read_csv("english_reviews.csv")
    original_reviews = pd.read_csv("../original_data/reviews.csv")
    reviews = pd.concat([original_reviews, english_reviews], axis=1)
    reviews['reviews'] = reviews['reviews'].apply(lambda x: clean_review(str(x)))

    listing_reviews = dict()
    for index, entry in reviews.iterrows():
        key = reviews['listing_id'][index]
        value = str(reviews['reviews'][index])
        if listing_reviews.get(key):
            value = str(listing_reviews.get(key)) + " " + value
        listing_reviews.update({reviews['listing_id'][index]: value})

    merged_reviews = pd.DataFrame()
    merged_reviews['listing_id'] = listing_reviews.keys()
    merged_reviews['reviews'] = merged_reviews['listing_id'].apply(lambda x: listing_reviews.get(x))

    listings = pd.read_csv("clean_listings.csv")
    merged_reviews = pd.merge(
        listings[['id', 'host_name']], merged_reviews, how='inner', left_on='id', right_on='listing_id')

    for index, entry in merged_reviews.iterrows():
        merged_reviews['reviews'][index] = str(merged_reviews['reviews'][index])\
            .replace(str(merged_reviews['host_name'][index]), "hostname")

    merged_reviews.to_csv("feature_engineering/listing_review_map.csv", index=False)


def get_features_from_reviews_text(tokenizer, min_df, ngram_min, ngram_max):
    df = pd.read_csv("listing_review_map.csv")
    df = df.fillna('0')
    vectorizer = CountVectorizer(
        tokenizer=tokenizer,
        stop_words=stopwords.words('english'),
        min_df=min_df,
        ngram_range=(ngram_min, ngram_max)
    )
    data = vectorizer.fit_transform(df['reviews']).toarray()
    columns = vectorizer.get_feature_names_out()
    features = pd.DataFrame(data=data, columns=["review" + "_" + column for column in columns])
    features.to_csv("feature_engineering/ohe_reviews.csv", index=False)


def text_processing():
    packages = ['punkt', 'wordnet', 'omw-1.4', 'stopwords']
    for package in packages:
        download(package)

    c_range = [0.5, 100, 200, 300]
    tokenizers = {
        "CountVectorizer": CountVectorizer().build_tokenizer(),
        # "Whitespace": WhitespaceTokenizer().tokenize,
        # "Word": word_tokenize,
        # "Wordpunct": wordpunct_tokenize,
        # "Sent": sent_tokenize,
        # "Punkt": load('tokenizers/punkt/english.pickle').tokenize,
        # "Regexp": RegexpTokenizer('\w+|\$[\d\.]+|\S+').tokenize
    }
    min_dfs = [70, 80, 90, 100]
    ngram_min = 1
    ngram_max = 5
    ngrams_range_max = 5

    print("Getting features from textual field in listings.csv")
    get_features_from_listings_text(tokenizers, min_dfs, ngram_min, ngram_max, ngrams_range_max)

    print("Cross validating for vectorization parameters")
    vectorization_parameter_cross_validation(c_range, tokenizers, min_dfs, ngram_min, ngram_max, ngrams_range_max)

    print("Translating all reviews to English")
    # Takes 8 hours. May time out and need to be run again from where the previous run stopped
    # translate_reviews(1)

    print("Mapping reviews to listings")
    map_reviews_to_listings()

    print("Getting features from reviews.")
    get_features_from_reviews_text(CountVectorizer().build_tokenizer(), 90, 1, 5)

