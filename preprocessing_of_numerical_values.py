import pandas as pd
import datetime
import calendar
import re


def get_epoch_from_date(x):
    if x != 0:
        parts_of_dates = str(x).split('-')
        t = datetime.datetime(int(parts_of_dates[0]), int(parts_of_dates[1]), int(parts_of_dates[2]))
        return calendar.timegm(t.timetuple())
    else:
        return 0


def get_int_from_response_time(x):
    y = 0
    if x == "within an hour":
        y = 60
    elif x == "within a few hours":
        y = 180
    elif x == "within a day":
        y = 1400
    elif x == "a few days or more":
        y = 4320
    return y


def get_int_from_boolean(x):
    if x == 't':
        return 1
    return 0


def get_number_from_bathroom_text(x):
    if x != 0:
        number = re.findall(r'\d*.\d+|\d+', x)
        if len(number) == 1:
            return number[0]
        return 1
    return 0


def get_word_presence_boolean_from_text(x, word):
    if str(x) in str(x).lower():
        return 1
    return 0


def get_word_presence_boolean_from_list(x, word):
    if word in x:
        return 1
    return 0


def get_int_from_percentage(x):
    if x == '0':
        return x
    return x[:-1]


def map_nearly_numbers():
    listings = pd.read_csv("clean_listings.csv")
    already_numeric = ['id', 'host_id', 'host_listings_count', 'host_total_listings_count', 'latitude', 'longitude',
                       'accommodates', 'bedrooms', 'beds', 'minimum_nights', 'maximum_nights', 'minimum_minimum_nights',
                       'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights',
                       'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'availability_30', 'availability_60',
                       'availability_90', 'availability_365', 'number_of_reviews', 'number_of_reviews_ltm',
                       'number_of_reviews_l30d', 'calculated_host_listings_count',
                       'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms',
                       'reviews_per_month']
    numerical = listings[already_numeric]

    numerical["host_location"] = listings["host_location"].apply(
        lambda x: get_word_presence_boolean_from_text(x, "Ireland")
    )
    numerical["host_response_time"] = listings["host_response_time"].apply(
        lambda x: get_int_from_response_time(x)
    )
    numerical["price"] = listings["price"].apply(
        lambda x: str(x)[1:].replace(',', '')
    )
    numerical["bathrooms"] = listings["bathrooms_text"].apply(
        lambda x: get_number_from_bathroom_text(x)
    )
    numerical["bathrooms_text"] = listings["bathrooms_text"].apply(
        lambda x: get_word_presence_boolean_from_text(x, "shared")
    )

    date_values = ["host_since", "first_review", "last_review"]
    for column in date_values:
        numerical[column] = listings[column].apply(lambda x: get_epoch_from_date(x))

    percentage_values = ["host_response_rate", "host_acceptance_rate"]
    for column in percentage_values:
        numerical[column] = listings[column].apply(lambda x: get_int_from_percentage(x))

    boolean_values = ["host_is_superhost", "host_has_profile_pic", "host_identity_verified",
                      "has_availability", "instant_bookable"]
    for column in boolean_values:
        numerical[column] = listings[column].apply(lambda x: get_int_from_boolean(x))

    verifications = ["email", "work_email", "phone"]
    for method in verifications:
        numerical[method + "_verification"] = listings["host_verifications"].apply(
            lambda x: get_word_presence_boolean_from_list(x, method))

    numerical.to_csv("feature_engineering/numerical_data.csv", index=False)
