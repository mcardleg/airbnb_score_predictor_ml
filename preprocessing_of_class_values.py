import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def simplify_property_type(x):
    if "Entire " in str(x):
        x = x.replace("Entire ", "")
    if "Private " in str(x):
        x = x.replace("Private ", "")
    if "Shared " in str(x):
        x = x.replace("Shared ", "")
    if "Room in " in str(x):
        x = x.replace("Room in ", "")
    if "room in " in str(x):
        x = x.replace("room in ", "")
    return x


def one_hot_encode():
    df = pd.read_csv("clean_listings.csv")
    df['property_type'] = df['property_type'].apply(lambda x: simplify_property_type(x))
    classification_fields = ["neighbourhood_cleansed", "property_type", "room_type"]
    ohe_df = pd.DataFrame()
    for field in classification_fields:
        encoder = OneHotEncoder()
        df[field] = df[field].apply(lambda x: x.lower())
        ohe = pd.DataFrame(data=encoder.fit_transform(df[[field]]).toarray(), columns=encoder.categories_)
        ohe_df = pd.concat([ohe_df, ohe], axis=1)

    ohe_df.to_csv("feature_engineering/ohe_listings.csv", index=False)
