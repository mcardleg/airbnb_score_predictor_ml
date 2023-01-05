import pandas as pd
from feature_engineering.preprocessing_of_numerical_values import map_nearly_numbers
from feature_engineering.preprocessing_of_class_values import one_hot_encode
from feature_engineering.preprocessing_of_textual_values import text_processing
from feature_engineering.feature_selection import feature_selection
from models import run_models

df = pd.read_csv("original_data/listings.csv")
y_fields = ["review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin",
            "review_scores_communication", "review_scores_location"]
df = df.dropna(subset=y_fields)
df = df.fillna(0)
df.to_csv("feature_engineering/clean_listings.csv", index=False)
df[y_fields].to_csv("feature_engineering/scores.csv", index=False)

print("MAPPING TEXTUAL DATA TO NUMBERS WHERE POSSIBLE")
map_nearly_numbers()

print("ONE-HOT ENCODING CLASSES")
one_hot_encode()

print("GETTING TEXTUAL FEATURES")
text_processing()

print("RUNNING FEATURE SELECTION")
feature_selection(y_fields)

print("RUNNING MODELS")
run_models(y_fields)
