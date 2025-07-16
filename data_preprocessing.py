import xgboost as xgb
import pickle
import pandas as pd
from sklearn import set_config
set_config(transform_output="pandas")
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

model = []

def init_text_model():
    """
    Initializes the text model, if it has not been initialized before.

    Loads all-mpnet-base-v2 from sentence transformers.
    """
    global model
    if not model:
        model = SentenceTransformer('all-mpnet-base-v2')

def features_and_labels(data):
    """
    Removes the Survived column (which has the binary classes) from
    the training vectors, and returns them as y.
    """
    X = data.drop('Survived', axis = 1)
    y = data.Survived
    return X,y

def prepare_data(data):
    """
    Performs a very modest data cleaning upon the Name field, fills
    missing values from the Embarked field, one-hot encodes the
    Embarked, Sex and Pclass fields and returns the result.
    """
    data.loc[:, 'Name'] = data.loc[:, 'Name'].replace('"', '')
    data.loc[:, 'Name'] = data.loc[:, 'Name'].replace("'", '')
    data.loc[:,'Embarked'] = data.loc[:,'Embarked'].fillna('S')

    embark_dummies_titanic  = pd.get_dummies(data['Embarked'].astype(pd.CategoricalDtype(categories=['S', 'C', 'Q'])), dtype=float)

    sex_dummies_titanic  = pd.get_dummies(data['Sex'].astype(pd.CategoricalDtype(categories=['male', 'female'])), dtype=float)

    pclass_dummies_titanic  = pd.get_dummies(data['Pclass'].astype(pd.CategoricalDtype(categories=[1, 2, 3])), prefix="Class", dtype=float)
    

    data = data.drop(['Embarked', 'Sex', 'Pclass'], axis=1)
    final_data = data.join([embark_dummies_titanic, sex_dummies_titanic, pclass_dummies_titanic])
    return final_data

def fill_missing_age(data, median=0):
    """
    Fill the missing values for the Age field, uses the median
    from the set (or any median provided).
    """
    current_median = median
    if not median:
        current_median = data.Age.median()
    data.loc[:,'Age'] = data.loc[:,'Age'].replace('', current_median)
    data.loc[:,'Age'] = data.loc[:,'Age'].fillna(current_median)
    return data, current_median

def compute_text_features(data, selected_features = ['Name', 'Ticket', 'Cabin']):
    """
    Creates a combined text field from the Name, Ticket and Cabin fields,
    and computes feature vectors for this new field.
    """
    init_text_model()
    text_features_to_compute = data[selected_features]
    text_features_to_compute = pd.DataFrame(text_features_to_compute['Name'].astype(str) + ' Ticket ' + text_features_to_compute['Ticket'].astype(str) + ' Cabin ' + text_features_to_compute['Cabin'].fillna('').astype(str), columns=['combined'])
    embeddings = model.encode(text_features_to_compute.combined.values)
    return embeddings

def join_and_scale(data, text_data, scaler, selected_features = ['Name', 'Ticket', 'Cabin'], fit_transform = True, skip=False):
    """
    Joins the text data features to the rest of the features.

    Removes the original text fields.

    Performs feature standardization.
    """
    data = data.drop(selected_features, axis=1)
    data.set_index('PassengerId', inplace=True)
    data.reset_index(drop=True, inplace=True)
    if not skip:
        data = data.join(pd.DataFrame(text_data))
    data = data.astype(float)
    data.columns = data.columns.astype(str)
    if fit_transform:
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)
    return data, scaler


def preprocess_samples(sample, has_y = False, sample_scaler=None, sample_median=None):
    """
    Combines all the previous functions to produce the feature vectors for a given
    set of samples.

    Note that if the scaler or median are provided, keeps using the provided objects.
    """
    sample = prepare_data(sample)
    y = None
    if has_y:
        sample, y = features_and_labels(sample)
    if sample_median is None:
        sample, sample_median = fill_missing_age(sample)
    else:
        sample, sample_median = fill_missing_age(sample, sample_median)

    text_features_to_combine = ['Name', 'Ticket', 'Cabin']

    sample_text = compute_text_features(sample, selected_features=text_features_to_combine)

    if sample_scaler is None:
        sample_scaler = StandardScaler()
        sample, sample_scaler = join_and_scale(sample, sample_text, sample_scaler, fit_transform=True)
    else:
        sample, sample_scaler = join_and_scale(sample, sample_text, sample_scaler, fit_transform=False)

    return sample, y, sample_scaler, sample_median

