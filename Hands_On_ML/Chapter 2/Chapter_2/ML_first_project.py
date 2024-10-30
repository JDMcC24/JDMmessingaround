from pathlib import Path
import pandas as pd
import tarfile
import sys
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()
# housing.hist(bins = 50, figsize = (12,8))
# plt.show()
#train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)
housing["income_cat"] = pd.cut(housing["median_income"], 
                               bins = [0.,1.5, 3.0,4.5,6,np.inf],
                               labels = [1,2,3,4,5])
# housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid = True)
# plt.xlabel("Income Category")
# plt.ylabel("Number of Districts")
# plt.show()



# splitter = StratifiedShuffleSplit(n_splits = 10, test_size=0.2, random_state =42)
# strat_splits = []
# for train_index, test_index in splitter.split(housing, housing['income_cat']):
#     strat_train_set_n = housing.iloc(train_index)
#     strat_test_set_n = housing.iloc[test_index]
#     strat_splits.append([strat_train_set_n, strat_test_set_n])

# strat_train_set, strat_test_set = strat_splits[0]

strat_train_set, strat_test_set = train_test_split(housing, test_size = 0.2, stratify = housing["income_cat"], random_state = 42)

# print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))

# print(housing["income_cat"].value_counts()/len(housing))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis = 1, inplace=True)

housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap = "jet",colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
#plt.show()

corr_matrix = housing.corr(numeric_only=True)
#print(corr_matrix["median_house_value"].sort_values(ascending=False))

from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))
#plt.show()

housing.plot(kind = "scatter", x= "median_income", y = "median_house_value", alpha = 0.1, grid = True)
#plt.show()

housing["rooms_per_house"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["people_per_house"] = housing["population"]/housing["households"]

#corr_matrix = housing.corr(numeric_only = True)
#print(corr_matrix["median_house_value"].sort_values(ascending=False))

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#housing.dropna(subset=["total_bedrooms"], inplace=True)  #Removes data points missing bedrooms

#housing.drop("total_bedrooms", axis=1) #Removes the total bedrooms attribute

median = housing["total_bedrooms"].median() 
housing["total_bedrooms"].fillna(median, inplace=True)  #Replaces missing values with the median value

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
housing_num = housing.select_dtypes(include= [np.number])

imputer.fit(housing_num)

# print(imputer.statistics_)
# print(housing_num.median().values)

X = imputer.transform(housing_num) #Replaces all missing values with the median of that attribute, stored as an numpy array
housing_tr = pd.DataFrame(X, columns = housing_num.columns, index = housing_num.index) #Replacing X as a data frame with colums

housing_cat = housing[["ocean_proximity"]]
#print(housing_cat.head(8))


from sklearn.preprocessing import OrdinalEncoder  #Turing categorical attribute into a numeric attribute
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
#print(housing_cat_encoded)


""" One Hot encoding categorical attributes"""
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
#print(cat_encoder.categories_)


"""Normalizing attributes"""
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)


from sklearn.linear_model import LinearRegression

target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())


""" Simple Linear Regression on scaled data"""
model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data = housing[["median_income"]].iloc[:5]  # pretend this is new data

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)

#print(predictions)

"""Doing it all at once"""
from sklearn.compose import TransformedTargetRegressor
model = TransformedTargetRegressor(LinearRegression(), transformer= StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)




from sklearn.preprocessing import FunctionTransformer
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])

from sklearn.metrics.pairwise import rbf_kernel

age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)


from sklearn.pipeline import Pipeline
num_pipeline = Pipeline([ ("impute", SimpleImputer(strategy="median")),
                         ("standardize", StandardScaler())
                         ])

from sklearn.pipeline import make_pipeline
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

housing_num_prepared = num_pipeline.fit_transform(housing_num)
#print(housing_num_prepared[:2].round(2))
df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
    index = housing_num.index
)
#print(df_housing_num_prepared.head())

from sklearn.compose import ColumnTransformer
num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households",
               "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy= "most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include= object))
)

housing_prepared = preprocessing.fit_transform(housing)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):  # no *args or **kwargs!
        self.with_mean = with_mean

    def fit(self, X, y=None):  # y is required even though we don't use it
        X = check_array(X)  # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]  # every estimator stores this in fit()
        return self  # always return self!

    def transform(self, X):
        check_is_fitted(self)  # looks for learned attributes (with trailing _)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_

from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


def column_ratio(X):
    return X[:, [0]]/X[:,[1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy= "median"),
        FunctionTransformer(column_ratio, feature_names_out= ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out= "one-to-one"),
    StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters= 10, gamma= 1., random_state= 42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]), 
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
    ("geo", cluster_simil, ["latitude", "longitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include= object))],
    remainder= default_num_pipeline 
    )

housing_prepared = preprocessing.fit_transform(housing)
#print(housing_prepared.shape)

from sklearn.linear_model import LinearRegression
lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

housing_predictions = lin_reg.predict(housing)
# print(housing_predictions[:5].round(-2))
# print(housing_labels.iloc[:5].values)

from sklearn.metrics import mean_squared_error
#lin_rmse = mean_squared_error(housing_labels, housing_predictions, squared =False)
#print(lin_rmse)

# from sklearn.tree import DecisionTreeRegressor

# tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
# tree_reg.fit(housing, housing_labels)
# housing_predictions = tree_reg.predict(housing)
# tree_rmse = mean_squared_error(housing_labels, housing_predictions, squared= False)
# #print(tree_rmse)

from sklearn.model_selection import cross_val_score
# tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, scoring = "neg_root_mean_squared_error", cv = 10)
# #print(pd.Series(tree_rmses).describe())

from sklearn.ensemble import RandomForestRegressor

forest_reg = make_pipeline(preprocessing,
                           RandomForestRegressor(random_state=42))
# forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
#                                 scoring="neg_root_mean_squared_error", cv=10)
# print(pd.Series(forest_rmses).describe())
# forest_reg.fit(housing, housing_labels)
# housing_predictions = forest_reg.predict(housing)
# forest_rmse = mean_squared_error(housing_labels, housing_predictions, squared= False)
# print(forest_rmse)

from sklearn.model_selection import GridSearchCV

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])
param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],
     'random_forest__max_features': [4, 6, 8]},
    {'preprocessing__geo__n_clusters': [10, 15],
     'random_forest__max_features': [6, 8, 10]},]
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                           scoring='neg_root_mean_squared_error')
grid_search.fit(housing, housing_labels)
print(grid_search.best_params_)
cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values( by = "mean_test_score", ascending= False, inplace= True)
print(cv_res.head())