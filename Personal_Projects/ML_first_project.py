from pathlib import Path
import pandas as pd
import tarfile
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid = True)
plt.xlabel("Income Category")
plt.ylabel("Number of Districts")
plt.show()