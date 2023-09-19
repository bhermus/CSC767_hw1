import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn import preprocessing

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def part1(dataset: pd.DataFrame):
    # ----- PART 1 -----
    # Descriptive statistics
    # Listing 2a - shape
    print('Number of Data Rows = ', dataset.shape[0]);
    print('Number of Data columns = ', dataset.shape[1]);

    # Listing 2b - head
    print(dataset.head(20))

    # Listing 2c - descriptions
    print(dataset.describe())

    # Listing 2d - class distribution
    print(dataset.groupby('class').size())

    # Listing 3 - Data visualizations

    # Listing 3a - box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(3, 4), sharex=False, sharey=False, figsize=(16, 12))
    pyplot.show()

    # Listing 3b - histograms
    dataset.hist(color='#607c8e', figsize=(14, 11))
    pyplot.show()

    # Listing 3c - scatter plot matrix
    pd.plotting.scatter_matrix(dataset, figsize=(17, 17))
    pyplot.show()


def part2(dataset: pd.DataFrame):
    # ----- PART 2 -----
    # Listing 4a - Pairwise Pearson correlations
    print(dataset.corr(method='pearson'))

    # Listing 4b - Skew for Each Attribute
    print(dataset.skew())

    # Listing 4c - Univariate Density Plot
    dataset.plot(kind='density', subplots=True, layout=(4, 4), sharex=False, figsize=(14, 14))
    pyplot.show()

    # Listing 4d - Correlation Matrix Plot
    fig = pyplot.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, 7, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names[:-1])
    ax.set_yticklabels(names[:-1])
    pyplot.show()

    # Listing 5a - Rescaling Data
    array = dataset.values
    rescaled = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(array)
    np.set_printoptions(precision=3)
    print(rescaled[0:5])  # view first 5 rows of rescaled data

    # Listing 5b - Standardize Data
    standardized = preprocessing.StandardScaler().fit_transform(array)
    np.set_printoptions(precision=3)
    print(standardized[0:5])  # view first 5 rows of standardized data

    # Listing 5c - Normalize Data
    normalized = preprocessing.Normalizer().fit_transform(array)
    np.set_printoptions(precision=3)
    print(normalized[0:5])  # view first 5 rows of normalized data

    # Listing 5d - Binarization
    pyplot.show()

if __name__ == '__main__':
    # Listing 1 - Load dataset
    filename = 'seed.csv'
    names = ["area", "perimeter", "compactness", "length", "width", "asymmetry", "groove", "class"]
    dataset = pd.read_csv(filename, delimiter=",", names=names, index_col=7)

    part1(dataset)
    part2(dataset)
