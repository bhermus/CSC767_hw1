import pandas as pd
from matplotlib import pyplot

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
    print(dataset.groupby('quality').size())

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


if __name__ == '__main__':
    # Listing 1 - Load dataset
    filename = 'winequality-white.csv'
    names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
             'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    dataset = pd.read_csv(filename, delimiter=";")

    # part1(dataset)
    part2(dataset)
