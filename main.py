import pandas as pd
from matplotlib import pyplot

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

if __name__ == '__main__':
    # Load dataset
    filename = 'winequality-white.csv'
    names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
             'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    dataset = pd.read_csv(filename, delimiter=";")

    # Summarize Data

    # Descriptive statistics
    # shape
    print('Number of Data Rows = ', dataset.shape[0]);
    print('Number of Data columns = ', dataset.shape[1]);

    # head
    print(dataset.head(20))

    # descriptions
    print(dataset.describe())

    # class distribution
    print(dataset.groupby('quality').size())

    # Data visualizations

    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(3, 4), sharex=False, sharey=False, figsize=(16, 12))
    pyplot.show()

    # histograms
    dataset.hist(color='#607c8e', figsize=(14, 11))
    pyplot.show()

    # scatter plot matrix
    pd.plotting.scatter_matrix(dataset, figsize=(17, 17))
    pyplot.show()
