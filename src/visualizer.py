import matplotlib.pyplot as plt

class Visualizer:

    @staticmethod
    def plot_sentiment_distribution(df):

        counts = df["sentiment"].value_counts()

        counts.plot(kind="bar")

        plt.title("Sentiment Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")

        plt.show()