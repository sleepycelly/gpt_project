import matplotlib.pyplot as plt
import pandas as pd


def main():
    evaluation = pd.read_csv("/raid/wald/gpt_results/evaluation.csv")

    bias_categories = evaluation["bias_category"].tolist()
    for bias in bias_categories:
        current = evaluation.loc[evaluation["bias_category"] == bias]
        demographics = current["demographic"].tolist()

        # contains the averages of sentiment (pos, neu, neg) and all toxicity scores
        evaluation_averages = current.groupby(["model", "demographic"]).mean().reset_index()

        # can also make more columns or rows
        fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(12,6))

        # plot sentiment averages for all demographics
        for demographic in enumerate(demographics):
            averages = evaluation_averages.loc[evaluation_averages["demographic"] == demographic[1]]
            # plot sentiment
            averages.plot.bar(stacked=True, y=["pos", "neu", "neg"], label = demographic[1], ax= axes[0][demographic[0]])

            # plot toxicity in 2nd row
            averages.plot.bar(stacked=True, y=["toxicity"], label = demographic[1], ax= axes[1][demographic[0]])


        # save plot
        plt.savefig()


        

if __name__ == "__main__":
    main()
