import pandas as pd


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    df = pd.read_csv("./dataset/Webpages_Classification_test_data.csv", sep=',', nrows=100)
    print(df.head(5))
