import pandas as pd

if __name__ == '__main__':
    dataset_volume_range = 1000
    df = pd.read_csv("./dataset/Webpages_Classification_test_data.csv", sep=',',
                     usecols=['url', 'https', 'js_len', 'label'])
    df = df.sort_values(
        by="label",
        ascending=False
    )


    bad = df[df['label'] == 'bad']
    good = df[df['label'] == 'good']

    print(bad[:dataset_volume_range].to_string())
    print(good[:dataset_volume_range].to_string())

# random sample of rows
# equal numbers
