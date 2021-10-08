# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def main():
    columns_gibberish = ["Response", "Label"]
    path_to_import_gibberish = "./dataset/Gibberish.csv"
    df_gibberish = pd.read_csv(path_to_import_gibberish, usecols=columns_gibberish, sep=',', encoding="ISO-8859-1")
    df_gibberish.tail()
    print(len(df_gibberish))

    path_to_import_amazon = "./dataset/Amazon.csv"
    df_amazon = pd.read_csv(path_to_import_amazon, encoding="ISO-8859-1")
    df_amazon = df_amazon.sample(n = 3767)
    print(len(df_amazon))
    test_data = [
        '23sadfkla2145nla',
        'hello my name is rik, this is just normal text nothing wrong with it. Lets see what the naive bayes will show'
    ]


    df_amazon.drop(df_amazon.columns[0], inplace=True, axis=1)
    df_amazon.columns = ["Response"]
    df_amazon["Label"] = 0


    # Remove title from text
    def remove_intro(x):
        if x.find(":") < 0:
            return x
        else:
            return x[x.find(":") + 1:len(x)].strip()

    df_amazon["Response"] = df_amazon["Response"].apply(remove_intro)
    vectorizer = CountVectorizer(stop_words='english')
    all_features = vectorizer.fit_transform(df_amazon.Response)
    all_features.shape
    vectorizer.vocabulary_

    df_merged = pd.concat([df_amazon, df_gibberish], ignore_index=True, sort=False)
    df_merged = df_merged.sample(n=150)
    print(df_merged.to_string())




    # x_train, x_test, y_train, y_test, = train_test_split(all_features, df_amazon.Label, test_size=0.3, random_state=88)
    #
    # # print(x_train.shape)
    # # print(x_test.shape)
    #
    # classifier = MultinomialNB()  # Create Model
    # classifier.fit(x_train, y_train)  # Train Model
    #
    # nr_correct = (y_test == classifier.predict(x_test)).sum()
    # print(f'{nr_correct} correctly predicted')
    # nr_incorrect = y_test.size - nr_correct
    # print(f'{nr_incorrect} incorrectly predicted')
    #
    # fraction_wrong = nr_incorrect / (nr_correct + nr_incorrect)
    # print(f' The testing accuracy of the model is {1-fraction_wrong:.2}%')
    #
    # print(classifier.score(x_test, y_test))
    # print(df_amazon.to_string())
    # Combine data
    # x = np.concatenate((df_amazon["Response"].values, df_gibberish["Response"].values))
    # y = np.concatenate((df_amazon["Label"].values, df_gibberish["Label"].values))

    # # Check same length
    # print(x)
    # print(y)

    # # Review the mean length of the reviews vs the gibberish
    # df = pd.DataFrame({"Response": x, "Label": y}) \
    #     .groupby("Label")["Response"] \
    #     .apply(lambda x: print(np.mean(x.str.len())))
    #
    # x_train, x_test, y_train, y_test, = train_test_split(x, y, test_size=0.25)
    #
    #
    # from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
    #
    # model = GaussianNB()
    # model.fit(x_train, y_train)

    # from sklearn.feature_extraction.text import CountVectorizer
    # v = CountVectorizer()
    # x_train_count = v.fit_transform(x_train)
    # from sklearn.naive_bayes import MultinomialNB
    # model = MultinomialNB()
    # model.fit(x_train_count, x_train)
    #

    # test_data_count = v.transform(test_data)
    # print(model.predict(test_data_count))
    #
    # X_test_count = v.transform(x_test)
    # print(model.score(X_test_count, y_test))

    # clf = Pipeline([
    #     ('vectorizer', CountVectorizer()),
    #     ('nb', MultinomialNB)
    # ])
    # clf.fit(x_train, y_train)
    # # print(clf.score(X_test, Y_test))
    # #
    # # print(clf.score(X_test, Y_test)
    # # clf.predict(test_data)


if __name__ == '__main__':
    main()
