import statistics

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import scipy
import scipy.stats

# now you can use
scipy.stats.poisson
# if you want it more accessible you could do what you did above
from scipy.stats import poisson

# then call poisson directly
poisson

if __name__ == '__main__':
    data_volume = 10
    # Read dataset and get given columns
    df = pd.read_csv("./dataset/Webpages_Classification_test_data.csv", sep=',',
                     usecols=['url', 'https', 'js_len', 'js_obf_len', 'label'])
    # Sort data
    df = df.sort_values(
        by="label",
        ascending=False
    )
    df.head()
    # Exclude invalid data rows with null values
    df_total = df[df['js_len'] != 0.0]

    # Devide in two subgroups by label
    df_bad = df_total[df_total['label'] == 'bad']
    df_good = df_total[df_total['label'] == 'good']

    # Get sample of with data_volume
    # df_bad = df_bad.sample(n=data_volume)
    # df_good = df_good.sample(n=data_volume)

    # region label bad and js_len
    total_label_bad = len(df_bad) / len(df_total) * 100
    total_label_good = len(df_good) / len(df_total) * 100
    distribution_percentage_label = total_label_good, total_label_bad

    # region  Distribution
    print("Distribution dataset label (good, bad)")
    print("Total label bad: ", str(round(total_label_bad, 2)))
    print("Total label good: ", str(round(total_label_good, 2)))

    labels = 'good', 'bad'
    colors = ['green', 'red']
    plt.pie(distribution_percentage_label, labels=labels, autopct='%1.1f%%', colors=colors)
    plt.title('Distribution dataset label (good, bad)')
    plt.axis('equal')
    plt.show()
    # endregion

    # region Boxplot Bad
    print("Boxplot Bad data:")
    df_bad_js_len = df_bad['js_len']
    mean_ban_js_len = statistics.fmean(df_bad_js_len)
    median_ban_js_len = statistics.median(df_bad_js_len)
    print("Mean:", mean_ban_js_len)
    print("Median:", median_ban_js_len)

    df_bad.boxplot(column=['js_len'])
    plt.show()
    # endregion

    print(df_bad.to_string())

    # region Boxplot Good
    print("Boxplot Good data:")
    df_good_js_len = df_good['js_len']
    mean_good_js_len = statistics.fmean(df_good_js_len)
    median_good_js_len = statistics.median(df_good_js_len)
    print("Mean:", mean_good_js_len)
    print("Median:", median_good_js_len)

    df_good.boxplot(column=['js_len'])
    plt.show()
    # endregion

    # region Boxplot Total
    df_total.boxplot(column='js_len', by='label')
    plt.show()
    # endregion

    # region histogram total
    df_total.groupby('label')['js_len'].plot(kind='kde', legend=True)
    plt.show()
    # endregion

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.scatter(x=df_total['js_len'], y=df_total['label'])
    plt.xlabel("Label (Good, Bad)")
    plt.ylabel("js_len")

    plt.show()
    # endregion

    # region bad & porn
    bad_words = ['adult', 'porn', 'ass', 'tits', 'xxx', 'dildo', 'naked', 'sex', 'slave', 'naughty', 'fetish',
                 'hardcore', 'escort', 'sluts', 'pussy', 'porno', 'eroti']
    df_bad_filtered = df_bad[~df_bad.url.str.contains('|'.join(bad_words)).groupby(level=0).any()]
    total_label_bad_porn = 100 - (len(df_bad_filtered) / len(df_bad) * 100)
    total_label_bad = len(df_bad_filtered) / len(df_bad) * 100
    distribution_percentage_label = total_label_bad_porn, total_label_bad

    # region  Distribution
    print("Distribution dataset label (bad, porn & bad)")
    print("Total label bad but normal: ", str(round(total_label_bad, 2)))
    print("Total label porn & bad: ", str(round(total_label_bad_porn, 2)))

    labels = 'normal url & bad', 'porn url & bad'
    colors = ['red', 'orange']
    plt.pie(distribution_percentage_label, labels=labels, autopct='%1.1f%%', colors=colors)
    plt.title('Distribution dataset label (bad, bad & porn) url')
    plt.axis('equal')
    plt.show()
    # endregion
    # endregion



"""
    Prep data 
    (1) Make random sample                                                      <- Anthon          
    (2) Select from df["label"] random select n of 'bad' and n of 'good'        <- Rik
    (3) Label url probability of gibberish                                      <- Rik & Anthon
    (4) Exclude 0 values 
    
    
    Analyse data
        (1) find correlation between the labels
            research how to find correlation between data 
            
        
        note: make visualization readable 
    
    Notebook
    
        Title
        Abstract
        Population 
            js_len is in kb 
            
        Testing 
            Testing order
        
        Data visualization 
            boxplot
            scatter plot
            subscatter plot
            density plot
            histogram 
            
            
        significance beta alfa confidence interval 
        
        Hypothesis testing
            Formulate hypothesis 
            H0 : no correlation
            H
            1 : first correlation | malware == big js volume 
            H2 : second correlation | malware == no ssl 
            H3 : third correlation  | malware == gibberish      p(x >= waarde | h1 true)
        
            check assumption 
            test hypothesis needs to be 99% otherwise it needs to be rejected 
            suppose the hypothesis is true
            
            qq plot
                normal distribution 
            
            data -> conclusion/hypothesis most likely h0 prior likelihood 
            
            assume h0 => p(data) what is the likelihood if this is true that i will get the same data
            
            H0 : =
            H1 : != 
            
            t-table 
                has to be more dan 5% to accept conclusion/hypothesis
            t-test
                alphavalue  = 0.05
            
            t-value 
            failed to reject 0 hypothesis 
            
        conclusion 
        
"""
