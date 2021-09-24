import pandas as pd

if __name__ == '__main__':
    data_volume = 50
    # Read dataset and get given columns
    df = pd.read_csv("./dataset/Webpages_Classification_test_data.csv", sep=',',
                     usecols=['url', 'https', 'js_len', 'js_obf_len', 'label'])
    # Sort data
    df = df.sort_values(
        by="label",
        ascending=False
    )

    # Devide in two subgroups by label
    df_bad = df[df['label'] == 'bad']
    df_good = df[df['label'] == 'good']

    # Get sample of with data_volume
    df_bad_sample = df_bad.sample(n=data_volume)
    df_good_sample = df_good.sample(n=data_volume)

    print(df_bad_sample.to_string())
    print(df_good_sample.to_string())



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
        
        Hypothesis testing
            Formulate hypothesis 
            H0 : no correlation
            H1 : first correlation
            H2 : second correlation
            H3 : third correlation
        
            check assumption 
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
