import urllib3
import json

import pandas as pd
import numpy as np
from scipy.stats import chi2

import matplotlib.pyplot as plt

from prettytable import PrettyTable

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from langdetect import detect_langs

import re
import string

from collections import Counter

from warnings import warn

# Construct API query for Springer Nature Meta API through dictionary
def springer_request(query,first_item,number_items,api_key,output='json'):

    # base api url and chosen output language
    api_url = 'http://api.springer.com/meta/v2/{}?'.format(output)
    
    # api key for springer
    api_url += 'api_key={}&'.format(api_key)
    
    # query construction
    api_url += 'q='
    for key, value in query.items():
        value_string = '+'.join(value.split(' '))
        api_url += '{}:"{}" AND '.format(key,value_string)
    api_url = api_url[:-5] + '&'
    
    # start from item
    api_url += 's={}&'.format(first_item)
    
    # show # of items
    api_url += 'p={}'.format(number_items)
    
    return api_url

# Search the online corpus using a query and save abstract and keywords in dataframe
def create_dataframe(query,api_key,max_number_shown=100):
    
    http = urllib3.PoolManager()
    
    # Get the total number of items that satisfy the query
    api_string = springer_request(query,1,1,api_key)
    request = http.request('GET',api_string)
    
    items = json.loads(request.data.decode('utf-8'))
    number_of_items = int(items['result'][0]['total'])

    # Load the items satisfying the query into a dataframe
    rows = []
    
    for i in range(1+number_of_items//max_number_shown):
        
        api_string = springer_request(query,1+i*max_number_shown,max_number_shown,api_key)
        request = http.request('GET',api_string)
        
        if request.status != 200:
            warn('request to Nature dataset has been unsuccesfull.')
            continue
            
        items = json.loads(request.data.decode('utf-8'))

        rows += [[item['title'],item['abstract'],item['keyword']] for item in items['records']]
        
    df = pd.DataFrame(rows,columns=['title','abstract','keywords'])
    
    return df

# check if some entries are missing a value in column
def missing_entries(dataframe,column):
    # total number of items
    total_items,_ = dataframe.shape
    
    # check if entry is missing
    entry_length = dataframe[column].apply(len).to_numpy()
    is_entry_missing = entry_length == 0
    
    missing_entries = np.sum(is_entry_missing)
    
    print('There are {} missing {} out of {}'.format(missing_entries,column,total_items))
    
    return is_entry_missing

# clean the tags by lowering all characters, selecting alphanum words, stemming them, remove stopwords
def clean_tags(tag_list,stemmer,stopwords):
    
    cleaned_tags = []

    for tag in tag_list:
        tag = tag.lower()
        tag = ' '.join([stemmer.stem(word) for word in re.findall('\w+',tag) if word not in stopwords])

        if len(tag) != 0:
            cleaned_tags.append(tag)
        
    return np.array(cleaned_tags)

# Create a tree of tags by exploiting the hirarchical nature of tags
def build_tag_tree(tree,parent_id,used_tags=[],depth=1,max_depth=5,min_size=100):
    
    tags_df = tree.get_node(parent_id).data
    
    visited_tags = []
    node_ids = []
    
    while not tags_df.empty:
        
        # Count the frequency of each tag
        tags = np.concatenate(tags_df['keywords'].to_numpy())
        tags_frequency = Counter(tags)
        
        # Select the most frequent not used yet
        for subtag,_ in tags_frequency.most_common():
            if subtag in visited_tags or subtag in used_tags:
                continue
            visited_tags.append(subtag)
            break
        
        # Select item in dataframe with subtag
        filter_subtag = lambda item: np.isin(subtag,item,assume_unique=True)
        contains_subtag = tags_df['keywords'].apply(filter_subtag)
        selected_tags_df = tags_df[contains_subtag]
        selected_size,_ = selected_tags_df.shape
        
        # If the items with the selected tag are just a few, stop iterating
        if selected_size < min_size:
            break
        
        # Create node in the tree
        node = tree.create_node(subtag,parent=parent_id,data=selected_tags_df)
        node_ids.append(node.identifier)
        
        # Repeat with remaning tags
        tags_df = tags_df[~contains_subtag]
        
    if depth > max_depth:
        return
        
    for tag,node_id in zip(visited_tags,node_ids):
        build_tag_tree(tree,node_id,used_tags+[tag],depth+1,max_depth,min_size)
        
# check if an item in dataframe is written in english with probability above a certain treshold
def is_foreign(item,treshold=0.7):
    
    languages = detect_langs(item)
    
    for lang in languages:
        if lang.lang == 'en' and lang.prob > treshold:
            return False
    
    return True

# clean the text and use lemmatizer
def clean_text_lemmatize(item,lemmatizer):
    
    # remove latex equations
    item = re.sub('\$+.*?\$+','',item)
    
    # tokenize item and remove punctuation
    item = [word for word in word_tokenize(item) if word not in string.punctuation]
    
    # lowecase everything and check if it is alphanumeic (or contains an aphon)
    item = [word.lower() for word in item if word.isalnum() or ('-' in word)]
    
    # remove english stopwords
    item = [word for word in item if word not in stopwords.words('english')]
    
    # lemmatize the words
    item = [lemmatizer.lemmatize(word) for word in item]
    
    return item

# get the frequency of each label in the database
def get_frequency_df(df,labels):
    
    N,_ = df.shape
    bars = []
    
    for tag in labels:
        N_tag,_ = df[df['keywords']==tag].shape
        bars.append(N_tag/N)
        
    return bars

# create bar plot with feature importance (works for RF and Boosting)
def plot_feature_importance(importance,number_features=20):
    
    importance_values = [importance[i][0] for i in range(number_features)]
    features_name = [importance[i][1] for i in range(number_features)]

    positions = np.arange(number_features)
    plt.barh(positions, importance_values)
    plt.yticks(positions,features_name)
    
    plt.xlabel('feature importance')

# implement the McNemar test
def mc_nemar_test(model1,model2,X_test,y_test):
    
    # instances where model 1 (model 2) is correct
    correct_m1 = model1.predict(X_test) == y_test
    correct_m2 = model2.predict(X_test) == y_test
    
    # Conditioning on correct instances for model 1, get correct (error) instances for model 2
    c_m1 = correct_m2[correct_m1].size

    a = np.sum(correct_m2[correct_m1])
    b = c_m1 - a

    # Conditioning on error instances for model 1, get correct (error) instances for model 2
    e_m1 = correct_m2[~correct_m1].size

    c = np.sum(correct_m2[~correct_m1])
    d = e_m1 - c
    
    # save the table with the performance comparison
    table = np.array([[a,b,a+b],[c,d,c+d],[a+c,b+d,c_m1+e_m1]])
    
    # Test statistics and p-value
    X = (b-c)**2/(b+c)
    p_value = 1 - chi2.cdf(X,df=1)
    
    return p_value,table

# Print contingency matrix generated by McNemar test
def print_table(table):

    contingency_table = PrettyTable()
    contingency_table.field_names = ['M1\M2', 'C2', 'E2', 'sum row']
    
    row_names = ['C1','E1','sum col']
    
    for name,row in zip(row_names,table):
        contingency_table.add_row(np.concatenate(([name],row)))
        
    print(contingency_table)

# Print report on statistical significance of different models accuracy (using McNemar test)    
def models_comparison(models,names,X_test,y_test,alpha=0.05):

    n_models = len(models)

    print('Null hypothesis (the two tests are equivalent) is rejected for p_val < {:.3f}'.format(alpha))

    for _ in range(n_models):
        
        model1 = models.pop()
        name1 = names.pop()
        
        for model2,name2 in zip(models,names):

            # Perform McNemar test on the two models
            p_value, contingency_table = mc_nemar_test(model1,model2,X_test,y_test)

            # print the results (p_value and contingency table)
            print('---')
            print('M1 = {} vs M2 = {}\n'.format(name1,name2))
            print_table(contingency_table)
            print('')

            if p_value < alpha:
                print('Null hypothesis rejected - models are significantly different (p_val= {:.3f})'.format(p_value))
            else:
                print('Null hypothesis cannot be rejected (p_val= {:.3f})'.format(p_value))