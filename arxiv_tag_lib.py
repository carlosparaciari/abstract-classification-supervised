import urllib3
import json

import pandas as pd
import numpy as np

import re

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

        rows += [[item['abstract'],item['keyword']] for item in items['records']]
        
    df = pd.DataFrame(rows,columns=['abstract','keywords'])
    
    return df

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