import pandas as pd
import os
import requests
import urllib.request
from bs4 import BeautifulSoup
dir_ ='json_data'
json_data = [i for i in os.listdir(dir_)]
print(json_data)
# for content based recomender system, I'll use notification,thoughts, following, contact
json_data =[ json_data[2],json_data[4],json_data[5],json_data[6]]
jn =[]
#create a list of the dataframes
for i in range(len(json_data)):
    jsn= pd.read_json('json_data/'+ json_data[i])
    jn.append(jsn)
jn[2].drop(['action'],axis=1,inplace= True)
df1 = pd.merge_ordered(jn[1],jn[2],on='id',how='inner')    
df1.head()

features = ['title', 'content', 'tags']
for feature in features:
    df1[feature] = df1[feature].fillna('')
    
df1['features'] = df1['title']+df1['content']+df1['tags']
df1['title'] = [str.lower(i) for i in df1['title']]
df1['combine_features'] =  [str.lower(i) for i in df1['features']]
df1['combine_features'] = [i.replace(" ","") for i in df1['combine_features']]
html = input()
def get_user_id_from_site(html):   
    url = html
    response = requests.get(url)
    seen = BeautifulSoup(response.text)
    metas = seen.find_all('meta')
    cd =[meta.attrs["content"] for meta in metas if 'name' in meta.attrs and meta.attrs['name'] == 'user_id']
    return cd

def title_from_site(html):
    url = html
    response = requests.get(url)
    seen = BeautifulSoup(response.content, "html.parser")
    metas2 = seen.find_all("meta")
    cg =[meta.attrs["content"] for meta in metas2 if 'property' in meta.attrs and meta.attrs['name'] == 'og.title']
    return cg

def get_title_from_id(id_):
    titles = df1[df1.id==id_].title
    return titles.values[0]

def get_title_from_index(index):
    titles = df1[df1.index == index].title
    return titles.values[0]

def check_action(id_):
    action = df1[df1.id==id_].action
    return action.values[0]
 
def content_recommender(titles,df1=df1):
    titles = str.lower(titles)
    from sklearn.feature_extraction.text import CountVectorizer
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df1['combine_features'])
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_sim= cosine_similarity(count_matrix,count_matrix)
    ind = pd.Series(df1.index, index=df1['combine_features'])
 
    idx = ind[df1['combine_features']] 
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, reverse=True)
    sim_scores = sim_scores[1:50]
    reco = [i[0] for i in sim_scores]
    gg = df1['title'].iloc[reco]
    cg = df1['action'].iloc[reco]
    ggg = pd.DataFrame([gg,cg],).transpose()
    cgg = ggg.groupby('title')['action'].count().sort_values(ascending = False)
    return cgg.index[1:11].values

def action_recommender(id_):
    action = check_action(id_)
    if (action=='Liked' or action=='Loved' or action=='commented'):
        title = get_title_from_id(id_)
        content_recommender(title,df1)
    else:
        return None
    
test = content_recommender('Task 2')


