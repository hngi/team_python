import pandas as pd
import os

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
#save the tag column
jj = jn[2].copy()
jn[2].dropna(axis=1,inplace = True)
jn[2]['tags'] = jj['tags']
jn[1].dropna(axis=1,inplace = True)
#get a dataframe
df1 = pd.merge_ordered(jn[1],jn[2],on='id',how='inner')    
df1.head()

features = ['title', 'content', 'tags']
for feature in features:
    df1[feature] = df1[feature].fillna('')
    
df1['features'] = df1['title']+df1['content']+df1['tags']
df1['title'] = [str.lower(i) for i in df1['title']]
df1['combine_features'] =  [str.lower(i) for i in df1['features']]
df1['combine_features'] = [i.replace(" ","") for i in df1['combine_features']]

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
    #return first 10 titles to be recommended
    cb = []
    for i in range(1,11):
        cb.append(cgg.index[i])
    x = pd.Series(cb)
    return x


def action_recommender(id_):
    action = check_action(id_)
    if (action=='Liked' or action=='Loved' or action=='commented'):
        title = get_title_from_id(id_)
        content_recommender(title,df1)
    else:
        return None
#simple test    
test = content_recommender('Task 2')


