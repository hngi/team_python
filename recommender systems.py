def content_recommender():
  #import some useful packages
  import pandas as pd
  import os
  #assign dir_ to the directory
  dir_ ='json_data'
  json_data = [i for i in os.listdir(dir_)]

  #get the title you want to use
  titles = input('Input the title of the post: ')
    
  # for content based recomender system, We'll use notifications,thoughts, following, contact
  json_data =[ json_data[2],json_data[4],json_data[5],json_data[6]]
  jn =[]

  #create a list of the dataframes
  for i in range(len(json_data)):
    jsn= pd.read_json(dir_+'/'+ json_data[i])
    jn.append(jsn)


  #save the tag column
  jj = jn[2].copy()
  jn[2].dropna(axis=1,inplace = True)
  jn[2]['tags'] = jj['tags']
  jn[1].dropna(axis=1,inplace = True)

  #get a dataframe
  df1 = pd.merge_ordered(jn[1],jn[2],on='user_id')    
  #get the columns we want to work with
  features = ['title', 'content', 'tags']
  for feature in features:
    df1[feature] = df1[feature].fillna('')
  
  #create a new column for the combination of the useful content
  df1['features'] = df1['title']+' '+df1['content']+' '+df1['tags']
  #get all  entries in small letters to avoid missing some words because of capital letter mismatch
  df1['titler'] = [str.lower(i) for i in df1['title']]
  df1['combine_features'] =  [str.lower(i) for i in df1['features']


  def get_title_from_index(index):
    titles = df1[df1.index == index].title
    return titles.values[0]


  try:
    titles = str.lower(titles)
    from sklearn.feature_extraction.text import CountVectorizer
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df1['combine_features'])
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_sim= cosine_similarity(count_matrix,count_matrix)
    ind = pd.Series(df1.index, index=df1['titler'])
 
    idx = ind[titles]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, reverse=True)
    sim_scores = sim_scores[0:]
    reco = [i[0] for i in sim_scores]
    gg = df1['title'].iloc[reco]
    cg = df1['action'].iloc[reco]
    ggg = pd.DataFrame([gg,cg],).transpose()
    cgg = ggg.groupby('title')['action'].count().sort_values(ascending = False)
    #return first 10 titles to be recommended based on popularity
    cb = []
    for i in range(0,10):
        cb.append(cgg.index[i])
    x = pd.Series(cb)
    return x
  except KeyError:
    cg = df1.groupby('title')['action'].count().sort_values(ascending = False)
    #return first 10 titles to be recommended based on popularity alone since your title couldn't be found
    cb = []
    for i in range(0,10):
        cb.append(cg.index[i])
    y = pd.Series(cb)
    return y

  #simple test for a title
  print('\t','Top Ten Recommended Articles','\n', content_recommender(titles))
                               
                              
                              
if __name__ == "__main__":
   content_recommender()                 
                        
