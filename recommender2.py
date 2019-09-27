def user_recommendation():
    import pandas as pd
    import os
    import numpy as np
    dir_ ='C:/Users/Ademola/Desktop/dataset/json_data'
    json_data = [i for i in os.listdir(dir_)]
    print(json_data)
    
    json_data = [json_data[0],json_data[4],json_data[5],json_data[7]]
    jn =[]
    #create a list of the dataframes
    for i in range(len(json_data)):
        jsn= pd.read_json('C:/Users/Ademola/Desktop/dataset/json_data/'+ json_data[i])
        jn.append(jsn)
    
    jj = jn[0].copy()
    jk = jn[2].copy()
    jl = jn[3].copy()
    for i in range (len(jn)):
        jn[i].dropna(axis=1,inplace = True)
    
    jn[2]['tags'] = jk['tags']
    jn[0]['display_message'] = jj['display_message']
    jn[3]['short_bio']= jl['short_bio']
    jn[3]['user_id']= jl['id']
    df1 = pd.merge_ordered(jn[0],jn[1],on='user_id')   
    df2 =pd.merge_ordered (jn[2],jn[3],on='user_id')
    df = pd.merge_ordered(df1,df2,on = 'user_id')
    
    features = ['display_message', 'short_bio','tags']
    for feature in features:
        df[feature] = df[feature].fillna('')
    
    df['features'] = df['display_message']+''+df['short_bio']+''+df['tags']
    df['combined_features'] =  [str.lower(i) for i in df['features']]
    #functions
    def get_combined_feature(user):
        feat = df[df.user_id==user].combined_features
        return feat.unique()[0]
    
    def get_index(f,act):
        return f[f==act].index.values
        
    def get_actions(user):
            act = df[df['user_id']== user].action
            act = list(act)
            f =[]
            for i in range(len(act)):
                if (act[i] =='Like' or act[i]=='Love' or act[i]== 'Followed'):
                    f.append(i) 
                else:
                    f = f
            acts= len(f)
            return acts
        
    def return_the_sender_id (user):
        #check for your user id among the sender id column to know the total
        #number of people you've sent notifications to
        user = df[df.sender_id==user].user_id
        return user.unique()
            
        
    def get_name_from_user_id (user):
        name = df[df.user_id==user].username
        return name.unique()[0]
    #sub main function
    def user_recommendation_base (user):
        #for a new user who hasn't made any actions yet:
        feat= get_combined_feature(user)
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['combined_features'])   
        from sklearn.metrics.pairwise import cosine_similarity
        cosine_sim = cosine_similarity(tfidf_matrix)
        
        ind = pd.Series(df.index, index=df['combined_features']).drop_duplicates()
        idx = ind[feat] 
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, reverse=True)
        sim_scores = sim_scores[0:]
        reco = [i[0] for i in sim_scores]
        gg = df['user_id'].iloc[reco]
        cg = df['action'].iloc[reco]
        ggg = pd.DataFrame([gg,cg],).transpose()
        cgg = ggg.groupby('user_id')['action'].count().sort_values(ascending = False)
        cb = []
        if len(cgg)< 11:
            for i in range(len(cgg)):
                cb.append(cgg.index[i])
        else:
            for i in range(0,11):
                cb.append(cgg.index[i])
        x = pd.Series(cb)
        return x.values
    #main function   
    def user_recommed(user):
        action = get_actions (user)
        #for new users
        if action==0:
               result = user_recommendation_base(user)
               return result
               
            #for a user who has already made an action:
        else:
                arr = return_the_sender_id(user)
                whys=[]
                exes = []
                for i in arr:
                    feat= get_name_from_user_id(i)
                    whys.append(feat)
                    #get recommedations based on popularly similar people u sent notifications
                    ex = user_recommendation_base(i)
                    ex = ex.astype(np.int32)
                    exes.append(ex)
                return exes
                return feat

    user = int(input('Input the user Id: '))

    output = user_recommed(user)
    print('\t',"Top 10 Recommended Users",'\n',output)
    
if __name__=='__main__':
  user_recommendation()





