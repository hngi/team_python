def user_recommendation():
    #import libraries
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    #reading the json file
    ds = pd.read_json("C:/json_data/lucid_table_users.json")
    #renaming the empty rows with space
    ds = ds.fillna(' ')

    #printing the rows under the short bio
    #print(ds['short_bio'])

    #analyzing the words in the column and removing common stop words, calculating the cosine similarities
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(ds['short_bio'])

    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    results = {}

    for idx, row in ds.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
        similar_items = [(cosine_similarities[idx][i], ds['username'][i]) for i in similar_indices]

        results[row['username']] = similar_items[1:]
        
    def item(username):
        return ds.loc[ds['username'] == username]['short_bio'].tolist()[0].split(' - ')[0]

    # a function that reads the results out of the column and the amount of results wanted.  
    #the username that the recommendation would acted upon
    uu= str(input('Input the username: '))
    #num = int(input('Input the amount of people to be recommmended: '))
    def recommended(item_username, num):
        print("Recommending " + " people similar to " + item(item_username) + "...")
        print("-------")
        recs = results[item_username][:num]
        for rec in recs:
            print("Recommended: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")
            gg =(ds.loc[ds['short_bio'] == item(rec[1])])
            nn = gg['name']
            username = gg['username']
            print(nn + ", username " +username)
    recommended(item_username=uu, num=3)

if __name__=='__main__':
  user_recommendation()
