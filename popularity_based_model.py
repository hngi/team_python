#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Created on Tue Sep 24 18:22:58 2019

@author: lumi
"""
#%%

# Import needed modules
import pandas as pd

# Reading Users csv file and storing data in users and followers
users = pd.read_csv('/Users/lumi/Downloads/table_for_users - table_for_users.csv')
following = pd.read_csv( '/Users/lumi/Downloads/Table for following - Sheet1.csv')

# a function that tells pandas to display all the fields
def display_all(df):
    with pd.option_context("display.max_rows", 1000,  "display.max_columns", 1000,):
        display(df)


#get follower for each user id

followers_per_users = following.groupby('my_id').my_id.apply(lambda x: len(x))
followers_per_users.to_frame().describe()

#popularity model. it recommeds the most followed people

users_followers = following.groupby('my_id').size().reset_index(name='users')
users_popularity = following.groupby('my_id')['follower_id'].sum().sort_values(ascending=False).reset_index()
users_popularity = pd.merge(users_popularity, users_followers, how='inner', on=['my_id'])
users_popularity = pd.merge(users_popularity, following[['my_id', 'follower_id', 'status']], how='inner', on=['my_id'])
users_popularity = users_popularity.sort_values(by=['follower_id_y'], ascending=False)
users_popularity.head()






# To copy sql file into a csv file.
#def copy_file(infilename, outfilename):
#    """ Opens two files and copies one into the other line by line. """
#    infile = open(infilename)
#    outfile = open(outfilename,'w')
#    
#    for line in infile:
#        outfile.write(line)
#        
#    infile.close()
#    outfile.close()


