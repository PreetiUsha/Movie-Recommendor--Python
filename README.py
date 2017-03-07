# Loading libraries
from math import *
import pandas as pd
import sklearn
from sklearn.cross_validation import KFold
from sklearn import cross_validation as cv
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
import warnings
import numpy as np


#Loading data
rating=pd.read_csv("D:\\Data_Mining\\CUS695 Project\\Data\\ratings.csv")
rating= rating.drop('timestamp', 1)
movies = pd.read_csv("D:\\Data_Mining\\CUS695 Project\\Data\\movies.csv")
#rating.head()
#len(rating)

n_users = rating.userId.unique().shape[0]
n_movies = rating.movieId.unique().shape[0]
print ('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_movies) )

#Spliting data into train and test
train_data, test_data = cv.train_test_split(rating, test_size=0.25)

#Creating matrix
ratings = np.zeros((668, 149532))
for row in rating.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]
ratings

train_data_matrix  = np.zeros((668, 149532))
for line in train_data.itertuples():
    train_data_matrix [line[1]-1, line[2]-1] = line[3]
    
test_data_matrix = np.zeros((668, 149532))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

#Mapping userid and movieid with ratings
M=rating.pivot_table(index=['movieId'],columns=['userId'],values='rating')
M.fillna(0, inplace=True)
M.head()


#Cosine similarity function
#user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
#osine_train=sklearn.metrics.pairwise.cosine_similarity(train_data_matrix,None,dense_output=False)
user_similarity = sklearn.metrics.pairwise.cosine_similarity(train_data_matrix,None,dense_output=False)
user1_similarity = sklearn.metrics.pairwise.cosine_similarity(train_data_matrix,None,dense_output=False)


#Taking user input
u1=int(input("Enter the user id: "))

#Calculating closest user to inputted user
nval=[]
mval=[]
nval=user1_similarity[u1-1][0:]
mval=user_similarity[u1-1][0:]
mval.sort()

first,second, third = mval[-2],mval[-3],mval[-4]

for i in range(0,len(nval)):
    if nval[i]==first:
        fst=i-1
        print("The most closest user is:", i-1)
        
for i in range(0,len(nval)):
    if nval[i]==second:
        sec=i-1
        print("The second closest user is:", i-1)
        
for i in range(0,len(nval)):
    if nval[i]==third:
        thd=i-1
        print("The third closest user is:", i-1)


usrgen=rating[rating.userId==u1]
usrgen1=rating[rating.userId==fst]
usrgen2=rating[rating.userId==sec]
usrgen3=rating[rating.userId==thd]
def calculation(usrgen,usrgenx):
    a=[]
    df1=usrgen[["movieId"]]
    dfr=usrgen[["rating"]]
    df2=usrgenx[["userId","movieId","rating"]]
    df=df1.merge(df2,how='inner')
    #a=df.movieId!=df2.movieId
    a = df[["movieId"]]
    dfk = pd.DataFrame(np.nan,index=range(0,len(df2)-len(df)),columns=['movieId'])
    dfk.fillna(value=0)
    a=a.append(dfk,ignore_index=True)
    a=a.fillna(value=0)
    b=df2.movieId!=a.movieId
    c=df2.rating>3.0
    dff=df2[b&c]
    dff=dff.sort('rating',ascending=False)
    lst=[]
    for i in range(0,len(dff)):
        dfft=dff["movieId"].iloc[i]
        ratavg=rating[rating.movieId==dfft]
        lst.append(ratavg["rating"].mean())
    dff['avgrating'] = lst
    dff_final=dff.sort('avgrating',ascending=False)
    return dff_final[0:1]

lstn = pd.DataFrame(np.nan,index=range(0),columns=['userId','movieId','rating','avgrating'])
lstn.fillna(value=0)
usrgenx=usrgen1
#calculation(usrgen,usrgenx)
lstn=lstn.append(calculation(usrgen,usrgenx))
usrgenx=usrgen2
#calculation(usrgen,usrgenx)
lstn=lstn.append(calculation(usrgen,usrgenx))
usrgenx=usrgen3
#calculation(usrgen,usrgenx)
lstn=lstn.append(calculation(usrgen,usrgenx))

#Predicting movies
def replace_name(x):
    return movies[movies["movieId"]==x].title.values[0]
lstn.movieId=lstn.movieId.map(replace_name)
lstn=lstn.sort('avgrating',ascending=False)

finalmovies=[]
for i in range(0,len(lstn)):
    fin=lstn["movieId"].iloc[i]
    finalmovies.append(fin)
print("The predicted movies for user " + str(u1) + ": " + str(finalmovies[0]) + 
      ","+ str(finalmovies[1]) + " and " + str(finalmovies[2]))


#Predicting ratings for the predicted movies
finalavgrating=[]
for i in range(0,len(lstn)):
    finravg=lstn["avgrating"].iloc[i]
    finravg=round(finravg,1)
    finalavgrating.append(finravg)


finalrating=[]
for i in range(0,len(lstn)):
    finr=lstn["rating"].iloc[i]
    finr=round(finr,1)
    finalrating.append(finr)


print("The predicted rating for " +  str(finalmovies[0]) + " should be between " + str(finalavgrating[0]) 
      + " and " + str(finalrating[0]))
print("The predicted rating for " +  str(finalmovies[1]) + " should be between " + str(finalavgrating[1]) 
      + " and " + str(finalrating[1]))
print("The predicted rating for " +  str(finalmovies[2]) + " should be between " + str(finalavgrating[2]) 
      + " and " + str(finalrating[2]))


lstnn=[]
def replace_name(x):
    return movies[movies["movieId"]==x].title.values[0]
a = test_data.userId==179
#b = train_data.movieId==1223
lstnn=test_data[a]

lstnn.movieId=lstnn.movieId.map(replace_name)
lstnn
CUS 695 Capstone Final Code.txt
Open with Google Docs
