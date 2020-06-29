import numpy as np 



ob = np.load('ml1main.dat')

t_all = np.shape(ob)[0]
user_all = np.shape(ob)[1]
movie_all = np.shape(ob)[2]

new_ob = np.zeros((user_all,movie_all),dtype = int)-1

for t in range(t_all):
    for user in range(user_all):
        for movie in range(movie_all):
            if ob[t,user,movie]>-1:
                new_ob[user,movie] = np.copy(ob[t,user,movie])

s = np.sum(new_ob,0)
index_movie = np.where(s==(np.shape(new_ob[0])*(-1)))[0]
new_ob = np.delete(new_ob, index_movie, 1)
s = np.sum(new_ob,1)
index_user = np.where(s==(np.shape(new_ob[1])*(-1)))[0]
new_ob = np.delete(new_ob, index_user, 0)


user,movie = np.where(new_ob>-1)
print(len(user))
rating = np.zeros((len(user),3))

rating[:,0] = np.copy(user)
rating[:,1] = np.copy(movie)
rating[:,2] = np.copy(new_ob[user,movie])

np.save('ml_rating', rating)






















