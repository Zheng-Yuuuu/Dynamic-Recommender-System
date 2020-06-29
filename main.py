import numpy as np
import init
import function as ft
import sampling as sp
# import lh
# import lh_cur_route as lhc
# import matplotlib.pyplot as plt
import pred

route_index = np.load('route_index.dat')
node_mat_dis = np.load('node_mat_dis.dat') 
one_mat = np.load('one_mat.dat')
route_num_node_to_node = np.load('route_num_node_to_node.dat')
# stamp_route_index = np.load('stamp_route_index.dat')
# map_tran = np.load('map_tran.dat')
ob = np.load('ml1_density20.dat')




zero_index = np.zeros(np.shape(ob)[2])
for i in range(np.shape(ob)[2]):
    if len(np.where(ob[:,:,i]>-1)[0]):
        pass
    else:
        zero_index[i]=1
 
ob = np.delete(ob, np.where(zero_index==1)[0], 2)




ob_real = np.load('ml1main.dat')
ob_real = np.delete(ob_real, np.where(zero_index==1)[0], 2)

print(np.shape(ob))
print(len(np.where(ob_real>-1)[0]))
print(len(np.where(ob==-2)[0]))
print(len(np.where(ob>-1)[0]))



num_node = 21
sample_iter = 200 #sample iteration 
burn_iter = 100 #burn iteration
eta = 1

emis_prior = np.ones(num_node)*eta  #emission probability_prior
tran_prior = np.zeros((num_node,num_node)) #emission probability_prior

x,y = np.where(node_mat_dis>0)
tran_prior[x,y] = eta
np.fill_diagonal(tran_prior, eta)

user_num = np.shape(ob)[1]
item_num = np.shape(ob)[2]
rate_mat_x = 20
rate_mat_y = 20
score_num = 10
time_span = np.shape(ob)[0]


emis_sample = np.zeros(num_node)
tran_route_user_sample = np.zeros((\
sample_iter,user_num, time_span-1, 2),dtype=int)
# 1-sample index 2-user index 3-time index 4.1 route index 4.2 division_index
rate_mat_prior = eta
state_prior, item_prior, rate_prior = np.ones(3)*eta

#current
node_tran_sample = np.zeros((num_node,num_node),dtype=int)
n_ab_mat = np.zeros((rate_mat_x,rate_mat_y,score_num),dtype=int)
z_ijl = np.ones((user_num,item_num,num_node,2),dtype=int)*(-1) #1 state group 2 item group


#history
state_group_hist = np.zeros((sample_iter,num_node,rate_mat_x),dtype=int)
n_ab_mat_hist = np.zeros((sample_iter,rate_mat_x,rate_mat_y,score_num),dtype=int)
z_i = np.zeros((time_span,user_num,rate_mat_x),dtype=int)
z_j_hist = np.zeros((sample_iter,rate_mat_y,item_num),dtype=int)
node_tran_sample_hist = np.zeros((sample_iter,num_node,num_node),dtype=int)
emis_sample_hist = np.zeros((sample_iter,num_node))

'''
initial parameter
'''
init.init(user_num, route_num_node_to_node, time_span,tran_route_user_sample,
         route_index,num_node, node_tran_sample, emis_sample,rate_mat_x,
         rate_mat_y,z_ijl,ob,n_ab_mat,tran_prior,eta,z_j_hist,state_group_hist,z_i)


# print(node_tran_sample,'node_tran_sample')
# print(emis_sample,'emis_sample')

'''
record
'''

node_tran_sample_hist[0,:,:] = np.copy(node_tran_sample)
emis_sample_hist[0,:] = np.copy(emis_sample)
n_ab_mat_hist[0,:,:,:] = np.copy(n_ab_mat)




'''
check
'''
check_change = np.zeros(item_num)
for i in range(item_num):
    ai = np.where(ob[:,:,i]>-1)[0]
    check_change[i] = len(ai)
    


qq = np.where(ob==-2)[0]
print(len(qq),'geshu')



state_group = np.zeros((num_node,rate_mat_x))
item_group = np.zeros((rate_mat_y,item_num))

for i in range(num_node):
    index, counts = np.unique(z_ijl[:,:,i,0], return_counts = True)
    if index[0]==-1:
        index = np.delete(index, 0, 0)
        counts = np.delete(counts, 0, 0)
    if len(index):
        state_group[i,index] = state_group[i,index]+counts

for i in range(item_num):
    index, counts = np.unique(z_ijl[:,i,:,1], return_counts = True)
    if index[0]==-1:
        index = np.delete(index, 0, 0)
        counts = np.delete(counts, 0, 0)
    if len(index):
        item_group[index,i] = item_group[index,i]+np.transpose(counts)




'''
sampling
'''
sp.sampling(sample_iter,burn_iter,user_num,emis_sample,tran_route_user_sample,ob,
             node_tran_sample,n_ab_mat,z_ijl,num_node,tran_prior,route_index,one_mat,
             time_span, emis_prior,score_num,state_prior, item_prior,
             rate_prior,rate_mat_x,rate_mat_y,state_group_hist,n_ab_mat_hist,z_j_hist,
             node_tran_sample_hist,emis_sample_hist,z_i,state_group,item_group)

'''
test init
'''
# index_1 = 0
# index_2 = 0
# x = 0
# y = 0
# for j in range(rate_mat_x):
#     for m in range(rate_mat_y):
#         
#         index_x,index_y,index_z = np.where(z_ijl[:,:,:,0]==j)
#         if len(index_x):
#             index_1 = np.where(z_ijl[index_x,index_y,index_z,1]==m)[0]
#             x = x+len(index_1)
#         index_2 = np.sum(n_ab_mat[j,m,:])    
#         y = y+index_2
#         if len(index_1)==index_2:
#             print(len(index_1),index_2,j,m)
#         else:
#             print(len(index_1),index_2,j,m)
#             
# s = np.where(ob!=(-1))[0]
# print(len(s),x,y)                

# com = np.zeros((num_node,num_node))
# for i in range(user_num):
#     for j in range(time_span-1):
#         r = np.copy(route_index[s[i,j,0].astype(int),s[i,j,1].astype(int),:])
#         
#         com = com+np.reshape(r,(num_node,num_node))  
# for i in range(num_node):
#     for j in range(num_node):
#         if node_tran_sample[i,j]!=com[i,j]:
#             print(node_tran_sample[i,j],com[i,j])
#             print('wrong')
#             
# print(np.sum(node_tran_sample))

'''
test ft.tran_prob passed!
'''

# prob_mat,prob_arr = ft.tran_prob(tran_prior,node_tran_sample,num_node,
#                                          route_index,one_mat)



tran_route_user_sample.dump('tran_route_user_sample.dat')
state_group_hist.dump('state_group_hist.dat')
z_j_hist.dump('z_j_hist.dat')
node_tran_sample_hist.dump('node_tran_sample_hist.dat')
emis_sample_hist.dump('emis_sample_hist.dat')
n_ab_mat_hist.dump('n_ab_mat_hist.dat')
'''
training likelihood
'''
# tlh = np.zeros(sample_iter)
#    
# for i in range(sample_iter):
#     state_mat = np.copy(state_group_hist[i,:,:])
#     item_mat = np.copy(z_j_hist[i,:,:])
#     tran_node = np.copy(node_tran_sample_hist[i,:,:])
#     emis_sample = np.copy(emis_sample_hist[i,:])
#     n_ab_mat = np.copy(n_ab_mat_hist[i,:,:,:])
#          
#     tlh[i] = lh.llh(n_ab_mat, state_mat, item_mat, ob, tran_node, emis_sample, tran_prior, 
#        emis_prior, num_node, rate_prior, item_num, score_num, state_prior, item_prior, 
#        rate_mat_y, rate_mat_x, route_index, one_mat, time_span)
#       
# print(tlh)
#     
#     
# x = np.arange(sample_iter)
# plt.plot(x,tlh)
# plt.show()


'''
training likelihood using current route
'''

# tlh = np.zeros(sample_iter)
#    
# for i in range(sample_iter):
#     state_mat = np.copy(state_group_hist[i,:,:])
#     item_mat = np.copy(z_j_hist[i,:,:])
#     tran_node = np.copy(node_tran_sample_hist[i,:,:])
#     emis_sample = np.copy(emis_sample_hist[i,:])
#     n_ab_mat = np.copy(n_ab_mat_hist[i,:,:,:])
#          
#     tlh[i] = lhc.llh(n_ab_mat, state_mat, item_mat, ob, tran_node, emis_sample, tran_prior, 
#        emis_prior, num_node, rate_prior, item_num, score_num, state_prior, item_prior, 
#        rate_mat_y, rate_mat_x, route_index, one_mat, time_span,
#        np.copy(tran_route_user_sample[i,:,:,:]))
#       
# print(tlh)
#     
#     
# x = np.arange(sample_iter)
# plt.plot(x,tlh)
# plt.show()





'''
prediction 
'''


pred.pd(z_i, z_j_hist, n_ab_mat_hist, ob, rate_prior, ob_real,burn_iter)
