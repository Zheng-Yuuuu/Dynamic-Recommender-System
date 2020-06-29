import numpy as np
import function as ft
import gs_function as gft

'''
1 calculate trans and emis probablity
'''
def sampling(sample_iter,burn_iter,user_num,emis_sample,tran_route_user_sample,ob,
             node_tran_sample,n_ab_mat,z_ijl,num_node,tran_prior,route_index,one_mat,
             time_span, emis_prior,score_num,state_prior, item_prior,
             rate_prior,rate_mat_x,rate_mat_y,state_group_hist,n_ab_mat_hist,z_j_hist,
             node_tran_sample_hist,emis_sample_hist,z_i,state_group,item_group):
    
    
#     if np.sum(check_change) == len(np.where(z_ijl[:,:,:,1]>-1)[0]):
#             print('rorororororororororo')
#     for i in range(len(check_change)):
#         if check_change[i]!=len(np.where(z_ijl[:,i,:,1]>-1)[0]):
#             print('rororoororororoororororoororororororroorororororororoor')
    for i in range(1,sample_iter):
        print(i)
#         if np.sum(z_j_hist[i-1,:,:])!=np.sum(check_change):
#             print(i,'i wrong')
        prob_mat,prob_arr = ft.tran_prob(tran_prior,node_tran_sample,num_node,
                                         route_index,one_mat)
        
        emis_sample, node_tran_sample = \
        gft.backward(prob_arr, prob_mat, time_span, tran_route_user_sample,
             ob, num_node, emis_prior, i,
             emis_sample, n_ab_mat,z_ijl,score_num,state_prior, item_prior,
             route_index,user_num,rate_prior,rate_mat_x,rate_mat_y,z_i,burn_iter,
             state_group,item_group)  
            
        '''
        record hist
        '''
        node_tran_sample_hist[i,:,:] = np.copy(node_tran_sample)
        emis_sample_hist[i,:] = np.copy(emis_sample)
        n_ab_mat_hist[i,:,:,:] = np.copy(n_ab_mat)

        for j in range(num_node):
            x,y = np.unique(z_ijl[:,:,j,0], return_counts=True)
            if x[0]==-1:
                x = np.delete(x, 0,0)
                y = np.delete(y, 0,0)
                
            if len(x):
                state_group_hist[i,j,x] = state_group_hist[i,j,x]+y
                
                
        for j in range(np.shape(ob)[2]):
            x,y = np.unique(z_ijl[:,j,:,1], return_counts=True)

            if x[0]==-1:
                x = np.delete(x, 0,0)
                y = np.delete(y, 0,0)
                
            if len(x):
                
                z_j_hist[i,x,j] = z_j_hist[i,x,j]+y
            else:
                print('wrong',i,j)
                print(np.sum(ob[:,:,j]))
            
            
            
            