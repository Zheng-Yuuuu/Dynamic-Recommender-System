import numpy as np
import numpy.random as npr
import function as ft
import time

def backward(tran_arr, tran_mat, time_span, tran_route_user_sample,
              ob, num_node, emis_prior, sample_index,
             emis_sample, n_ab_mat,z_ijl,score_num,state_prior, item_prior,
             route_index,user_num,rate_prior,rate_mat_x,rate_mat_y,z_i,burn_iter,
             state_group,item_group):
    
    cur_node_tran_sample = np.zeros((num_node,num_node),dtype=int)
    cur_emis_sample = np.zeros(num_node,dtype=int)
    
    tran = np.reshape(tran_arr, (num_node,num_node))
    emis_prob = npr.dirichlet(emis_sample+emis_prior)
    for user_id in range(user_num):
        
        
        x_time, y_item, score_arr,state_mat,item_mat = \
        sample_minus_cur_user(ob,n_ab_mat,z_ijl,user_id,num_node,rate_mat_x,rate_mat_y,state_group,item_group)
        
        rate_mat = n_ab_mat  + rate_prior
        s = np.sum(rate_mat,2)
        s = np.repeat(s[:,:,np.newaxis], np.shape(n_ab_mat)[2],2)
        rate_prob = rate_mat/s 
    
        score_prob = prob_state_item_user(rate_prob, score_arr, state_mat, item_mat,
                              state_prior, item_prior,score_num,num_node)
        time_state_prob = all_item_prob_for_time(score_prob,x_time,time_span,num_node)
        pred_prob_mat = np.zeros((num_node,time_span))
    

        s = np.ones(num_node)
        for i in range(time_span-1,-1,-1):
#             print(i,'gg')
#             print(np.shape(pred_prob_mat[:,i]))
#             print(np.shape(s))     
#             print(s,'s')       
            pred_prob_mat[:,i] = np.copy(s)
            s = s*time_state_prob[:,i]
            s = np.dot(tran,s)
#         print(pred_prob_mat,'pred_prob_mat')
        
        pval = emis_prob*time_state_prob[:,0]*pred_prob_mat[:,0]
#         print(np.shape(pval))
        norm_pval(pval)
#         print(pval,'emis')
        index = np.where(npr.multinomial(1,pval)==1)[0]
#         print(pval,'pval') 
#         print(time_state_prob,'time_state_prob[:,0]')
#         print(pred_prob_mat,'pred_prob_mat[:,0]')
#         print(index)
        route_index_arr = np.zeros(time_span,dtype=int)
#         print(index,'index')
        route_index_arr[0] = index
        

        for i in range(1,time_span):
            trans_prob = np.copy(tran[index,:])
            pval = trans_prob*time_state_prob[:,i]*pred_prob_mat[:,i]
            pval = np.reshape(pval,num_node)
            norm_pval(pval)
#             print(pval,'pval tran')
            index = np.where(npr.multinomial(1,pval)==1)[0]
            route_index_arr[i] = index
        

        chosen_arr = choose_route_division(tran_mat,route_index_arr,time_span,num_node)
        ft.record_tran_route_user_sample(chosen_arr, tran_route_user_sample, sample_index,
                                      user_id)
     
        ft.record_node_tran_sample(chosen_arr, cur_node_tran_sample, route_index, time_span,
                                    num_node)
        ft.record_emis_sample(route_index_arr[0],cur_emis_sample)
        
        sample_item_state(user_id, state_prior, item_prior,rate_prior,
                       score_arr,rate_mat_x,rate_mat_y,
                       n_ab_mat, state_mat, item_mat,z_ijl, y_item, x_time, route_index_arr,
                       time_span,z_i,burn_iter,sample_index,state_group,item_group)

#         if np.sum(check_change) != len(np.where(z_ijl[:,:,:,1]>-1)[0]):
#             print('wowowowowowowowowowow')
#         else:
#             print('rorororoororororroror')
#         for i in range(len(check_change)):
#             if check_change[i]!=len(np.where(z_ijl[:,i,:,1]>-1)[0]):
#                 print('wowowoowowowowoowowwwowowoowowow',i)
    return cur_emis_sample,cur_node_tran_sample
        
def sample_minus_cur_user(ob,n_ab_mat,z_ijl,user_id,num_node,rate_mat_x,rate_mat_y,state_group,item_group):
    
#     print('sample_minus_cur_user ')
    ob_user = np.copy(ob[:,user_id,:])     
    x_time, y_item = np.where(ob_user>-1)
    score_arr = ob_user[x_time,y_item]
    
#     print(x_time,'x_time')
#     print(y_item,'y_item')
#     print(score_arr,'score_arr')
    
    index_state = np.where(z_ijl[user_id,y_item,:,0]!=-1)[1] 
#     print(index_state,'index_state')
#     print(len(y_item),'len(y_item)')
#     print(len(index_state),'len(index_state)')

    '''
    row - item index col - state index 
    so choose col
    '''
    for i in range(len(y_item)):
        rate = score_arr[i]
        index_a = z_ijl[user_id,y_item[i],index_state[i],0]
        index_b = z_ijl[user_id,y_item[i],index_state[i],1]
#         print(rate,'rate')
#         print(index_a,'ind a')
#         print(index_b,'ind b')
        n_ab_mat[index_a ,index_b,rate]= \
        n_ab_mat[index_a ,index_b,rate] - 1
        state_group[index_state[i],index_a] = state_group[index_state[i],index_a] -1
        item_group[index_b,y_item[i]] = item_group[index_b,y_item[i]]-1
        
        
    z_ijl[user_id,y_item,:,:] = -1   

#     state_mat = np.zeros((num_node,rate_mat_x))
#     for i in range(num_node):
#         index, counts = np.unique(z_ijl[:,:,i,0], return_counts = True)
#         if index[0]==-1:
#             index = np.delete(index, 0, 0)
#             counts = np.delete(counts, 0, 0)
#         if len(index):
#             state_mat[i,index] = state_mat[i,index]+counts
#     print(time.time() - startTime1,'s2')
#     
#     state_group_check = np.copy(state_group)

    state_mat = np.copy(state_group)
#     item_mat = np.zeros((rate_mat_y,len(y_item)))
#     for i in range(len(y_item)):
#         index, counts = np.unique(z_ijl[:,y_item[i],:,1], return_counts = True)
#         if index[0]==-1:
#             index = np.delete(index, 0, 0)
#             counts = np.delete(counts, 0, 0)
#         if len(index):
#             item_mat[index,i] = item_mat[index,i]+np.transpose(counts)
#     print(time.time() - startTime1,'s3')
#     
#     item_group_check = np.copy(item_group[:,y_item])
    item_mat = np.copy(item_group[:,y_item])
#     if np.array_equal(state_group_check,state_mat):
#         pass
#     else:
#         print('caonima0')
#     if np.array_equal(item_group_check,item_mat):
#         pass
#     else:
#         print('caonima1')
#     print('sample_minus_cur_user ends')
#     print(len(y_item),'len(y_item)')
#     for hh in range(np.shape(item_mat)[1]):
#         hhh = np.where(item_mat[:,hh]>0)[0]
#         if len(hhh)==0:
#             print('wwwwwwwwwwwrong')
    return x_time, y_item, score_arr,state_mat,item_mat
    
    
def prob_state_item_user(rate_prob, score_arr, state_mat, item_mat, state_prior, item_prior,
                         score_num,num_node):
    
    score_prob = np.zeros((num_node,len(score_arr)))
    state_prob = (state_mat+state_prior)/cal_mat_row_prob(state_mat,state_prior)
    item_prob = (item_mat+item_prior)/(cal_mat_col_prob(item_mat,item_prior))
    for i in range(score_num):
        s = np.dot(np.dot(state_prob,rate_prob[:,:,i]),item_prob)
        index = np.where(score_arr==i)[0]
        if len(index):
            score_prob[:,index] = np.copy(s[:,index])
            '''
            all *10 to increase not to be 0 
            '''
    
#     print(np.where(score_prob==0),'np.where(score_prob==0)')
    return score_prob
        
        
    
def cal_mat_row_prob(a,a_prior):
    mat = a+a_prior
    s = np.sum(mat,1)
    s = np.repeat(s[:,np.newaxis], np.shape(mat)[1], 1)
    return s 
    
def cal_mat_col_prob(a,a_prior):
    mat = a+a_prior
    s = np.sum(mat,0)
    s = np.repeat(s[np.newaxis,:], np.shape(mat)[0], 0)
    return s 

def all_item_prob_for_time(score_prob,x_time,time_span,num_node):
    '''
    initial value set 1 because if no item for time_span 
    '''
    time_state_prob = np.ones((num_node,time_span))
    for i in range(time_span):
        index = np.where(x_time==i)[0]
#         print(len(index),'index',i,'i')
        if len(index):
            s = np.copy(score_prob[:,index])
#             hh,hhh = np.where(s==0)
#             print(hh,'hh',hhh,'hhh')
            s = np.log(s)
            s = np.sum(s,1)
            s_log_max = np.amax(s)
            s = s+np.abs(s_log_max)
            s = np.exp(s)
            time_state_prob[:,i] = np.copy(s) 
            
    return time_state_prob
    
def choose_route_division(tran_mat,index_route_arr,time_span,num_node):
    arr = np.zeros((time_span-1,2),dtype=int)
    for i in range(time_span-1):
        s = index_route_arr[i]*num_node+index_route_arr[i+1]
        pval = np.copy(tran_mat[s,:])
#         print(int(s/num_node),s%num_node,'int(s/num_node),s%num_node choose_route_division')
#         print(pval,'pval ss')
        norm_pval(pval)
#         print(pval,'pval')
        index_pval = npr.multinomial(1,pval)
        index = np.where(index_pval==1)[0]
        arr[i,0] = s 
        arr[i,1] = index
        
    return arr
    
def sample_item_state(user_id, state_prior, item_prior,rate_prior,
                       score_arr,rate_mat_x,rate_mat_y,
                       n_ab_mat, state_mat, item_mat,z_ijl, y_item, x_time, state_index_arr,
                       time_span,z_i,burn_iter,sample_index,state_group,item_group):
#     print('sample_item_state')
    for i in range(time_span):
        time_index = np.where(x_time==i)[0]
        if len(time_index):
            state = np.copy(state_mat[state_index_arr[i],:])
            state = state+state_prior
#             state = state/np.sum(state)
            state = np.repeat(state[:,np.newaxis], rate_mat_y, 1)
#             print(time_index,'time_index')
            for j in range(len(time_index)):
                item = np.copy(item_mat[:,time_index[j]])
                item = item+item_prior
#                 item = item/np.sum(item)
                item = np.repeat(item[np.newaxis,:],rate_mat_x,0)
                
                rate_mat = n_ab_mat  + rate_prior
                s = np.sum(rate_mat,2)
                s = np.repeat(s[:,:,np.newaxis], np.shape(n_ab_mat)[2],2)
                rate_prob = rate_mat/s 
                rate_prob = rate_prob[:,:,score_arr[time_index[j]]]
                
                prob_mat = state*rate_prob*item
                prob = np.reshape(prob_mat, (rate_mat_x*rate_mat_y))
                norm_pval(prob)
                pval_index = npr.multinomial(1,prob)
                index = np.where(pval_index==1)[0]
                x_index = int(index/rate_mat_y)
                y_index = index%rate_mat_y
#                 print(y_item[time_index[j]],'y_item[time_index[j]]')
                z_ijl[user_id,y_item[time_index[j]],state_index_arr[i],0] = x_index
                z_ijl[user_id,y_item[time_index[j]],state_index_arr[i],1] = y_index
                
                state_mat[state_index_arr[i],x_index] = state_mat[state_index_arr[i],x_index]+1
                state_group[state_index_arr[i],x_index] = state_group[state_index_arr[i],x_index]+1
                item_group[y_index,y_item[time_index[j]]] = item_group[y_index,y_item[time_index[j]]]+1
                
                n_ab_mat[x_index,y_index,score_arr[time_index[j]]] = \
                n_ab_mat[x_index,y_index,score_arr[time_index[j]]]+1
                if sample_index>=burn_iter:
                    z_i[i,user_id,x_index] = z_i[i,user_id,x_index]+1
                
#     print('sample_item_state end')

def norm_pval(pval):
    pval[:] = pval[:]/np.sum(pval)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    