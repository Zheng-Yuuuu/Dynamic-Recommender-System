import numpy as np
import numpy.random as npr
import function as ft

def init(user_num, route_num_node_to_node, time_span,tran_route_user_sample,
         route_index,num_node, node_tran_sample, emis_sample,rate_mat_x,
         rate_mat_y,z_ijl,ob,n_ab_mat,tran_prior,eta,z_j_hist,state_group_hist,z_i):
    
    pval = np.ones((num_node))/num_node
    arr = npr.multinomial(user_num,pval) #choose route as departure and terminal
    
    shuffle(arr, time_span, route_index, tran_route_user_sample,
            route_num_node_to_node, num_node, node_tran_sample, emis_sample, user_num,
            rate_mat_x,rate_mat_y,z_ijl,ob,n_ab_mat,tran_prior,eta,
            z_j_hist,state_group_hist,z_i)
    

def shuffle(arr, time_span, route_index, tran_route_user_sample,
            route_num_node_to_node, num_node, node_tran_sample, emis_sample, user_num,
            rate_mat_x,rate_mat_y,z_ijl,ob,n_ab_mat,tran_prior,eta,
            z_j_hist,state_group_hist,z_i):
    
    emis_sample[:] = emis_sample[:]+arr  #record all initial departure
    
    arr = append_mult(arr, np.where(arr!=0)[0] )
    npr.shuffle(arr)
    arr = arr.astype(int)
    
#     print(arr,'arr')
    index_arr = np.ones(time_span,dtype=int)
    for i in range(user_num):
        
        index_arr[0] = arr[i]
        
        for j in range(1,time_span):
            
            arr_cur = np.where(tran_prior[index_arr[j-1],:]==eta)[0]
            next_index = npr.randint(len(arr_cur))
            index_arr[j] = arr_cur[next_index]
            
        #route choosen

        ob_user = np.copy(ob[:,i,:])
        for m in range(time_span):
            item_index =  np.where(ob_user[m,:]>-1)[0]
            rand_zi = npr.randint(rate_mat_x, size = len(item_index))
            rand_zj = npr.randint(rate_mat_y, size = len(item_index))
            for n in range(len(item_index)):
                n_ab_mat[rand_zi[n],rand_zj[n],ob_user[m,item_index[n]]] =\
                n_ab_mat[rand_zi[n],rand_zj[n],ob_user[m,item_index[n]]]+1
                z_ijl[i,item_index[n],index_arr[m],0] =  rand_zi[n]
                z_ijl[i,item_index[n],index_arr[m],1] =  rand_zj[n]
                z_j_hist[0,rand_zj[n],item_index[n]] =\
                z_j_hist[0,rand_zj[n],item_index[n]]+1
                state_group_hist[0,index_arr[m],rand_zi[n]]=\
                state_group_hist[0,index_arr[m],rand_zi[n]]+1
                z_i[m,i,rand_zi[n]] = z_i[m,i,rand_zi[n]]+1
        # n_ab_mat,z_ijl done
        
        route_division_arr = np.zeros((time_span-1,2),dtype=int)
        for j in range(time_span-1):
            index_route = index_arr[j]*num_node+index_arr[j+1]
            num_division = route_num_node_to_node[index_route]
            route_division = npr.randint(num_division,size = 1)
            route_division_arr[j,0] = index_route
            route_division_arr[j,1] = route_division
            
        ft.record_tran_route_user_sample(route_division_arr, tran_route_user_sample, 0, i)
    
        for j in range(time_span-1):
            s = np.copy(route_index[route_division_arr[j,0],route_division_arr[j,1],:])
            s = np.reshape(s,(num_node,num_node))
            node_tran_sample[:,:] = node_tran_sample[:,:]+s  

def append_mult(arr, index_not_zero):
    shuffle_arr = []
    shuffle_arr = np.asarray(shuffle_arr)
    for i in range(len(index_not_zero)):
        s = np.ones(arr[index_not_zero[i]])*index_not_zero[i]
        shuffle_arr = np.append(shuffle_arr, s)
    
    return shuffle_arr
    
    
    
    