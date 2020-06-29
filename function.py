import numpy as np
import numpy.random as npr

'''
sample-index: ?th sample 
node to node probability
'''
def node_prob(tran_prior,node_tran_sample,num_node ):
    node_prob_mat = tran_prior+node_tran_sample
    for i in range(num_node):
        node_prob_mat[i,:] = npr.dirichlet(node_prob_mat[i,:])
    node_prob_arr = np.reshape(node_prob_mat,num_node*num_node)
#     print(node_prob_arr,'node_prob_arr')
#     print(np.sum(node_prob_arr[0:9]),'np.sum(node_prob_arr[0:9])')

    return node_prob_arr
    
def tran_prob(tran_prior,node_tran_sample,num_node,route_index,one_mat):
    node_prob_arr = node_prob(tran_prior,node_tran_sample,num_node )
    node_prob_mat = np.repeat(node_prob_arr[np.newaxis,:], np.shape(route_index)[1], 0)
    node_prob_mat = np.repeat(node_prob_mat[np.newaxis,:,:], np.shape(route_index)[0], 0)
    node_prob_mat = node_prob_mat*route_index+one_mat
    prob_mat = np.prod(node_prob_mat, 2)
    
    prob_arr = np.sum(prob_mat, 1)
#     print(prob_mat[8,:],'prob_mat[8,:]')
#     print(prob_arr,'prob_arr')
#     print(np.sum(prob_arr[0:9]),'np.sum(prob_arr[0:9])')
#     print(prob_mat,'prob_mat')
    return prob_mat,prob_arr   
        
        
def map_one_to_two(num_node):
    s = num_node*num_node
    arr = np.zeros((s,2),dtype=int)
    for i in range(num_node):
        for j in range(num_node):
            arr[i*num_node+j,0]=i
            arr[i*num_node+j,1]=j
            
    return arr
    
    
def record_node_tran_sample(arr,node_tran_sample,route_index,time_span,num_node):
    arr = arr.astype(int)
    for i in range(time_span-1):
        s = np.copy(route_index[arr[i,0],arr[i,1]])
        s = np.reshape(s, (num_node,num_node))
        node_tran_sample = node_tran_sample+s

def record_tran_route_user_sample(index_arr, tran_route_user_sample, sample_index, user_id):
#     print(index_arr,'index_arr')
    tran_route_user_sample[sample_index,user_id,:,:] = np.copy(index_arr)
#     print(tran_route_user_sample[sample_index,user_id,:,:])

def record_emis_sample(s,cur_emis_sample):

#     chosen_arr = chosen_arr.astype(int)
    cur_emis_sample[s] = cur_emis_sample[s]+1
    
    
    
    