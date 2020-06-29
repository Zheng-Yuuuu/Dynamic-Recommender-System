import numpy as np


def pd(z_i,z_j_hist,n_ab_mat_hist,ob,rate_prior,ob_real,burn_iter):
    '''
    in ob -2 reprenst to be predicted
    '''
    ob_pred = np.copy(ob)
    ob_pred = ob_pred.astype(float)
    
    
    n_ab_mat = np.sum(n_ab_mat_hist[burn_iter:-1,:,:,:],0)
    score_num = np.shape(n_ab_mat)[2]
    
    rate = rate_prior+n_ab_mat
    s = np.sum(rate,2)
    s = np.repeat(s[:,:,np.newaxis], np.shape(rate)[2],2 )
    rate = rate/s 
    
    s = np.arange(score_num)
    s = np.repeat(s[np.newaxis,:], np.shape(rate)[0], 0)
    s = np.repeat(s[:,np.newaxis,:],np.shape(rate)[1],1)
    
    rate = rate*s 
    rate = np.sum(rate,2)
    
    s = np.sum(z_i,2)
    x,y = np.where(s==0)
    s[x,y] = 1
    s = np.repeat(s[:,:,np.newaxis], np.shape(z_i)[2], 2)
    z_i_prob = z_i/s  
    
    
    
    z_j_hist = np.sum(z_j_hist[burn_iter:-1,:,:],0)
    s = np.sum(z_j_hist,0)
#     b = 0
#     for i in range(len(s)):
#         if s[i]==0:
#             b = b+1
#             print(i)
#             print('wrong')
#     print(b)
    s = np.repeat(s[np.newaxis,:], np.shape(z_j_hist)[0], 0)
    z_j_prob = z_j_hist/s
    
    
    time, user, item = np.where(ob==-2)
    print(len(time),'geshu')
    
    x,y,z = np.where(ob_real==-2)
    print(len(x),'ob real')

    
    for i in range(len(time)):
        ob_pred[time[i],user[i],item[i]] = np.dot(np.dot(z_i_prob[time[i],user[i],:],rate),
                                        z_j_prob[:,item[i]])
        
    s = (ob_pred[time,user,item]-ob_real[time,user,item])*\
    (ob_pred[time,user,item]-ob_real[time,user,item])
    
    s = np.sqrt(np.sum(s)/len(time))
    print(s)
    
    pred_arr = np.zeros(np.shape(ob)[0])
    for t in range(np.shape(ob)[0]):
        user,item = np.where(ob[t,:,:]==-2)
        if len(user):
            s = (ob_pred[t,user,item]-ob_real[t,user,item])*\
            (ob_pred[t,user,item]-ob_real[t,user,item])
            pred_arr[t] = np.sqrt(np.sum(s)/len(user))
    
    pred_arr.dump('train_pred.dat')
        
        
    print(pred_arr)
