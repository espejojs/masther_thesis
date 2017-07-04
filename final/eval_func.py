import numpy as np
from multiprocess import Pool,cpu_count
from cvxpy import *
import dill
import time
import sys

results=dill.load( open( sys.argv[1], "rb" ))

for res in results.keys():
    exec (res + "= results[res]")
    
for var in variables.keys():
    exec (var + "= variables[var]")
for cons in constants.keys():
    exec (cons + "= constants[cons]")
    
p=Variable(p.size[0],p.size[1])
w=Variable(w.size[0],w.size[1])
s=Variable(s.size[0],s.size[1])
load=Variable(load.size[0],load.size[1])
on_off_param=Parameter(num_p,n_h)
r_parameter=Parameter(s.size[0],s.size[1])


if PWL:
    gk=Variable(gk.size[0],gk.size[1])

on_off_value=np.zeros((num_p,n_h))
on_off_init=np.zeros(num_p)

r=np.mean(scenarios_bus,2)

def eval_obj (on_off,on_off_cost=on_off_cost,n_h=24):
    obj_eval=0
    for i in range(1,n_h):
        obj_eval = obj_eval + np.sum(on_off_cost*np.maximum(on_off[:,i] - on_off[:,i-1],0))
    obj_eval = obj_eval + np.sum((on_off_cost)*np.maximum(on_off[:,0].T - on_off_init,0).T)
        
    return obj_eval


PWL=False

# init constraints second stage

Constraints=[]
for i in range (n_h):
    Constraints.append(p[:,i]<=mul_elemwise(p_max[:],on_off_param[:,i]))
for i in range (n_h):
    Constraints.append(p[:,i]>=mul_elemwise(p_min[:],on_off_param[:,i]))
for i in range (n_h):
    Constraints.append(s[:,i]<=r_parameter[:,i])

for i in range (n_h):
    if PWL:
        for j in range (pwl_points):
            Constraints.append(gk[num_p*j:num_p*(j+1),i]<=p_max/(pwl_points))
            Constraints.append(gk[num_p*j:num_p*(j+1),i]>=0)
    
        Constraints.append( p[:,i]== np.sum ([gk[num_p*j:num_p*(j+1),i] for j in range(pwl_points)]) )
       
    Constraints.append(G*p[:,i]+R*s[:,i]-A*w[:,i]==Dm*load[:,i])
    Constraints.append(w[:,i]<=w_max)
    Constraints.append(w[:,i]>=w_min)
    Constraints.append(J*w[:,i]<=z_max)
    Constraints.append(J*w[:,i]>=z_min)
    #Changed here
    Constraints.append(s[:,i]>=0)
    Constraints.append(load[:,i]>=0)
    Constraints.append(load[:,i]<=b[:,i])
for i in range(1,n_h):
    Constraints.append(p[:,i]-p[:,i-1]<=rampup[:])
    Constraints.append(p[:,i]-p[:,i-1]>=-rampdown[:])   
    

# init objective secon stage

Objective=0
for i in range (n_h):
    if PWL:
        for j in range(pwl_points):
            Objective=Objective + sum_entries(mul_elemwise(pwl_cost[:,j],gk[num_p*j:num_p*(j+1),i]))
#             for k in range(num_p):
#                 Objective=Objective + gk[k+(num_p*j),i]*pwl_cost[k,j]
    else:
        Objective=Objective + quad_form(p[:,i],H0*0.5)
        Objective=Objective + g0*p[:,i]# check 
        
    Objective=Objective + Load_Penalty*norm(load[:,i]-b[:,i],1)
# Objective=Objective/10
prob = []
prob = Problem(Minimize(Objective), Constraints)


def eval_on_off(on_off_value,r,j,k,duals=False,n_h=n_h,num_p=num_p):     
    on_off_param.value=on_off_value[k]
    r_parameter.value=r[:,:,j]
    Q=0
    try:
        Q=prob.solve(solver=ECOS,reltol=1e-8,verbose=False,)
    except:
        Q=np.nan
        print k,j,Q
        
    if not duals:
        return Q,p.value,load.value,s.value
    
Q_dict={}
p_dict={}
load_dict={}
s_dict={}
for k in range(len(k_save)):
    print "start",
    st=time.time()
    pool=Pool(int(sys.argv[2]),initargs=(k,))
    ocost=eval_obj(on_off=on_off_array[k])#results['on_off_array'][0]
    st=time.time()
    res = list(zip(*pool.map(lambda i: eval_on_off(on_off_value=on_off_array,
                                                   r=scenarios_bus[:,:,n_samples:n_samples+n_test],j=i,k=k),
                                                   range(n_test),
                                                    chunksize=10)))
    #Q_nan=list(res[0])
    #Q_nan=Q_nan[~numpy.isnan(Q_nan)]
    Q_dict[str(k)]=list(res[0])+ocost
    p_dict[str(k)]=np.nanmean(res[1],0) # changed here for converged ieee14  day 1
    load_dict[str(k)]=np.nanmean(res[2],0)
    s_dict[str(k)]=np.nanmean(res[3],0)
    pool.close()
    pool.join()
    print time.time()-st
    print np.nanmean(Q_dict[str(k)])
                        
    
results={}
results['Q_dict']=Q_dict
results['p_dict']=p_dict
results['load_dict']=load_dict
results['s_dict']=s_dict
results['time_eval']=time_eval
results['iter_time_total']=iter_time_total
results['k_final']=k_final
results['k_save']=k_save
dill.dump( results, open( "eval/"+sys.argv[1].rsplit('/')[-3]+"/"+sys.argv[1].rsplit('/')[-2]+"/"+sys.argv[1].rsplit('/')[-1], "wb" ) )      
