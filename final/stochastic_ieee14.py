import numpy as np
import pfnet as pf
from scipy.sparse import triu,bmat,coo_matrix,eye,spdiags
import matplotlib.pyplot as plt
from scikits.sparse.cholmod import cholesky
import time
import dill
import sys
from multiprocess import Pool,cpu_count

#%matplotlib inline  

net=pf.Network()
#net.load('PFNET-multi_period/data/ieee14.mat')
#net.load('PFNET-multi_period/data/ieee300.mat')
#net.load('data/ieee300.mat')
net.load('data/ieee14.mat')

penetration=50.
net.add_vargens(net.get_gen_buses(),#[net.get_bus(1)], 
                penetration, 50., 10, 0.1)

# Problem set up

total_load = sum([l.P for l in net.loads])
total_capacity=sum([gen.P_max for gen in net.generators])
uncertainty = 100.*sum([g.P_std for g in net.var_generators])/sum([g.P_max for g in net.var_generators])
corr_value = net.vargen_corr_value 
corr_radius = net.vargen_corr_radius

for br in net.branches:
    if br.ratingA==0:
        br.ratingA = 15

for gen in net.generators:
    gen.P_min = 0.
    gen.P_max = np.maximum(gen.P_max,0.)
    assert(gen.P_min <= gen.P_max)


num_w = net.num_buses-net.get_num_slack_buses() # voltage angles
num_p = net.get_num_P_adjust_gens()             #net.get_num_generators()         # adjustable generators
#num_r = net.num_var_generators                       # renewable generators
num_r = net.num_vargens
num_bus = net.num_buses                         # buses
num_br = net.num_branches                       # branches
num_l = net.num_loads

capacity_per=np.ones(5)/5
# capacity_per=np.array([0.23494860499265785, 0.00279001468428781, 0.48208516886930984, 0.13906020558002939, 0.141116005873715125])
#capacity_per=np.array([0.23494860499265785, 0.30279001468428781, 0.18208516886930984, 0.23906020558002939, 0.041116005873715125])

#([0.25, 0.50, 0.15, 0.05, 0.05 ])#
capacity_per_cumsum=np.cumsum(capacity_per)
sorted_gen=np.array(sorted([gen.P_max  for gen in net.generators]))
gen_cumsum=np.cumsum(sorted_gen)
gen_bounds=np.zeros(len(capacity_per)+1)
gen_bounds[0]=100
for idx,cap in enumerate(capacity_per_cumsum):
    gen_bounds[idx+1]=np.min(sorted_gen[gen_cumsum/total_capacity>=(1-cap)])
gen_bounds[-1]=0
parameters=[('Nuclear', 0.029917833750000022, 3.0741044999999803, 100, 400.0, 280.0, 280.0, 24.0, 168.0, 40000.0),
('Coal', 0.11954207142857155, 12.270104999999957, 140, 350.0, 140.0, 140.0, 5.0, 8.0, 12064.0),
('IGCC', 0.25331129032258076, 10.655729999999988, 54, 155.0, 70.0, 80.0, 16.0, 24.0, 2058.0),
('CCGT', 0.14437728426395943, 7.7242649999999946, 104, 197.0, 310.0, 310.0, 3.0, 4.0, 230.0),
('OCGT', 2.2680749999999974, 13.795700000000027, 8, 20.0, 90.0, 100.0, 1.0, 2.0, 46.0)]


min_down=np.zeros([num_p])
min_up=np.zeros([num_p])
ramp_up=np.zeros([num_p])
ramp_down=np.zeros([num_p])
on_off_cost=np.zeros([num_p])
beta=1.0

gen_tech=np.chararray([num_p],itemsize=10)
for i in reversed(range(0,len(gen_bounds)-1)): 
    for gen in net.generators:
        if gen.P_max >= gen_bounds[i+1] and gen.P_max < gen_bounds[i]:
            aggregator= (np.maximum(gen.P_max/(parameters[i][4]/net.base_power),0))
#             print aggregator,parameters[i][0],gen.index
            gen.P_max= aggregator*parameters[i][4]/net.base_power
            gen.P_min= aggregator*parameters[i][3]/net.base_power
            gen.cost_coeff_Q2= net.base_power**2*parameters[i][1]/aggregator
            gen.cost_coeff_Q1= net.base_power*parameters[i][2]/aggregator
            #gen.dP_max= parameters[i][5]/net.base_power*aggregator
            
            ramp_up[gen.index]=(aggregator*parameters[i][5])/net.base_power*beta
            ramp_down[gen.index]=(aggregator*parameters[i][6])/net.base_power*beta
            min_down[gen.index]= parameters[i][7]
            min_up[gen.index] = parameters[i][8]
            on_off_cost[gen.index] =parameters[i][9]*aggregator
            
            gen_tech[gen.index] = parameters[i][0]
            
name_techno=list(set([techno[0] for techno in parameters]))

for name in name_techno:
    print (name,np.sum([ gen.P_max for idx,gen in enumerate( net.generators) if gen_tech[idx]==name ])/total_capacity)


net.clear_flags()
net.set_flags(pf.OBJ_BUS,
              pf.FLAG_VARS,
              pf.BUS_PROP_NOT_SLACK,
              pf.BUS_VAR_VANG)
net.set_flags(pf.OBJ_GEN,
              pf.FLAG_VARS,
              pf.GEN_PROP_P_ADJUST,
              pf.GEN_VAR_P)
net.set_flags(pf.OBJ_VARGEN,
              pf.FLAG_VARS,
              pf.VARGEN_PROP_ANY,
              pf.VARGEN_VAR_P)
net.set_flags(pf.OBJ_LOAD,
              pf.FLAG_VARS,
              pf.LOAD_PROP_ANY,
              pf.LOAD_VAR_P)

x = net.get_var_values()
Pw = net.get_var_projection(pf.OBJ_BUS,pf.BUS_VAR_VANG)
Pp = net.get_var_projection(pf.OBJ_GEN,pf.GEN_VAR_P)
Pr = net.get_var_projection(pf.OBJ_VARGEN,pf.VARGEN_VAR_P)
Pd = net.get_var_projection(pf.OBJ_LOAD,pf.LOAD_VAR_P)


pf_eq = pf.Constraint(pf.CONSTR_TYPE_DCPF,net)
pf_eq.analyze()
pf_eq.eval(x)
AA = pf_eq.A.copy()
bb = pf_eq.b.copy()

fl_lim = pf.Constraint(pf.CONSTR_TYPE_DC_FLOW_LIM,net)
fl_lim.analyze()
fl_lim.eval(x)
GG = fl_lim.G.copy()
hl = fl_lim.l.copy()
hu = fl_lim.u.copy()

cost = pf.Function(pf.FUNC_TYPE_GEN_COST,1.,net)
cost.analyze()
cost.eval(x)
H = (cost.Hphi + cost.Hphi.T - triu(cost.Hphi)) # symmetric
g = cost.gphi - H*x
l = net.get_var_values(pf.LOWER_LIMITS)
u = net.get_var_values(pf.UPPER_LIMITS)

p_max = Pp*u
p_min = Pp*l
w_max = 5*np.ones(num_w)
w_min = -5*np.ones(num_w)
r_max = Pr*u
r_base = Pr*x
z_max = hu
z_min = hl 
H0 = Pp*H*Pp.T # change costs
g0 = Pp*g
#H1 = self.H0*self.parameters['cost_factor']
g1 = np.zeros(num_p)
G = AA*Pp.T
R = AA*Pr.T
A = -AA*Pw.T
Dm = -AA*Pd.T
J = GG*Pw.T 
b = bb
ll= Pd*x #np.array([l.P for l in net.loads]) # check 


rr_cov = Pr*net.create_vargen_P_sigma(corr_radius,corr_value)*Pr.T
r_cov = (rr_cov+rr_cov.T-triu(rr_cov)).tocsc()
factor = cholesky(r_cov)
LL,D = factor.L_D()
P = factor.P()
PT = coo_matrix((np.ones(P.size),(P,np.arange(P.size))),shape=D.shape)
D = D.tocoo()
Dh = coo_matrix((np.sqrt(D.data),(D.row,D.col)),shape=D.shape)
L = PT*LL*Dh

from cvxpy import *
#import mosek
#import gurobi

PWL=True

# PWL approximation
pwl_cost=[]
if PWL:
    pwl_points=int(str(sys.argv[6]))
    pwl_cost=np.zeros([num_p,pwl_points])
    for gen_id in range(num_p):
        xx=np.linspace(p_min[gen_id]*0,p_max[gen_id],pwl_points+1)
        y=np.zeros(pwl_points+1)
        for i in range(0,pwl_points+1):
            y[i]=0.5*H0.diagonal()[gen_id]*xx[i]**2+g0[gen_id]*xx[i]
            pwl_cost[gen_id][i-1]=((y[i]-y[i-1])/(xx[i]-xx[i-1]))
#         plt.plot(xx,y)

    for gen_id in range(num_p):
        for i in range(1,pwl_points):
            assert pwl_cost[gen_id][i]>pwl_cost[gen_id][i-1]
            
# plt.show()
norm_factor=24*num_p
H0=H0/(norm_factor)
g0=g0/(norm_factor)
pwl_cost=pwl_cost/norm_factor
on_off_cost=on_off_cost/norm_factor

load_data=np.genfromtxt('BPA_data/day'+str(sys.argv[4])+'.csv',delimiter=',',names=True)
r_test=load_data['wind']*total_load*(penetration/100)/num_r#np.flipud(
if int(sys.argv[4])==6:
    r_test=np.flipud(load_data['wind']*total_load*(penetration/100)/num_r)   

#np.flipud(

np.random.seed(seed=100)
n_h=24
p=Variable(num_p,n_h)
w=Variable(num_w,n_h)
z=Variable(num_br,n_h)
s=Variable(num_r,n_h)
load=Variable(num_l,n_h)
on_off=Bool(num_p,n_h)
on_off_c=Variable(num_p,n_h)
on_off_param=Parameter(num_p,n_h)
theta=Variable(1)
r_parameter=Parameter(num_r,n_h)
g_k2=Parameter(2*n_h,num_p)

Pforecast=total_load
Load_Penalty=np.max(pwl_cost)*int(sys.argv[2])#1e0
# p_min = p_max*0.0
rampup = ramp_up #p_max/(H0.diagonal()/np.max(H0))*0.04 # p_max*0.01
rampdown=ramp_down #p_max/(H0.diagonal()/np.max(H0))*0.04 # p_max*0.01

if PWL:
    gk=Variable(num_p*pwl_points,n_h)


ontime=np.zeros(num_p)#-25
on_off_init=np.zeros(num_p)

np.random.seed(seed=100)
mult=int(sys.argv[5])/10.
n_samples=1000
n_test=1000
scenarios=np.zeros([n_samples+n_test,n_h])
scenarios_bus=np.zeros([num_r,n_h,n_samples+n_test])
b=[]
b=np.zeros([num_l,n_h])
for i in range (n_h):
    b[:,i]=mult*np.abs(ll)*load_data['load'][i]#/load_data['load'][0]

for j in range(n_samples+n_test):
    r=[]
    r=np.zeros([num_r,n_h])
    for i in range (n_h):
        r[:,i]=mult*np.minimum(np.maximum(r_test[i]*np.ones(num_r)# +0.1 ieee14
                                     +L*np.random.randn(num_r)*np.sqrt(r_test[i])*np.sqrt((i*1.+1)/(n_h)),1e-3),r_max)
    scenarios_bus[:,:,j]=r
    scenarios[j,:]=sum(r)
#     plt.plot(sum(r)/total_load,color='gray')

# plt.plot(np.mean(scenarios,0)/total_load,color='r',linewidth=3.0)
# plt.plot(mult*r_test*num_r/total_load,color='k',linewidth=3.0)
# plt.show()

r=np.mean(scenarios_bus,2)*0
rr=np.mean(scenarios_bus,2)

r=rr
u_bound=0
#if sys.argv[1]=="True":
CONT=False
#else:
#   CONT=False

print isinstance(CONT,bool)

def eval_obj (on_off,on_off_cost=on_off_cost,n_h=24):
    obj_eval=0
    for i in range(1,n_h):
        obj_eval = obj_eval + np.sum(on_off_cost*np.maximum(on_off[:,i] - on_off[:,i-1],0))
    obj_eval = obj_eval + np.sum((on_off_cost)*np.maximum(on_off[:,0].T - on_off_init,0).T)
      
    return obj_eval


def eval_on_off(on_off_value,r,j,prob,Constraints,duals=False,solver=ECOS):
    dual_multiplier_pos=np.matrix(np.zeros((n_h,num_p)))
    dual_multiplier_neg=np.matrix(np.zeros((n_h,num_p)))
    Q=0
    on_off_param.value=on_off_value
    if j ==-1:
        r_parameter.value=r
    else:
        r_parameter.value=r[:,:,j]
    #print on_off_param.value
    if solver==GUROBI:
        try:    
            Q=prob.solve(solver=GUROBI,BarQCPConvTol=1e-7,verbose=False,Method=4,warm_start=True)
        except:
            print "Gurobi Failed"
            Q=prob.solve(solver=ECOS,reltol=1e-8,max_iters=1000,warm_start=True)
    else:
        try:
            Q=prob.solve(solver=ECOS,reltol=1e-8,max_iters=1000,warm_start=True,verbose=False)#warm_start=True
        except:
            print "Failed ECOS"
            Q=np.nan
            dual_multiplier_neg[:]=np.nan
            dual_multiplier_pos[:]=np.nan
        

    if not np.isnan(Q):
        for i in range(n_h):
            dual_multiplier_pos[i,: ]= np.squeeze( np.asarray((np.diag(p_max)*Constraints[i].dual_value).T))
        for i in range(n_h):
            dual_multiplier_neg[i,: ]= np.squeeze( np.asarray((np.diag(p_min)*Constraints[i+n_h].dual_value).T))
        
    if not duals:
        return Q,p.value,load.value,s.value
    else:
        return Q, dual_multiplier_pos, dual_multiplier_neg
    
def second_stage(on_off_value,PWL,slope=False):
    Constraints=[]
    for i in range (n_h):
        Constraints.append(p[:,i]<=mul_elemwise(p_max[:],on_off_value[:,i]))
    for i in range (n_h):
        Constraints.append(p[:,i]>=mul_elemwise(p_min[:],on_off_value[:,i]))
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
                for k in range(num_p):
                    Objective=Objective + gk[k+(num_p*j),i]*pwl_cost[k,j]
        else:
            Objective=Objective + quad_form(p[:,i],H0*0.5)
            Objective=Objective + g0*p[:,i]# check 

        Objective=Objective + Load_Penalty*norm(load[:,i]-b[:,i],1)
    if slope:
        Objective=Objective+slope_update
    
    return Objective , Constraints

def first_stage(on_off,PWL,cont=True,benders=False):
    Objective_det=0
    Constraints_det=[]
    if not benders:
        Objective_det, Constraints_det = second_stage(on_off_value=on_off,PWL=PWL )
    if cont:
        
        Constraints_det.append(on_off>=0)
        Constraints_det.append(on_off<=1)
        
    else:
    
        for j in range (num_p):
            #if on_off_init[j]==1:
            indi= on_off[j,0]-on_off_init[j]
            rang= range(0,min(n_h,int(min_up[j]-ontime[j])))
            if rang != []:
                Constraints_det.append(on_off[j,rang]>=indi)
            #else:
            #    indi= 1
            #    rang= range(0,min(n_h,int(min_down[j]+ontime[j]-1)))
            #    if rang != []:
            #        Constraints_det.append(on_off[j,rang]<=1-indi)

        for i in range(1,n_h):
            for j in range (num_p):
                if i != 0:
                    indi=(on_off[j,i] - on_off[j,i-1])
                else:
                    indi=on_off[j,i]
                rang= range(i,min(n_h,int(i+min_up[j])))
                Constraints_det.append(on_off[j,rang]>=indi)


        for i in range(1,n_h):
            for j in range (num_p):
                indi=(on_off[j,i-1]-on_off[j,i] )
                rang= range(i,min(n_h,int(i+min_down[j])))
                if rang != []:
                    Constraints_det.append(on_off[j,rang]<=1-indi)


    for i in range(1,n_h):
        Objective_det=Objective_det + sum(mul_elemwise(on_off_cost[:],max_elemwise(on_off[:,i] - on_off[:,i-1],0)))

    Objective_det=Objective_det + sum(mul_elemwise(on_off_cost[:],max_elemwise(on_off[:,0] - on_off_init,0)))
#    if benders:
#        Objective_det=Objective_det+theta
    
    if cont:
        for i in range(0,n_h):
            Objective_det=Objective_det + u_bound*norm(on_off[:,i])

    slope_update=0
    slope_update=sum([-(g_k2[i,:])*(on_off[:,i]) +(g_k2[n_h+i,:])*(on_off[:,i]) for i in range(n_h) ])
    prob_det = []
    if not benders:
        prob_det = Problem(Minimize(Objective_det+slope_update), Constraints_det)
        
    return prob_det , Objective_det , Constraints_det



on_off_value=np.matrix(np.zeros((num_p,n_h)))

if not CONT:
    Objective_approx=0
    Constraints_approx=[]
    Objective_approx, Constraints_approx=second_stage(on_off_value=on_off_param,PWL=False ) #changed here
    prob_approx=[]
    prob_approx=Problem(Minimize(Objective_approx),Constraints_approx)

Objective=0
Constraints=[]
Objective,Constraints=second_stage(on_off_value=on_off_param,PWL=False )
prob=[]
prob=Problem(Minimize(Objective),Constraints)


if sys.argv[1]=="b":
    benders=True
elif sys.argv[1]=="s":
    benders=False

Objective_det=0
Constraints_det=[]
prob_det = []
Objective_bin=0
Constraints_bin=[]
prob_bin = []
prob_bin, Objective_bin , Constraints_bin = first_stage(on_off=on_off,PWL=True,cont=False)
if CONT:
    prob_det, Objective_det , Constraints_det = first_stage(on_off=on_off_c,PWL=False,cont=CONT)
    prob_bin, Objective_bin , Constraints_bin = first_stage(on_off=on_off,PWL=True,cont=False)
else:
    prob_det, Objective_det , Constraints_det = first_stage(on_off=on_off,PWL=True,cont=CONT,benders=benders)


sim_step=1000
if benders:
    sim_step=400
benders_sim=300
cost=[]
lower=[]
on_off_array=[]
time_eval=[]
k_save=[]
if benders:

    stop=0
    on_off_value=np.matrix(np.zeros((num_p,n_h)))
    T=[]
    gap=1e-3
    st=time.time()
    for k in range(sim_step+1):
        if k != 0:
            on_off_value=np.rint(on_off.value)

        try:
            pool=Pool(10,initargs=(on_off_value,))
            res = list(zip(*pool.map(lambda i: eval_on_off(on_off_value=on_off_value,
                                                           r=scenarios_bus[:,:,0:benders_sim],
                                                           j=i,prob=prob,Constraints=Constraints,duals=True,solver=ECOS),
                                                           range(benders_sim),chunksize=10)))
            pool.close()
            pool.join()
            Q_value_stoch=list(res[0])
            M1_stoch=list(res[1])
            M2_stoch=list(res[2])
        except:
            print "OS memory"
            break

        Q_eval=np.nanmean(Q_value_stoch)
        if k%5==0 and k !=0 :
            on_off_array.append(on_off_value)
            time_eval.append(time.time()-st)
            k_save.append(k)
            st=time.time()
        cost.append(eval_obj(on_off=on_off_value)+Q_eval)
        
        
        print cost[-1],
        if  stop==1:
            break
        m1=np.nanmean(M1_stoch,0)
        m2=np.nanmean(M2_stoch,0)
        benders_cut=sum([-(m1[i,:])*(on_off[:,i]-on_off_value[:,i]) 
                        +(m2[i,:])*(on_off[:,i]-on_off_value[:,i])
                        for i in range(n_h) ])

        Constraints_det.append(theta>=Q_eval+benders_cut)
        prob_det=[]
        prob_det = Problem(Minimize(Objective_det+theta), Constraints_det)
        F_eval=prob_det.solve(solver=GUROBI,MIPGap=gap,warm_start=True)
        T.append(theta.value)
        lower.append(F_eval)
        print lower[-1],np.sum(on_off_value)
        if k==sim_step-2:
            stop=1
    print "finish benders"
        
# Hybrid Algorithm
det=[]
g_k1=[]
g_k2.value=np.zeros((2*n_h,num_p))
delta_uk0=[]
k0=50
n_par=10
#n_par=1
iter_time_total=[]
if not benders:
    iter_time=time.time()
    if CONT:
        delta_slope_approx1=np.matrix(np.zeros((n_h,num_p)))
        delta_slope_approx2=np.matrix(np.zeros((n_h,num_p)))
        print "start",

   
    on_off_array_cont=[]
 

    for k in range (sim_step/n_par):
        print k
        if k==0:
            st=time.time()
        r_parameter.value=r
        if not CONT:
            print "solving MILP",
            F_eval=prob_det.solve(solver=GUROBI,MIPGap=1e-3,warm_start=True)#GUROBI,MIPGap=1e-1,warm_start=True
        else:
            print "solving relaxation",
            F_eval=prob_det.solve(solver=ECOS,reltol=1e-8)


        print "time", time.time()-st
        det.append(F_eval)
        print det[-1]
        if not CONT:
            on_off_value=np.rint(on_off.value)

        else:
            on_off_value=on_off_c.value

        on_off_array_cont.append(on_off_value)
        m=k
        if k!=0:
            delta_uk0.append(np.linalg.norm((np.squeeze(np.asarray(on_off_array_cont[k]-on_off_array_cont[k-1]))).reshape(1,num_p*n_h),1))
            print "|u_k1-u_k0|=",delta_uk0[-1]

        if not CONT:
            print "Solving PWL"
            Q_eval_approx,delta_slope_approx1,delta_slope_approx2=eval_on_off(on_off_value=on_off_value,r=r,j=-1,
                                                                              prob=prob_approx,Constraints=Constraints_approx,
                                                                              duals=True,solver=ECOS)
        else:

            for i in range(n_h):
                delta_slope_approx1[i,: ]= np.squeeze( np.asarray((np.diag(p_max)*Constraints_det[i].dual_value).T))
            for i in range(n_h):
                delta_slope_approx2[i,: ]= np.squeeze( np.asarray((np.diag(p_min)*Constraints_det[i+n_h].dual_value).T))

        print "Solving QP"
        # parallel
        pool=Pool(n_par+1,initargs=(on_off_value,))# n_par+1
        res = list(zip(*pool.map(lambda i: eval_on_off(on_off_value=on_off_value,
                                                       r=scenarios_bus[:,:,0:sim_step],
                                                       j=i,prob=prob,Constraints=Constraints,duals=True,solver=ECOS),
                                                       range(n_par*m,n_par*(m+1)),chunksize=1)))
        pool.close()
        pool.join()
        Q_value_stoch=list(res[0])
        M1_stoch=list(res[1])
        M2_stoch=list(res[2])
        Q_eval=np.nanmean(Q_value_stoch)
        delta_slope1=np.nanmean(M1_stoch,0)
        delta_slope2=np.nanmean(M2_stoch,0)

        #serial
        
        #Q_eval,delta_slope1,delta_slope2=eval_on_off(on_off_value=on_off_value,r=scenarios_bus[:,:,0:sim_step],j=m,
        #                                             prob=prob,Constraints=Constraints,duals=True,solver=ECOS)


        alpha=1./(k+1+k0)
        if k != 0:
            dg = (np.concatenate((delta_slope1,delta_slope2),axis=0)
                  - np.concatenate((delta_slope_approx1,delta_slope_approx2),axis=0)
                  - g_k1[-1])
            g_k1.append(g_k1[-1] + alpha*dg )
            g_k2.value= g_k1[-1] 
        else:
            dg = np.concatenate((delta_slope1,delta_slope2),axis=0)- np.concatenate((delta_slope_approx1,delta_slope_approx2),axis=0)
            g_k1.append(alpha*dg)
            g_k2.value = g_k1[-1]

        if CONT and (k%5==0 or k==(sim_step/n_par)-1):
            F=prob_bin.solve(solver=GUROBI,MIPGap=1e-2,warm_start=True)
            on_off_value=np.rint(on_off.value)
            on_off_array.append(on_off_value)
            time_eval.append(time.time()-st)
            st=time.time()
            k_save.append(k)
            print "total time", time_eval[-1]
        elif not CONT and (k==0 or (k+1)%5==0 ):#((not np.all(on_off_array_cont[k]==on_off_array_cont[k-1]))  or k==0)
            on_off_array.append(on_off_value)
            time_eval.append(time.time()-st)
            st=time.time()
            k_save.append(k)
            print "saving solution"
            print "total time", time_eval[-1]

        iter_time_total.append(time.time()-iter_time)
        print "iter time", iter_time_total[-1]
        iter_time=time.time()
        #print "total time", time_eval[-1]

    print "finish"


Load_Penalty=np.max(pwl_cost)*int(sys.argv[3])
result={}
result['on_off_array']=on_off_array
result['time_eval']=time_eval
result['iter_time_total']=iter_time_total
result['k_final']=k

variables={}
variables['p']=p
variables['w']=w
variables['s']=s
variables['load']=load
if PWL:
    variables['gk']=gk

constants={}
constants['k_save']=k_save
constants['Load_Penalty']=Load_Penalty
constants['ontime']=ontime
constants['H0']=H0
constants['g0']=g0
constants['pwl_cost']=pwl_cost
constants['on_off_cost']=on_off_cost
constants['on_off_init']=on_off_init
constants['p_max']=p_max
constants['p_min']=p_min
constants['b']=b
constants['rampup']=rampup
constants['rampdown']=rampdown
constants['w_max']=w_max
constants['w_min']=w_min
constants['z_max']=z_max
constants['z_min']=z_min
constants['on_off_init']=on_off_init
constants['pwl_points']=pwl_points
constants['G']=G
constants['Dm']=Dm
constants['R']=R
constants['A']=A
constants['J']=J
constants['PWL']=PWL
constants['n_h']=n_h
constants['num_p']=num_p
constants['n_samples']=n_samples
constants['n_test']=n_test
constants['gen_tech']=gen_tech
constants['r_parameter']=r_parameter
constants['on_off_param']=on_off_param
constants['det']=det
constants['g_k1']=g_k1
constants['delta_uk0']=delta_uk0
constants['cost']=cost
constants['lower']=lower

result['constants']=constants
result['variables']=variables
result['scenarios_bus']=scenarios_bus

if benders:
    name="benders"
else:
    name="hybrid"
save_path="pickle/ieee14/uptime/Stoch_"+name+"_ieee14_"+str(sim_step)+"iter_"+str(pwl_points)+"pwl_"+"relaxation_"+str(CONT)+ "_cluster_warm_start_1_alpha"+str(k0)+"norm_u_"+str(u_bound)+"_gamma_"+str(sys.argv[2])+"_evalgamma_"+str(sys.argv[3])+"_day_"+str(sys.argv[4])+"_mult_"+str(sys.argv[5])+"_parallel_newmix_quad_bugfix_ontime"+".p"
print save_path
dill.dump( result, open( save_path, "wb" ) )

