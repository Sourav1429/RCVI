import numpy as np
from Machine_Rep import *
from Garnet import *
from KL_uncertainity_evaluator import Robust_pol_Kl_uncertainity
import pickle
import time
import pandas as pd

class Garnet:
    def __init__(self,nS=10,nA=5):
        self.nA = nA
        self.nS = nS
    def gen_probability(self):
        self.P = np.zeros((self.nA,self.nS,self.nS))
        for s in range(self.nS):
            for a in range(self.nA):
                mu,sigma = np.random.uniform(0,100),np.random.uniform(0,100)
                self.P[a,s,:] = np.random.normal(mu,sigma,self.nS)
                self.P[a,s] = np.exp(self.P[a,s])
                self.P[a,s] = self.P[a,s]/np.sum(self.P[a,s])
        return self.P
    def gen_reward(self):
        R = np.zeros((self.nA,self.nS,self.nS))
        self.exp_rew = self.gen_expected_reward()
        for a in range(self.nA):
            for s in range(self.nS):
                R[a,s,:] = self.exp_rew[s,a]
        return R
    def gen_cost(self):
        R = np.zeros((self.nA,self.nS,self.nS))
        self.exp_rew = self.gen_expected_constraint()
        for a in range(self.nA):
            for s in range(self.nS):
                R[a,s,:] = self.exp_rew[s,a]
        return R
    def gen_expected_reward(self):
        self.R = np.zeros((self.nS,self.nA))
        for s in range(self.nS):
            for a in range(self.nA):
                mu,sigma = np.random.uniform(0,10),np.random.uniform(0,10)
                self.R[s,a] = np.random.normal(mu,sigma)/10
        return self.R
    def gen_expected_constraint(self):
        self.R = np.zeros((self.nS,self.nA))
        for s in range(self.nS):
            for a in range(self.nA):
                mu,sigma = np.random.uniform(0,10),np.random.uniform(0,10)
                self.R[s,a] = np.random.normal(mu,sigma)/10
        return self.R
    
def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

def get_policy_from_theta(theta):
    nS, nA = theta.shape
    return np.array([softmax(theta[s]) for s in range(nS)])

def kl_divergence(pi_new, pi_old):
    kl = 0.0
    for s in range(len(pi_new)):
        kl += np.sum(pi_new[s] * (np.log(pi_new[s] + 1e-8) - np.log(pi_old[s] + 1e-8)))
    return kl

def flatten_grad(grad):
    return grad.flatten()

def reshape_grad(vec, shape):
    return vec.reshape(shape)

def natural_gradient_update(theta, grad, kl_lambda, alpha,ch_dep,ch):
    # Perform a natural gradient-like update based on the objective:
    # max_\theta_new grad^T (\theta_new - \theta) - \lambda KL(\theta_new || \theta)
    # This is equivalent to solving: \theta_new = \theta + 1/\lambda * grad
    if(ch_dep==0):
        return theta - alpha * grad / kl_lambda
    elif(ch_dep==1):
        if(ch==0):
           return theta + alpha * grad / kl_lambda 
        else:
            return theta - alpha * grad / kl_lambda
    else:
        return theta + alpha * grad / kl_lambda


# === Environment Setup === #
nS,nA = 10,5
env = River_swim()
env_dep = 1  #0 for MR, 1 for RS and 2 for Garnet
nS, nA = env.nS, env.nA
P = env.gen_probability()
R = env.gen_expected_reward()
C = env.gen_expected_cost()

# === Oracle === #
cost_list = [R, C]
init_dist = np.exp(np.random.normal(0,1,nS))
init_dist = init_dist/np.sum(init_dist)
init_dist = init_dist.tolist()
rpe = Robust_pol_Kl_uncertainity(nS, nA, cost_list, init_dist, alpha=0.000001)

# === Parameters === #
C_KL = 0.005
kl_lambda = 50
alpha = 0.1
#T = 1000
T = 1000
b = 4

# === Initialize theta === #
theta = np.random.randn(nS, nA)
vf = []
cf = []
start = time.time()
for t in range(T):
    policy = get_policy_from_theta(theta)

    # Get both objectives and gradients
    J_v, grad_v = rpe.evaluate_policy(policy, P, C_KL, n=0, t=t)
    J_c, grad_c = rpe.evaluate_policy(policy, P, C_KL, n=1, t=t)
    vf.append(J_v)
    cf.append(J_c)

    # Choose which gradient to follow
    if env_dep!=2:
        ch = np.argmax([J_v, kl_lambda*(np.max(J_c-b,0))])
    else:
        ch = np.argmax([J_v, kl_lambda*(np.max(b-J_c,0))])
    grad = grad_v if ch == 0 else grad_c

    # Flatten gradient and apply natural-like update
    grad_vec = flatten_grad(grad)
    theta_vec = flatten_grad(theta)
    theta_new_vec = natural_gradient_update(theta_vec, grad_vec, kl_lambda, alpha,env_dep,ch)
    theta_new = reshape_grad(theta_new_vec, (nS, nA))

    # Check KL divergence
    pi_old = get_policy_from_theta(theta)
    pi_new = get_policy_from_theta(theta_new)
    kl = kl_divergence(pi_new, pi_old)

    # Accept the update
    theta = theta_new

    print(f"[Iter {t}] J_v={J_v:.4f}, J_c={J_c:.4f}, KL={kl:.6f}, ch={ch}")

print("Time taken:",time.time()-start)

data = {'vf':vf,'cf':cf}
df = pd.DataFrame(data)
df.to_excel('VF_CF_kl_lambda_Gar_RNPG_aistats_rvi_baseline_comp.xlsx')

# Final policy
#final_policy = get_policy_from_theta(theta)
#print("Final policy:")
#print(final_policy)
