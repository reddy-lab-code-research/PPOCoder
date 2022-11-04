import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from IPython.display import display


#load baselines
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
l1, l2 = 'cpp', 'python'
######################
res = pd.read_csv('results/codet5_ppo_' +l1+'_'+l2+'6.csv')[['bs', 'lr', 'input_len','reward_type', 'output_len','run' ,'epoch','step','sample_reward','sample_nerrors','sample_nerrors_ref','sample_nnodes','sample_nnodes_ref','sample_kl','sample_entropy','sample_loss','sample_pg_loss','sample_vf_loss','sample_advg','sample_val','sample_comp-rate','sample_ast-match','sample_dfg-match']]
# res = pd.read_csv('results/codet5_ppo_' +l1+'_'+l2+'5.csv')[['bs', 'lr', 'input_len','reward_type', 'output_len','run' ,'epoch','step','sample_reward','sample_nerrors','sample_nerrors_ref','sample_nnodes','sample_nnodes_ref','sample_kl','sample_entropy','sample_loss','sample_pg_loss','sample_vf_loss','sample_advg','sample_val']]
# res = pd.read_csv('results/codet5_ppo_steps5.csv')[['bs', 'lr', 'input_len','reward_type', 'output_len','run' ,'epoch','step','sample_reward','sample_nerrors','sample_nerrors_ref','sample_nnodes','sample_nnodes_ref','sample_kl','sample_entropy','sample_loss','sample_pg_loss','sample_vf_loss','sample_advg','sample_val']]
# res = pd.read_csv('results/codet5_ppo_steps4.csv')[['bs', 'lr', 'input_len','reward_type', 'output_len','run' ,'epoch','step','sample_reward','sample_nerrors','sample_nerrors_ref','sample_nnodes','sample_nnodes_ref','sample_kl','sample_entropy','sample_loss','sample_pg_loss','sample_vf_loss']]
# res = pd.read_csv('results/codet5_ppo_steps3.csv')[['bs', 'lr', 'input_len','reward_type', 'output_len','run' ,'epoch','step','sample_reward','sample_nerrors','sample_nerrors_ref','sample_nnodes','sample_nnodes_ref','sample_kl','sample_entropy',]]
######################
batch_size = 16
# batch_size = 48
input_len = 320
output_len = 320
# run= 146
run= 6
reward_id = 2
lr = 1e-6
# lr = 1e-5
# lr = 1e-7
kl_coef = 100
reward_W = 0.01
res = res[(res['reward_type'] == 'reward_'+str(reward_id)) & (res['bs'] == batch_size) & (res['input_len']==input_len) & (res['output_len']==output_len)&(res['run']==run) & (res['lr']==lr)]



##########################################
# plt.figure(figsize=(10, 10))
colors = ['blue','green','red','black','orange','purple','pink','grey','gold','teal','brown','lime','olive','aqua','magenta']
titles = ['reward','nerrors','nerrors_ref','nnodes','nnodes_ref','kl','entropy','loss','pg_loss','vf_loss','advg','val','comp-rate','ast-match','dfg-match']
fig, axes = plt.subplots(3,5, figsize=(20,10))
k = 0
for i in range(3):
    for j in range(5):
        axes[i,j].plot(range(len(res.loc[:,"sample_"+titles[k]])),res.loc[:,"sample_"+titles[k]],color = colors[k])
        axes[i,j].set_title("sample_"+titles[k])
        print(k)
        k += 1
##########################################
# plt.figure(figsize=(10, 10))
# colors = ['blue','green','red','black','orange','purple','pink','grey','gold','teal','brown','lime']
# titles = ['reward','nerrors','nerrors_ref','nnodes','nnodes_ref','kl','entropy','loss','pg_loss','vf_loss','advg','val']
# fig, axes = plt.subplots(3,4, figsize=(16,10))
# k = 0
# for i in range(3):
#     for j in range(4):
#         axes[i,j].plot(range(len(res.loc[:,"sample_"+titles[k]])),res.loc[:,"sample_"+titles[k]],color = colors[k])
#         axes[i,j].set_title("sample_"+titles[k])
#         print(k)
#         k += 1
##########################################        
# plt.figure(figsize=(10, 10))
# colors = ['blue','green','red','black','orange','purple','pink','grey','gold','teal']
# titles = ['reward','nerrors','nerrors_ref','nnodes','nnodes_ref','kl','entropy','loss','pg_loss','vf_loss']
# fig, axes = plt.subplots(3,4, figsize=(16,8))
# fig.delaxes(axes[1,3])
# fig.delaxes(axes[2,3])
# k = 0
# for i in range(3):
#     for j in range(4):
#         if (i == 2 and j == 3) or (i == 1 and j == 3):
#             break
#         axes[i,j].plot(range(len(res.loc[:,"sample_"+titles[k]])),res.loc[:,"sample_"+titles[k]],color = colors[k])
#         axes[i,j].set_title("sample_"+titles[k])
#         print(k)
#         k += 1
##########################################
# plt.figure(figsize=(10, 10))
# colors = ['blue','green','red','black','orange','purple','pink']
# titles = ['reward','nerrors','nerrors_ref','nnodes','nnodes_ref','kl','entropy']
# fig, axes = plt.subplots(2,4, figsize=(16,8))
# fig.delaxes(axes[1,3])
# k = 0
# for i in range(2):
#     for j in range(4):
#         if (i == 1 and j == 3):
#             break
#         axes[i,j].plot(range(len(res.loc[:,"sample_"+titles[k]])),res.loc[:,"sample_"+titles[k]],color = colors[k])
#         axes[i,j].set_title("sample_"+titles[k])
#         print(k)
#         k += 1     
##########################################

# fig.savefig('figures/codet5_ppo_reward%s_bs%d_in-len%d_out-len%d_lr%.5f_FixKLcoef%d_rewardW%.1f_r%d.png'%(reward_id,batch_size,input_len,output_len,lr,kl_coef,reward_W,run), bbox_inches = 'tight')
fig.savefig('figures/codet5_ppo_reward%s_bs%d_in-len%d_out-len%d_lr%.6f_FixKLcoef%.1f_rewardW%.2f_r%d.png'%(reward_id,batch_size,input_len,output_len,lr,kl_coef,reward_W,run), bbox_inches = 'tight')
# fig.savefig('figures/codet5_ppo_'+l1+'_'+l2+'_reward%s_bs%d_in-len%d_out-len%d_lr%.6f_FixKLcoef%.1f_rewardW%.2f_r%d.png'%(reward_id,batch_size,input_len,output_len,lr,kl_coef,reward_W,run), bbox_inches = 'tight')
# 
