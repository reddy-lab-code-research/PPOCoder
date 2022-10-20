import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from IPython.display import display


#load baselines
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
res = pd.read_csv('results/codet5_ppo_steps2.csv')[['bs', 'lr', 'input_len','reward_type', 'output_len','run' ,'epoch','step','sample_reward','sample_nerrors','sample_nerrors_ref','sample_nnodes','sample_nnodes_ref','sample_kl','sample_entropy']]
batch_size = 16
input_len = 400
output_len = 400
run= 48
reward_id = 2
lr = 1e-6
kl_coef = 100
reward_W = 0.01
res = res[(res['reward_type'] == 'reward_'+str(reward_id)) & (res['bs'] == batch_size) & (res['input_len']==input_len) & (res['output_len']==output_len)&(res['run']==run) & (res['lr']==lr)]

# plt.figure(figsize=(10, 10))
colors = ['blue','green','red','black','orange','purple','pink']
titles = ['reward','nerrors','nerrors_ref','nnodes','nnodes_ref','kl','entropy']
fig, axes = plt.subplots(2,4, figsize=(16,8))
fig.delaxes(axes[1,3])
k = 0

# print(max(res.loc[:,"sample_reward"]), min(res.loc[:,"sample_reward"]))
# print(max(res.loc[:,"sample_nnodes"]), min(res.loc[:,"sample_nnodes"]))

for i in range(2):
    for j in range(4):
        if i == 1 and j == 3:
            break
        axes[i,j].plot(range(len(res.loc[:,"sample_"+titles[k]])),res.loc[:,"sample_"+titles[k]],color = colors[k])
        axes[i,j].set_title("sample_"+titles[k])
        print(k)
        k += 1

# fig.savefig('figures/codet5_ppo_reward%s_bs%d_in-len%d_out-len%d_lr%.5f_FixKLcoef%d_rewardW%.1f_r%d.png'%(reward_id,batch_size,input_len,output_len,lr,kl_coef,reward_W,run), bbox_inches = 'tight')
fig.savefig('figures/codet5_ppo_reward%s_bs%d_in-len%d_out-len%d_lr%.6f_FixKLcoef%.1f_rewardW%.2f_r%d.png'%(reward_id,batch_size,input_len,output_len,lr,kl_coef,reward_W,run), bbox_inches = 'tight')

