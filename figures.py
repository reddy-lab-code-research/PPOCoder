import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from IPython.display import display


#load baselines
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
res = pd.read_csv('results/codet5_ppo_steps.csv')[['bs', 'input_len','reward_type', 'output_len','run' ,'epoch','step','sample_reward','sample_nerrors','sample_nerrors_ref','sample_nnodes','sample_nnodes_ref']]
batch_size = 16
input_len = 400
output_len = 400
run=8
reward_id = 2
res = res[(res['reward_type'] == 'reward_'+str(reward_id)) & (res['bs'] == batch_size) & (res['input_len']==input_len) & (res['output_len']==output_len)&(res['run']==run)]

# plt.figure(figsize=(10, 10))
colors = ['blue','green','red','black','orange']
titles = ['reward','nerrors','nerrors_ref','nnodes','nnodes_ref']
fig, axes = plt.subplots(2,3, figsize=(16,8))
fig.delaxes(axes[1,2])
k = 0

# print(max(res.loc[:,"sample_reward"]), min(res.loc[:,"sample_reward"]))
# print(max(res.loc[:,"sample_nnodes"]), min(res.loc[:,"sample_nnodes"]))

for i in range(2):
    for j in range(3):
        if i == 1 and j == 2:
            break
        axes[i,j].plot(range(len(res.loc[:,"sample_"+titles[k]])),res.loc[:,"sample_"+titles[k]],color = colors[k])
        axes[i,j].set_title("sample_"+titles[k])
        print(k)
        k += 1

fig.savefig('figures/codet5_ppo_reward%s_bs%d_in-len%d_out-len%d_r%d.png'%(reward_id,batch_size,input_len,output_len,run), bbox_inches = 'tight')

