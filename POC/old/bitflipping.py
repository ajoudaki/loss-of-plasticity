import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
from datetime import datetime

# --- Configuration ---
# Experiment Duration
num_total_examples = 250000
T_flip_interval = 2500
log_interval = 1000

# Problem Parameters
m_input_bits = 20
f_flipping_bits = 15
n_hidden_target = 100
beta_ltu = 0.7

# Neural Network (LearnerNet) Parameters
n_hidden_learner = 5
learner_activation_fn = nn.ReLU()
USE_DROPOUT = False # <<<<------ NEW FLAG
DROPOUT_PROB = 0.15 # <<<<------ Dropout Probability (try 0.1, 0.15, 0.2)

# Optimizer & Learning Rate
USE_ADAM_FOR_NN = True
USE_ADAM_FOR_LINEAR = True

if USE_ADAM_FOR_NN:
    nn_learning_rate = 5e-4
else:
    nn_learning_rate = 1e-3

if USE_ADAM_FOR_LINEAR:
    linear_learning_rate = 1e-3
else:
    linear_learning_rate = 1e-2

batch_size = 32

# --- Matplotlib Style ---
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 14,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.titlesize': 16
})

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
opt_nn_name = "Adam" if USE_ADAM_FOR_NN else "SGD"
opt_linear_name = "Adam" if USE_ADAM_FOR_LINEAR else "SGD"
dropout_info = f"_Dropout{DROPOUT_PROB}" if USE_DROPOUT else ""

print(f"NN Optimizer: {opt_nn_name}, LR: {nn_learning_rate}{dropout_info.replace('_', ', ')}")
print(f"Linear Optimizer: {opt_linear_name}, LR: {linear_learning_rate}")
print(f"Total examples: {num_total_examples}, Flip interval: {T_flip_interval} (~{num_total_examples//T_flip_interval} flips)")


# --- Target Network Definition (Identical) ---
class TargetNet(nn.Module):
    def __init__(self, input_size, n_hidden, beta):
        super(TargetNet, self).__init__()
        self.input_size=input_size; self.n_hidden=n_hidden; self.beta=beta
        self.fc1_weights=nn.Parameter(torch.empty(n_hidden,self.input_size,device=device),requires_grad=False)
        self.fc1_thresholds=nn.Parameter(torch.empty(n_hidden,device=device),requires_grad=False)
        self.fc2=nn.Linear(n_hidden,1).to(device)
        self._initialize_weights()
    def _initialize_weights(self):
        self.fc1_weights.data=(torch.randint(0,2,self.fc1_weights.shape,device=device).float()*2-1)
        S_i=(self.fc1_weights.data==-1).sum(dim=1).float()
        self.fc1_thresholds.data=(self.input_size*self.beta)-S_i
        nn.init.kaiming_uniform_(self.fc2.weight,nonlinearity='linear');nn.init.zeros_(self.fc2.bias)
    def ltu_activation(self,x,thresholds):return(x>thresholds.unsqueeze(0)).float()
    def forward(self,x):
        h=self.ltu_activation(torch.matmul(x,self.fc1_weights.t()),self.fc1_thresholds)
        return self.fc2(h)

# --- Learner Network Definition (MODIFIED with Dropout) ---
class LearnerNet(nn.Module):
    def __init__(self, input_size, n_hidden, activation_fn, use_dropout=False, dropout_prob=0.5):
        super(LearnerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, n_hidden)
        self.activation = activation_fn
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(n_hidden, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        if isinstance(self.activation,nn.ReLU):nl='relu'
        elif isinstance(self.activation,nn.Tanh):nl='tanh'
        else:nl='leaky_relu'
        g=nn.init.calculate_gain(nl)if nl not in['sigmoid']else 1.0
        if nl in['relu','leaky_relu']:nn.init.kaiming_uniform_(self.fc1.weight,nonlinearity=nl)
        elif nl in['tanh','sigmoid']:nn.init.xavier_uniform_(self.fc1.weight,gain=g)
        else:nn.init.kaiming_uniform_(self.fc1.weight,a=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight,nonlinearity='linear');nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        if self.use_dropout and self.training: # Dropout only during training
            x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- Linear Model Definition (Identical) ---
class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    def forward(self, x): return self.linear(x)

# --- Data Generation (Identical) ---
class BitFlippingDataGenerator:
    def __init__(self,m,f,T,target_net_instance):
        self.m=m;self.f=f;self.T=T;self.target_net=target_net_instance
        self.input_size_to_net=self.m+1
        self.flipping_bits_state=torch.randint(0,2,(self.f,),device=device,dtype=torch.float32)
        self.current_example_count_in_generator=0
    def _generate_single_example_tensors(self):
        if self.current_example_count_in_generator>0 and self.current_example_count_in_generator%self.T==0:
            # Simplified flip: pick one bit and flip it
            idx_to_flip = random.randint(0, self.f - 1)
            self.flipping_bits_state[idx_to_flip]=1-self.flipping_bits_state[idx_to_flip]
        r_bits=torch.randint(0,2,(self.m-self.f,),device=device,dtype=torch.float32)
        b_bit=torch.ones(1,device=device,dtype=torch.float32)
        i_vec=torch.cat((self.flipping_bits_state,r_bits,b_bit))
        self.current_example_count_in_generator+=1
        return i_vec
    def generate_batch(self,batch_size_val):
        b_x_l=[self._generate_single_example_tensors()for _ in range(batch_size_val)]
        b_x_t=torch.stack(b_x_l)
        with torch.no_grad():b_y_t=self.target_net(b_x_t).squeeze(-1)
        return b_x_t,b_y_t

# --- Main Experiment ---
def run_experiment():
    start_time=time.time()
    input_dim_to_nets=m_input_bits+1
    figures_dir="figures";os.makedirs(figures_dir,exist_ok=True)
    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    learner_name=learner_activation_fn.__class__.__name__
    base_filename=f"s_c_reg_{learner_name}_{opt_nn_name}LR{nn_learning_rate}{dropout_info}_Lin{opt_linear_name}LR{linear_learning_rate}_ex{num_total_examples}_flips{T_flip_interval}_{timestamp}"

    target_network_instance=TargetNet(input_dim_to_nets,n_hidden_target,beta_ltu).to(device);target_network_instance.eval()
    learner_nn=LearnerNet(input_dim_to_nets,n_hidden_learner,learner_activation_fn, use_dropout=USE_DROPOUT, dropout_prob=DROPOUT_PROB).to(device) # Pass dropout params
    learner_linear=LinearModel(input_dim_to_nets).to(device)

    if USE_ADAM_FOR_NN:optimizer_nn=optim.Adam(learner_nn.parameters(),lr=nn_learning_rate)
    else:optimizer_nn=optim.SGD(learner_nn.parameters(),lr=nn_learning_rate)
    if USE_ADAM_FOR_LINEAR:optimizer_linear=optim.Adam(learner_linear.parameters(),lr=linear_learning_rate)
    else:optimizer_linear=optim.SGD(learner_linear.parameters(),lr=linear_learning_rate)
    
    criterion=nn.MSELoss()
    data_generator=BitFlippingDataGenerator(m_input_bits,f_flipping_bits,T_flip_interval,target_network_instance)

    nn_errors_raw=[];linear_errors_raw=[]
    nn_errors_binned=[];linear_errors_binned=[]
    example_counts_binned=[]
    
    num_batches=num_total_examples//batch_size;actual_total_examples=num_batches*batch_size
    print("Starting training...")
    for batch_idx in range(num_batches):
        current_example_num=(batch_idx+1)*batch_size
        inputs,targets=data_generator.generate_batch(batch_size)

        learner_nn.train() # Ensure dropout is active
        optimizer_nn.zero_grad()
        outputs_nn=learner_nn(inputs)
        loss_nn=criterion(outputs_nn.squeeze(-1),targets)
        loss_nn.backward()
        optimizer_nn.step()
        nn_errors_raw.append(loss_nn.item())

        learner_linear.train()
        optimizer_linear.zero_grad()
        outputs_linear=learner_linear(inputs)
        loss_linear=criterion(outputs_linear.squeeze(-1),targets)
        loss_linear.backward()
        optimizer_linear.step()
        linear_errors_raw.append(loss_linear.item())

        if current_example_num%log_interval==0 or batch_idx==num_batches-1:
            n_b_in_log=log_interval//batch_size
            s_idx_nn=max(0,len(nn_errors_raw)-n_b_in_log)
            s_idx_lin=max(0,len(linear_errors_raw)-n_b_in_log)
            avg_loss_nn=np.mean(nn_errors_raw[s_idx_nn:])if s_idx_nn<len(nn_errors_raw)else(nn_errors_raw[-1]if nn_errors_raw else 0)
            avg_loss_linear=np.mean(linear_errors_raw[s_idx_lin:])if s_idx_lin<len(linear_errors_raw)else(linear_errors_raw[-1]if linear_errors_raw else 0)
            print(f"Ex [{current_example_num}/{actual_total_examples}], NN Loss: {avg_loss_nn:.4f}, Linear Loss: {avg_loss_linear:.4f}")
            nn_errors_binned.append(avg_loss_nn);linear_errors_binned.append(avg_loss_linear)
            example_counts_binned.append(current_example_num)

    end_time=time.time();print(f"Training finished in {end_time-start_time:.2f} seconds.")
    print(f"Total bit flips in generator: {data_generator.current_example_count_in_generator//T_flip_interval}")

    fig,ax=plt.subplots(figsize=(10,6))
    nn_label = f'Neural Net ({learner_name}, {opt_nn_name}, LR={nn_learning_rate}'
    if USE_DROPOUT:
        nn_label += f', DO={DROPOUT_PROB})'
    else:
        nn_label += ')'
    ax.plot(example_counts_binned,nn_errors_binned,label=nn_label,marker='.',linestyle='-',markersize=4,linewidth=1.5,color='tab:blue')
    ax.plot(example_counts_binned,linear_errors_binned,label=f'Linear Model ({opt_linear_name}, LR={linear_learning_rate})',marker='x',linestyle='--',markersize=4,linewidth=1.0,color='tab:green')
    flip_legend_added=False
    for i in range(1,(data_generator.current_example_count_in_generator//T_flip_interval)+1):
        f_ex_num=i*T_flip_interval
        if f_ex_num<=actual_total_examples:
            ax.axvline(x=f_ex_num,color='salmon',linestyle=':',alpha=0.6,linewidth=1.0,label='Bit Flip'if not flip_legend_added else None)
            flip_legend_added=True
    ax.set_xlabel("Number of Examples")
    ax.set_ylabel(f"Average MSE (Smoothed over ~{log_interval} Examples)")
    ax.set_title(f"Performance on Slowly-Changing Regression Task",loc='center')
    fig.suptitle(f"{actual_total_examples} Examples, {data_generator.current_example_count_in_generator//T_flip_interval} Flips",fontsize=10,y=0.92)
    ax.legend(loc='best');ax.grid(True,linestyle=':',alpha=0.5);ax.set_yscale('log')
    fig.tight_layout(rect=[0,0,1,0.96])
    pdf_filename=os.path.join(figures_dir,f"{base_filename}.pdf")
    try:
        fig.savefig(pdf_filename,format='pdf',bbox_inches='tight');print(f"Figure saved to {pdf_filename}")
    except Exception as e:print(f"Error saving figure: {e}")
    plt.show()
    # plt.close(fig)

if __name__=="__main__":
    run_experiment()