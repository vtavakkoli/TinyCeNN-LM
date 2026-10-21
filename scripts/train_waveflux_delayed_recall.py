"""Small CPU/GPU functional learning smoke test. Not a scientific benchmark."""
import argparse, json, time
import torch
from tinycenn_lm.waveflux_cenn import WaveFluxConfig, WaveFluxCeNNLM

parser=argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=120)
parser.add_argument('--delay',type=int,default=8)
parser.add_argument('--batch',type=int,default=64)
parser.add_argument('--classes',type=int,default=4)
parser.add_argument('--device',type=str,default='cuda' if torch.cuda.is_available() else 'cpu')
a=parser.parse_args()
torch.manual_seed(1234)
if a.device=='cpu':torch.set_num_threads(2)
cfg=WaveFluxConfig(hidden_size=32,groups=8,cell_width=4,velocities=(0,1,-1,2,-2),
                   initial_write=0.08, initial_decay=0.999, residual_gain=0.8)
model=WaveFluxCeNNLM(a.classes+2,cfg,layers=2).to(a.device)
optim=torch.optim.AdamW(model.parameters(), lr=0.005,weight_decay=0.0)

# First token is randomly selected symbol in 1..classes, then delay filler tokens 0,
# then an uninformative query token (classes+1). Only last output is trained to
# identify the FIRST symbol. There is no shortcut in the last token's embedding.
def batch(n):
    value=torch.randint(1,a.classes+1,(n,),device=a.device)
    seq=torch.zeros(n,a.delay+2,dtype=torch.long,device=a.device)
    seq[:,0]=value
    seq[:,-1]=a.classes+1
    return seq,value

start=time.time()
initial=None
for i in range(a.steps):
    model.train()
    ids,target=batch(a.batch)
    logits=model(ids)[:,-1,1:a.classes+1]
    loss=torch.nn.functional.cross_entropy(logits,target-1)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    optim.step()
    if i==0: initial=float(loss.item())

model.eval()
with torch.no_grad():
    ids,target=batch(512)
    pred=model(ids)[:,-1,1:a.classes+1].argmax(-1)+1
    acc=(pred==target).float().mean().item()
print(json.dumps({'model':'WaveFluxCeNNLM','test':'delayed_first_symbol',
                  'steps':a.steps,'delay':a.delay,'classes':a.classes,
                  'chance_accuracy':1/a.classes,'accuracy':acc,
                  'initial_train_loss':initial,'last_train_loss':float(loss.item()),
                  'seconds':round(time.time()-start,2),'device':a.device},indent=2))
