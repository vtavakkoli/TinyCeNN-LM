"""Reference tests for attention-free WaveFlux-CeNN; CPU or CUDA."""
import torch
from tinycenn_lm.waveflux_cenn import WaveFluxConfig, WaveFluxCeNN, WaveFluxCeNNLM


def toy():
    return WaveFluxCeNN(WaveFluxConfig(hidden_size=24,groups=6,cell_width=4,
                           velocities=(0,1,-1,2),initial_decay=0.99))


def test_affine_scan_matches_recurrence():
    torch.manual_seed(1)
    cell=toy().eval()
    x=torch.randn(2,31,24)
    with torch.no_grad():
        y1,s1=cell(x,return_state=True,mode='scan')
        y2,s2=cell(x,return_state=True,mode='recurrent')
    torch.testing.assert_close(y1,y2,atol=3e-6,rtol=3e-5)
    torch.testing.assert_close(s1,s2,atol=3e-6,rtol=3e-5)


def test_chunked_matches_whole_sequence():
    torch.manual_seed(2)
    cell=toy().eval()
    x=torch.randn(2,37,24)
    with torch.no_grad():
        target,final=cell(x,return_state=True)
        outputs=[];s=None;start=0
        for stop in (1,3,16,18,37):
            y,s=cell(x[:,start:stop],s,return_state=True)
            outputs.append(y)
            start=stop
    torch.testing.assert_close(torch.cat(outputs,1),target,atol=3e-6,rtol=3e-5)
    torch.testing.assert_close(s,final,atol=3e-6,rtol=3e-5)


def test_strict_causality():
    torch.manual_seed(3)
    cell=toy().eval()
    x=torch.randn(1,29,24)
    pert=x.clone();pert[:,13:]+=100*torch.randn_like(pert[:,13:])
    with torch.no_grad():
        a=cell(x);b=cell(pert)
    torch.testing.assert_close(a[:,:13],b[:,:13],atol=2e-6,rtol=2e-5)


def test_flux_conserves_cellular_mass():
    torch.manual_seed(4)
    cell=toy()
    s=torch.randn(2,7,4,6,4)
    k=torch.rand(2,7,4,6,1)*0.24
    out=cell.flux_readout(s,k)
    torch.testing.assert_close(s.sum(dim=3),out.sum(dim=3),atol=2e-6,rtol=2e-5)


def test_transport_moves_to_adjacent_cell():
    cell=toy()
    state=torch.zeros(1,4,6,4)
    state[0,1,2,0]=1.0 # lane velocity +1
    a=torch.ones(1,4,4,6,1)
    b=torch.zeros(1,4,4,6,4)
    trace=cell._scan(a,b,state)
    assert trace[0,0,1,3,0]==1
    assert trace[0,1,1,4,0]==1
    assert trace[0,3,1,0,0]==1


def test_finite_gradients_and_learning_interface():
    torch.manual_seed(5)
    model=WaveFluxCeNNLM(32,WaveFluxConfig(hidden_size=24,groups=6,cell_width=4,
        velocities=(0,1,-1)),layers=2)
    ids=torch.randint(0,32,(2,9))
    logits=model(ids)
    loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,32),ids[:,1:].reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_full_lm_cached_chunk_equivalence():
    torch.manual_seed(6)
    model=WaveFluxCeNNLM(16,WaveFluxConfig(hidden_size=24,groups=6,cell_width=4,
        velocities=(0,1,-1)),layers=2).eval()
    ids=torch.randint(0,16,(2,23))
    with torch.no_grad():
        full,all_states=model(ids,return_states=True)
        chunks=[];s=None;start=0
        for end in (1,6,19,23):
            logits,s=model(ids[:,start:end],s,return_states=True)
            chunks.append(logits);start=end
    torch.testing.assert_close(full,torch.cat(chunks,1),atol=3e-6,rtol=3e-5)
    for a,b in zip(s,all_states):torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-5)
