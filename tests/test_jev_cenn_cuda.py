"""Jev_CeNN CUDA fallback, stride and complete Colab integration contract."""
import ast
import json
from pathlib import Path

import pytest
import torch

from tinycenn_lm import jev_cenn_cuda
from tinycenn_lm import sliding_window_fusion as swf

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"notebooks"/"Jev_CeNN_CUDA_Fast_AllAttention_Colab.ipynb"


@pytest.mark.parametrize("t,radius",[(9,4),(73,16),(129,64)])
def test_fast_cuda_wrapper_cpu_reference_and_autograd(t,radius):
    torch.manual_seed(2028)
    b,h,d=2,2,16
    # Match ModernBERT QKV views' noncontiguous strides.
    stack=torch.randn(b,t,3,h,d)
    q,k,v=stack.transpose(3,1).unbind(2)
    q=q.detach().requires_grad_(True)
    k=k.detach().requires_grad_(True)
    v=v.detach().requires_grad_(True)
    valid=torch.ones(b,t,dtype=torch.bool)
    valid[1,-3:]=False
    got=jev_cenn_cuda.fast_sliding_sdpa(
        q,k,v,valid,radius=radius,block_size=32,
        reference=swf.local_sliding_sdpa)
    expected=swf.local_sliding_sdpa(q,k,v,valid,radius=radius,block_size=32)
    torch.testing.assert_close(got,expected,rtol=1e-6,atol=1e-6)
    got.sum().backward()
    for tensor in (q,k,v):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_fast_cuda_backend_install_is_reversible():
    original=swf.local_sliding_sdpa
    flag=jev_cenn_cuda.install_fast_sliding_backend(swf)
    assert isinstance(flag,bool)
    assert swf.local_sliding_sdpa is not original
    jev_cenn_cuda.install_fast_sliding_backend(swf)
    assert swf._jev_original_sliding_sdpa is original
    jev_cenn_cuda.uninstall_fast_sliding_backend(swf)
    assert swf.local_sliding_sdpa is original


def test_jev_cenn_colab_fully_replaces_trains_exports():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"]==4 and len(nb["cells"])>=17
    cells=[c for c in nb["cells"] if c["cell_type"]=="code"]
    allcode="\n".join("".join(c["source"]) for c in cells)
    for feature in (
        "Jev_CeNN", "KERNEL_PREFLIGHT", "STRICT_KERNEL_PARITY",
        "install_fast_sliding_backend", "SHARED_TEACHER_CACHE",
        "_capture_all_layers", "DEV_EVAL_EVERY_LAYERS",
        "GLOBAL_LAYERS", "SLIDING_LAYERS", "SlidingWindowFusion(",
        "build_v28(FULL_WINDOW)", "JOINT_STEPS",
        "decision_distill_loss(", 'split="test"',
        "export_v31_all_attention(", "verify_standalone_bundle(",
        "upload_standalone_to_hub(", "PUBLISH_ONLY_IF_PASSES",
        'HF_PRIVATE = True', "student_state",
        "runtime_file.write_text(", '"architecture"]="Jev_CeNN"',
    ):
        assert feature in allcode,feature
    assert all(not c.get("outputs") and c.get("execution_count") is None
               for c in cells)
    assert allcode.index("final_checkpoint=OUTPUT_DIR") < allcode.index('split="test"')
    # Triton implementation must live in the independent model bundle, not
    # depend on importing the research repo on Hugging Face deployment.
    assert "_jev_reference_sliding" in allcode
    assert 'from __future__ import annotations", "")' in allcode


def test_cuda_module_does_not_change_checkpoint_parameter_names():
    # It is a pure function; the V3.0 model remains the original nn.Module.
    src=(ROOT/"src"/"tinycenn_lm"/"jev_cenn_cuda.py").read_text()
    tree=ast.parse(src)
    assert not any(isinstance(n,ast.ClassDef) and n.name.endswith("Attention")
                   for n in ast.walk(tree))
    for name in ("fast_sliding_sdpa","install_fast_sliding_backend",
                 "uninstall_fast_sliding_backend"):
        assert name in src
