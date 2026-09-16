import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v21_patches_real_gemma4_causal_classmethod():
    code = r'''
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [str(root), str(root / "src")]
import scripts.benchmark_gemma4_e2b_integrated_memory_v2 as benchmark
from transformers import Gemma4ForCausalLM
assert getattr(Gemma4ForCausalLM.from_pretrained, "__func__", None) is benchmark._mapped_from_pretrained.__func__
print("REAL_CLASSMETHOD_INTERCEPT_OK")
'''
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout
    assert "REAL_CLASSMETHOD_INTERCEPT_OK" in result.stdout
