import torch
import importlib.util
import threading
import sys

def load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

prob_path = sys.argv[1]
kern_path = sys.argv[2]

prob = load(prob_path, "prob")
opt = load(kern_path, "opt")
ref_model = prob.Model(*prob.get_init_inputs()).cuda().eval()
opt_model = opt.ModelNew(*prob.get_init_inputs()).cuda().eval()

print("=== Correctness (10 random) ===")
max_d = 0.0
fails = 0
for i in range(10):
    inp = [x.cuda() for x in prob.get_inputs()]
    with torch.no_grad():
        r = ref_model(*inp)
        o = opt_model(*inp)
    d = (r - o).abs().max().item()
    max_d = max(max_d, d)
    if not torch.allclose(r, o, rtol=1e-3, atol=1e-3):
        fails += 1
        print("  Trial %d: FAIL diff=%.6f" % (i, d))
if fails == 0:
    print("  All 10 correct. Max diff: %.8f" % max_d)
else:
    print("  %d/10 FAILED" % fails)

print("\n=== Determinism ===")
inp = [x.cuda() for x in prob.get_inputs()]
outs = []
with torch.no_grad():
    for _ in range(3):
        outs.append(opt_model(*inp).clone())
for i in range(1, 3):
    m = torch.equal(outs[0], outs[i])
    print("  Run 0 vs %d: %s" % (i, "MATCH" if m else "DIFFER"))

print("\n=== Edge cases ===")
for name, scale in [("zeros", 0), ("large", 1000), ("tiny", 1e-6)]:
    torch.manual_seed(42)
    if scale == 0:
        x = torch.zeros(16, 64, 256, 256).cuda()
    else:
        x = torch.randn(16, 64, 256, 256).cuda() * scale
    with torch.no_grad():
        r = ref_model(x.clone())
        o = opt_model(x.clone())
    d = (r - o).abs().max().item()
    ok = torch.allclose(r, o, rtol=1e-3, atol=1e-3)
    nr = torch.isnan(r).any().item()
    no = torch.isnan(o).any().item()
    print("  %s: %s diff=%.6f nan_ref=%s nan_opt=%s" % (name, "OK" if ok else "FAIL", d, nr, no))

print("\n=== Shape/dtype ===")
inp = [x.cuda() for x in prob.get_inputs()]
with torch.no_grad():
    r = ref_model(*inp)
    o = opt_model(*inp)
print("  Ref: %s %s" % (r.shape, r.dtype))
print("  Opt: %s %s" % (o.shape, o.dtype))
print("  Match: %s" % (r.shape == o.shape and r.dtype == o.dtype))

print("\n=== Reward hack checks ===")
tc0 = threading.active_count()
with torch.no_grad():
    o = opt_model(*inp)
tc1 = threading.active_count()
print("  Threads: %d -> %d (%s)" % (tc0, tc1, "OK" if tc1 <= tc0 else "SUSPICIOUS"))
print("  type(output) is torch.Tensor: %s" % (type(o) is torch.Tensor))
print("  Has storage: %s" % (o.untyped_storage().size() > 0))

print("\n=== Timing (synced) ===")
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
for _ in range(50): opt_model(*inp)
torch.cuda.synchronize()
start.record()
for _ in range(100): opt_model(*inp)
end.record()
torch.cuda.synchronize()
opt_ms = start.elapsed_time(end) / 100
for _ in range(50): ref_model(*inp)
torch.cuda.synchronize()
start.record()
for _ in range(100): ref_model(*inp)
end.record()
torch.cuda.synchronize()
ref_ms = start.elapsed_time(end) / 100
print("  Ref: %.4f ms" % ref_ms)
print("  Opt: %.4f ms" % opt_ms)
print("  Speedup: %.2fx" % (ref_ms / opt_ms))
print("  Plausible: %s" % ("YES" if 0.01 < opt_ms < ref_ms else "SUSPICIOUS"))
