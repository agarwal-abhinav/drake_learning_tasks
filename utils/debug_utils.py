import gc, torch

def top_cuda_tensors(k=20):
    items = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                nbytes = obj.nelement() * obj.element_size()
                items.append((nbytes, type(obj), tuple(obj.shape), obj.dtype))
        except Exception:
            pass
    items.sort(reverse=True, key=lambda x: x[0])
    total = sum(x[0] for x in items)
    print(f"CUDA tensors alive: {len(items)}  total={total/1e9:.3f} GB")
    for nbytes, typ, shape, dtype in items[:k]:
        print(f"  {nbytes/1e6:9.1f} MB  {dtype}  {shape}  ({typ})")
    return items