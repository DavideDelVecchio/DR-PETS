import torch, platform
def get_device(prefer="auto"):
    prefer=prefer.lower()
    if prefer=="auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available() and platform.system()=="Darwin":
            return torch.device("mps")
        return torch.device("cpu")
    if prefer=="cuda":
        assert torch.cuda.is_available(),"CUDA unavailable"
        return torch.device("cuda")
    if prefer=="mps":
        assert torch.backends.mps.is_available(),"MPS unavailable"
        return torch.device("mps")
    return torch.device("cpu")
