import torch

def check_cuda_availability():
    # Check if PyTorch is available
    # if not torch.is_available():
    #     print("PyTorch is not available.")
    #     return

    print("PyTorch is available.")

    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    check_cuda_availability()
