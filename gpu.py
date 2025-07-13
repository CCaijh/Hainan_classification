import torch


def gpu_info() -> str:
    info = ''
    for id in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(id)
        print(p)
        info += f'CUDA:{id} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n'
    return info[:-1]


if __name__ == '__main__':
    print(gpu_info())