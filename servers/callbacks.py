import torch

def empty_cach(memory_record_dic):
    if memory_record_dic is not None:
        torch.cuda.empty_cache()
        
def log_memory(memory_record_dic, device):

    if memory_record_dic is not None:
        memory_record_dic[device.index] = {}
        memory_record_dic[device.index]['max_memory_allocated'] = torch.cuda.max_memory_allocated(device)
        memory_record_dic[device.index]['max_memory_reserved'] = torch.cuda.max_memory_reserved(device)

