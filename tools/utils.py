import torch
import os
import re
import shutil
import psutil

def get_max_memory(config): 
    if config.device == 'cuda':
        max_memory = torch.cuda.max_memory_allocated(device= config.device)
        return max_memory
    else:
        process = psutil.Process()
        max_memory = process.memory_info().rss  # 获取当前进程的实际内存使用量，单位字节
        return max_memory 


def get_module_memory(module):
    for param in module.parameters():
        print(f'Parameter element size: {param.element_size()}')
        break
    mem = sum(param.nelement() * param.element_size() for param in module.parameters())
    return mem


def  initialize_distributed(config, args):
    print("Initializing process group...")
    torch.distributed.init_process_group(
        backend=config.distributed_backend,
        init_method=config.init_method,
        world_size=args.world,
        rank=args.rank,
    )
    print("Initialization of process group complete!")
    

def get_rank():
    return torch.distributed.get_rank()


def get_world_size():
    return torch.distributed.get_world_size()


def get_model_type(config_name):
    if 'vicuna_7b' in config_name:
        return 'vicuna_7b'
    else:
        raise NotImplementedError


def save_state_dict(state_dict, save_path):
    # Ensure the directory exists or create it
    print("Saving model to {}".format(save_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Remove contents if directory already exists
    if os.path.exists(save_path):
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        else:
            os.remove(save_path)

    # Save the state_dict to the specified path
    torch.save(state_dict, save_path)


def get_medusa_zephyr_model_state_dict(base_model_path):
    # for zephyr and vicuna 13b, weight only in base_model_path
    pretrained_dict =  {}
    print( base_model_path)
    all_files = os.listdir( base_model_path)
    weight_file_list =  [os.path.join(   base_model_path, f) for f in all_files if f.endswith('.bin')]
    for weigt_file in weight_file_list:
        pretrained_dict1 = torch.load(weigt_file)
        pretrained_dict = {**pretrained_dict, **pretrained_dict1}
    return  pretrained_dict


def get_medusa_model_state_dict(base_model_path,medusa_head_path):
    # for vicuna, with weight in both base_model_path and medusa_head_path
    pretrained_dict =  {}
    all_files = os.listdir( base_model_path)
    weight_file_list =  [os.path.join(   base_model_path, f) for f in all_files if f.endswith('.bin')]
    for weigt_file in weight_file_list:
        pretrained_dict1 = torch.load(weigt_file)
        pretrained_dict = {**pretrained_dict, **pretrained_dict1}
    medusa_head_path = os.path.join( medusa_head_path, "medusa_lm_head.pt")
    medusa_head_state_dict = torch.load(medusa_head_path ) # key  0.0.linear.weight
    # medusa_head_state_dict 修改key
    new_medusa_head_state_dict = {}
    for k,v in medusa_head_state_dict.items():
        new_name = "medusa_head." + k
        new_medusa_head_state_dict[new_name] = v
    all_state_dict = {**pretrained_dict, **new_medusa_head_state_dict}
    return  all_state_dict


def get_layer_dicts(all_state_dict,total_layer_num):
    
    not_layer_dict = {k: v for k, v in all_state_dict.items() if  "model.layers" not in k}   
    print("len(not_layer_dict)", len(not_layer_dict))
    layer_dicts = []# 每一层的layer的权重
    print("total_layer_num:", total_layer_num)
    for i in range(total_layer_num):
        layer_name = f'model.layers.{i}'
        layer_dict = {k: v for k, v in all_state_dict.items() if ".".join( k.split(".")[:3]) == layer_name} #TODO:这里可能不同模型的key不一样
        layer_dicts.append(layer_dict)
    print("len(layer_dicts)", len(layer_dicts))
    print("len(layer_dicts[0])", len(layer_dicts[0]))
    return not_layer_dict,layer_dicts


def get_stage_state_dict( base_model_path,
                         medusa_head_path,
                         stage_num_hidden_layers_list,
                         rank):
    if   'vicuna-7b'  in base_model_path:
        print("base_model_path", base_model_path)
        all_state_dict = get_medusa_model_state_dict(base_model_path,medusa_head_path)
    elif  'vicuna-13b' in base_model_path:
        all_state_dict = get_medusa_zephyr_model_state_dict(base_model_path)
    elif 'zephyr' in base_model_path:
        all_state_dict = get_medusa_zephyr_model_state_dict(base_model_path)
    else:
        raise NotImplementedError("暂不支持该模型")
    not_layer_dict,layer_dicts = get_layer_dicts(all_state_dict, sum(stage_num_hidden_layers_list))
    print("len(not_layer_dict)", len(not_layer_dict))
    print("len(layer_dicts)", len(layer_dicts))
    stage_state_dict = not_layer_dict
    if rank == 0: # 序号是0开始
        left= 0
        right = stage_num_hidden_layers_list[0]
        print( "left:", left, "right:", right)
        for i in range( left, right):
            print("i:", i,end=" ")
            stage_state_dict.update(layer_dicts[i])
        print("len(stage_state_dict)", len(stage_state_dict))
        return stage_state_dict
    else:
        print(  "stage_layer_num_list", stage_num_hidden_layers_list)
        print("rank", rank)
        left = sum(stage_num_hidden_layers_list[:rank ])
        right = sum(stage_num_hidden_layers_list[ :rank+1]  )
        print( "left:", left, "right:", right)
        # 取出对应layer范围的权重
        for i in range( left, right):
            print("i:", i,end=" ")
            stage_state_dict.update(layer_dicts[i])
        print("len(stage_state_dict)", len(stage_state_dict))
        # 修改layer编号
        new_dict = {}
        for k,v in stage_state_dict.items():
            match = re.search(r'layers\.(\d+)\.', k) # 原本权重layer.x.
            if match==None: # 不是layer.x的权重，直接拷贝
                new_dict[k] = v
            else:
                old_index = int(match.group(1))
                new_index = old_index - left
                # print(  old_index,  "->", new_index)
                new_name = re.sub(r'layers\.(\d+)\.', f'layers.{new_index}.', k) # layer.old. -> layer.new. 修改layer编号
                new_dict[new_name] = v
        return new_dict