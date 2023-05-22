import os
import re

import torch
import torch.distributed as dist


def init_from_mpi():
    if 'MASTER_ADDR' not in os.environ:
        omp_uri = os.environ["OMPI_MCA_orte_hnp_uri"]
        target_ip = re.search(r'.*tcp://((\d{1,3}\.){3}\d{1,3})[:,].*', omp_uri)
        auto_addr = target_ip.group(1)
        os.environ['MASTER_ADDR'] = auto_addr
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    local_id = os.environ.get('OMPI_COMM_WORLD_RANK')
    ntasks = os.environ.get('OMPI_COMM_WORLD_SIZE')
    os.environ['WORLD_SIZE'] = ntasks
    os.environ['RANK'] = local_id
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)


def init_from_slurm():
    job_id = int(os.environ['SLURM_JOB_ID'])
    port = 24553 + job_id % 10000
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1, pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')
    os.environ['MASTER_PORT'] = str(port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
