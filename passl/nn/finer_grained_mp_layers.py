# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import time
from paddle.autograd import PyLayer
from paddle.fluid import core
from paddle.nn import functional as F

import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.base import topology as tp
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker

__all__ = [
    "finer_grained_rowsharded_linear",
    "finer_grained_columnsharded_linear",
    "finer_grained_row_parallel_linear",
    "finer_grained_column_parallel_linear",
    "FinerGrainedRowParallelLinear",
    "FinerGrainedColumnParallelLinear"
]

def get_finer_grained_model_parallel_communication_info():
    hcg = fleet.get_hybrid_communicate_group()
    mp_rank = hcg.get_model_parallel_rank()
    mp_ranks = hcg.get_model_parallel_world_size()
    assert hasattr(hcg, '_mp_ring_comm_group'), "hcg must have _mp_ring_comm_group, you need to initialize model parallel ring group first"
    mp_ring_comm_group = hcg.get_model_parallel_ring_group()
    mp_group = hcg.get_model_parallel_group()

    next_mp_rank = (mp_rank + 1) % len(mp_group.ranks)
    prev_mp_rank = (mp_rank - 1 + len(mp_group.ranks)) % len(mp_group.ranks)
    send_group = mp_ring_comm_group[f'mp_{mp_rank}to{next_mp_rank}']
    recv_group = mp_ring_comm_group[f'mp_{prev_mp_rank}to{mp_rank}']
    send_dst = mp_group.ranks[next_mp_rank]
    recv_src = mp_group.ranks[prev_mp_rank]

    return mp_rank, mp_ranks, mp_group, send_group, recv_group, send_dst, recv_src


def finer_grained_sharded_row_col_col_linear(x, weight, bias=None, transpose_y=False, name=None):
    """
    y = x * weight + b = matmul(x, weight) + b
    """
 
    mp_rank, mp_ranks, mp_group, send_group, recv_group, send_dst, recv_src = \
        get_finer_grained_model_parallel_communication_info()
 
 
    hidden_size = x.shape[-1]
    assert hidden_size % mp_ranks == 0, f"hidden_size {hidden_size} must be divided by mp_ranks {mp_ranks}"
    micro_hidden_size = hidden_size // mp_ranks
    # reverse order [mp_ranks-1, ..., 1, 0]
    cal_index = list(range(mp_ranks-1, -1, -1))
    # shift
    shift = mp_ranks - mp_rank - 1
    cal_index = cal_index[shift:] + cal_index[:shift]
 
    x_recvs = []
    for idx in range(mp_ranks-1):
       x_recvs.append(paddle.empty(x.shape, dtype=x.dtype))
 
    xi = x
    y = []
 
    for idx, t in enumerate(cal_index):
        # launch async send and recv
        if idx < mp_ranks-1:
            if mp_rank % 2 == 0:
                task_send = dist.isend(xi, dst=send_dst, group=send_group)
            else:
                task_recv = dist.irecv(x_recvs[idx], src=recv_src, group=recv_group)
 
            if mp_rank % 2 == 0:
                task_recv = dist.irecv(x_recvs[idx], src=recv_src, group=recv_group)
            else:
                task_send = dist.isend(xi, dst=send_dst, group=send_group)
 
        yi = paddle.matmul(xi, weight, transpose_y=transpose_y)
            
        y.append(yi)
        # we need to sync and get received xi
        if idx < mp_ranks-1:
            task_send.wait()
            task_recv.wait()
            xi = x_recvs[idx]
    # shift results
    shift = mp_rank + 1
    y = y[shift:] + y[:shift]
    y = y[::-1]
    y = paddle.concat(y, axis=-2)
 
    if bias is not None:
        y = y + bias
 
    return y
 
def finer_grained_sharded_col_row_row_linear(x, weight, bias=None, transpose_y=False, name=None):
    """
    y = x * weight + b = matmul(x, weight) + b
    """
 
    mp_rank, mp_ranks, mp_group, send_group, recv_group, send_dst, recv_src = \
        get_finer_grained_model_parallel_communication_info()
 
    size = x.shape[-2]
    assert size % mp_ranks == 0, f"size {size} must be divided by mp_ranks {mp_ranks}"
    micro_size = size // mp_ranks
    # reverse order [mp_ranks-1, ..., 1, 0]
    cal_index = list(range(mp_ranks-1, -1, -1))
    # shift
    shift = mp_ranks - mp_rank
    cal_index = cal_index[shift:] + cal_index[:shift]
 
    y = None
 
    for idx, t in enumerate(cal_index):
        start = t * micro_size
        end = start + micro_size
        # slice and calculate matmul
        xi = paddle.slice(x, axes=[-2], starts=[start], ends=[end])
        yi = paddle.matmul(xi, weight)
        # we need to sync and get received
        if idx > 0:
            task_send.wait()
            task_recv.wait()
            yi_send = yi + yi_recv
        else:
            yi_send = yi
        # launch async send and recv
        if idx < mp_ranks-1:
            if mp_rank % 2 == 0:
                task_send = dist.isend(yi_send, dst=send_dst, group=send_group)
            else:
                yi_recv = paddle.empty_like(yi)
                task_recv = dist.irecv(yi_recv, src=recv_src, group=recv_group)
 
            if mp_rank % 2 == 0:
                yi_recv = paddle.empty_like(yi)
                task_recv = dist.irecv(yi_recv, src=recv_src, group=recv_group)
            else:
                task_send = dist.isend(yi_send, dst=send_dst, group=send_group)
 
    y = yi_send
    if bias is not None:
        y = y + bias
 
    return y

def finer_grained_sharded_row_row_row_linear(x, weight, bias=None, transpose_y=False, name=None):
    """
    y = x * weight + b = matmul(x, weight) + b
    """

    mp_rank, mp_ranks, mp_group, send_group, recv_group, send_dst, recv_src = \
        get_finer_grained_model_parallel_communication_info()

    hidden_size = x.shape[-1]
    assert hidden_size % mp_ranks == 0, f"hidden_size {hidden_size} must be divided by mp_ranks {mp_ranks}"
    micro_hidden_size = hidden_size // mp_ranks

    # reverse order [mp_ranks-1, ..., 1, 0]
    cal_index = list(range(mp_ranks-1, -1, -1))
    # shift
    shift = mp_ranks - mp_rank - 1
    cal_index = cal_index[shift:] + cal_index[:shift]

    wi = weight
    y = None

    for idx, t in enumerate(cal_index):
        start = t * micro_hidden_size
        end = start + micro_hidden_size

        # launch async send and recv
        if idx < mp_ranks-1:
            if mp_rank % 2 == 0:
                task_send = dist.isend(wi, dst=send_dst, group=send_group)
            else:
                w_recv = paddle.zeros_like(wi)
                task_recv = dist.irecv(w_recv, src=recv_src, group=recv_group)

            if mp_rank % 2 == 0:
                w_recv = paddle.zeros_like(wi)
                task_recv = dist.irecv(w_recv, src=recv_src, group=recv_group)
            else:
                task_send = dist.isend(wi, dst=send_dst, group=send_group)

        # slice and calculate matmul
        xi = paddle.slice(x, axes=[-1], starts=[start], ends=[end])
        yi = paddle.matmul(xi, wi, transpose_y=transpose_y)

        # sum
        if idx == 0:
            y = yi
        else:
            y = y + yi

        # we need to sync and get received xi
        if idx < mp_ranks-1:
            task_send.wait()
            task_recv.wait()
            wi = w_recv

    if bias is not None:
        y = y + bias

    return y

def finer_grained_sharded_row_col_row_linear(x, weight, bias=None, transpose_y=False, name=None):
    """
    y = x * weight + b = matmul(x, weight) + b
    """

    mp_rank, mp_ranks, mp_group, send_group, recv_group, send_dst, recv_src = \
        get_finer_grained_model_parallel_communication_info()

    wi = weight
    y = []

    for idx in range(mp_ranks):
        # launch async send and recv
        if idx < mp_ranks-1:
            if mp_rank % 2 == 0:
                task_send = dist.isend(wi, dst=send_dst, group=send_group)
            else:
                w_recv = paddle.zeros_like(wi)
                task_recv = dist.irecv(w_recv, src=recv_src, group=recv_group)

            if mp_rank % 2 == 0:
                w_recv = paddle.zeros_like(wi)
                task_recv = dist.irecv(w_recv, src=recv_src, group=recv_group)
            else:
                task_send = dist.isend(wi, dst=send_dst, group=send_group)

        # slice and calculate matmul
        yi = paddle.matmul(x, wi, transpose_y=transpose_y)

        y.append(yi)

        # we need to sync and get received wi
        if idx < mp_ranks-1:
            task_send.wait()
            task_recv.wait()
            wi = w_recv

    # shift results
    shift = mp_rank + 1
    y = y[shift:] + y[:shift]
    y = y[::-1]
    y = paddle.concat(y, axis=-1)

    if bias is not None:
        y = y + bias

    return y

