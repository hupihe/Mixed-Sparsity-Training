name='dense_s0'
out_dir = 'out/' + name
log_dir = 'log'
seed = 0

batch_size = 12
block_size = 1024
gradient_accumulation_steps = 40
max_iters = 600000
lr_decay_iters = 600000

static_topo = False
sparsity = 0
topo_interval = 100