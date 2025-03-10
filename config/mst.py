name='mst_s0'
out_dir = 'out/' + name
log_dir = 'log'
seed = 0

batch_size = 12
block_size = 1024
gradient_accumulation_steps = 40
max_iters = 600000
lr_decay_iters = 600000

static_topo = False
sparsity = .90
topo_interval = 300
gamma = 1
grow_rate = 1
grow_abs_rate = .3
grow_margin = 20
drop_rate = 0
drop_abs_rate = .1
drop_margin = 100000
target_loss = 2.9
grow_only = True