IntervalDim = 100
VelocityDim = 32
NoteOnDim = 128
NoteOffDim = 128

VelocityOffset = IntervalDim
NoteOnOffset = IntervalDim + VelocityDim
NoteOffOffset = IntervalDim + VelocityDim + NoteOnDim
EventDim = IntervalDim + VelocityDim + NoteOnDim + NoteOffDim # 388
# EventDim = 10247

maxLength = Time = 500
EmbeddingDim = 512 #词嵌入维度
HeadDim = 32 
Heads = 16 
ContextDim = HeadDim * Heads # 512
Layers = 6 #多少个decoder

batch_size = batchSize = 1

class hparams(object):
    n_vocab=EventDim,
    n_ctx=ContextDim,
    n_embd=EmbeddingDim,
    n_head=Heads,
    n_layer=Layers,
    n_time=Time

# type = 'poetry'
# type = 'random_music'
type = 'music'

# mode = "pretrain"
mode = "finetune"