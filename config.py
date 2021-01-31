IntervalDim = 100
NoteOnDim = 128
NoteOffDim = 128

NoteOnOffset = IntervalDim
NoteOffOffset = IntervalDim + NoteOnOffset
EventDim = IntervalDim + NoteOnDim + NoteOffDim # 356
# EventDim = 9833

maxLength = Time = 100
EmbeddingDim = 512 
HeadDim = 32 
Heads = 16 
ContextDim = HeadDim * Heads # 512
Layers = 6 # how many decoders

batch_size = batchSize = 64

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