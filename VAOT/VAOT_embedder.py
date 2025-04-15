from torch import nn
from VAOT.VAOT_models import BaseModel, ConvEmbedder

class AlignNet(nn.Module):
    def __init__(self, cfg):
        super(AlignNet, self).__init__()
        self.base_cnn = BaseModel(pretrained=True)
        self.emb = ConvEmbedder(emb_size=cfg.VAOT.EMBEDDING_SIZE, l2_normalize=cfg.VAOT.NORMALIZE_EMBDS)
        self.NUM_CONTEXT_STEPS = cfg.VAOT.NUM_CONTEXT_STEPS

    def forward(self, x):
        num_ctxt = self.NUM_CONTEXT_STEPS

        num_frames = x.size(1) // num_ctxt
        x = self.base_cnn(x)
        x = self.emb(x, num_frames)
        return x