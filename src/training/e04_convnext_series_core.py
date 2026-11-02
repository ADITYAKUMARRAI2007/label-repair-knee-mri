"""Owned ConvNeXt-Small series reader, with strict generic pretrained loading."""

class SeriesMIL(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.encoder = timm.create_model("convnext_small", pretrained=False,
                                         num_classes=1000, drop_path_rate=0.1)
        state = load_file(str(checkpoint), device="cpu")
        self.encoder.load_state_dict(state, strict=True)
        self.encoder.reset_classifier(0)
        self.encoder.set_grad_checkpointing(True)
        dim = 256
        self.project = nn.Sequential(nn.Dropout(0.2), nn.Linear(768, dim), nn.LayerNorm(dim))
        self.slot_embedding = nn.Embedding(5, dim)
        self.position_embedding = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, 8, 512, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.attention = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Dropout(0.2),
                                       nn.Linear(128, len(TARGETS)))
        self.classifier_weight = nn.Parameter(torch.randn(len(TARGETS), dim) * 0.02)
        self.classifier_bias = nn.Parameter(torch.zeros(len(TARGETS)))

    def forward(self, x, mask, pos, training=False):
        b, slots, k = x.shape[:3]
        if slots != 5 or not mask.any(1).all():
            raise RuntimeError("invalid acquisition mask")
        keep = mask.reshape(-1)
        windows = x.reshape(b * slots, k, *x.shape[3:])[keep].flatten(0, 1)
        views = make_views(windows, IMAGE_SIZE, training, k)
        encoded = self.encoder(views.contiguous(memory_format=torch.channels_last))
        if encoded.shape != (len(windows), 768):
            raise RuntimeError("encoder feature shape drift")
        feat = encoded.new_zeros(b * slots, k, 768)
        feat[keep] = encoded.reshape(-1, k, 768)
        feat = feat.reshape(b, slots, k, 768)
        slot_ids = torch.arange(slots, device=x.device).view(1, slots, 1)
        h = (self.project(feat) + self.slot_embedding(slot_ids)
             + self.position_embedding(pos.unsqueeze(-1).to(feat.dtype)))
        h = h.reshape(b, slots * k, -1)
        pad = ~mask.unsqueeze(-1).expand(b, slots, k).reshape(b, slots * k)
        h = self.transformer(h, src_key_padding_mask=pad)
        a = self.attention(h).float().masked_fill(pad.unsqueeze(-1), float("-inf")).softmax(1)
        pooled = torch.einsum("btc,btd->bcd", a.to(h.dtype), h)
        return (pooled * self.classifier_weight).sum(-1) + self.classifier_bias
