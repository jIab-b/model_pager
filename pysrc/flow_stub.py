from transformers import PretrainedConfig, PreTrainedModel

class FluxConfig(PretrainedConfig):
    model_type = "flux"
    num_frames: int = 16

class FluxModelStub(PreTrainedModel):
    config_class = FluxConfig
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *a, **kw):
        config = FluxConfig.from_pretrained(pretrained_model_name_or_path)
        return cls(config)

    def forward(self, *_, **__):
        raise RuntimeError("FluxModelStub never runs a forward pass")

class TrellisConfig(PretrainedConfig):
    model_type = "trellis"

class TrellisModelStub(PreTrainedModel):
    config_class = TrellisConfig
    def __init__(self, config): super().__init__(config)
