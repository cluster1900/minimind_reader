"""A real, randomly initialized tiny SigLIP. Only checkpoint loading is replaced."""
from contextlib import contextmanager
from unittest.mock import patch
from transformers import SiglipVisionConfig,SiglipVisionModel,SiglipImageProcessor
@contextmanager
def tiny_vision_loading():
    encoder=SiglipVisionModel(SiglipVisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,image_size=32,patch_size=16))
    processor=SiglipImageProcessor(size={'height':32,'width':32})
    with patch.object(SiglipVisionModel,'from_pretrained',return_value=encoder),patch.object(SiglipImageProcessor,'from_pretrained',return_value=processor):
        yield encoder,processor
