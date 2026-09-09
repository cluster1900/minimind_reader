"""Small real Omni backbone; no external pretrained encoder is loaded."""
def tiny_omni(**overrides):
    from model.model_omni import OmniConfig,MiniMindOmni
    kw=dict(hidden_size=32,num_hidden_layers=4,num_attention_heads=8,num_key_value_heads=4,
        talker_hidden_size=32,num_talker_hidden_layers=2,intermediate_size=128,max_position_embeddings=128,
        audio_hidden_size=512,image_hidden_size=16,image_token_len=4,flash_attn=False,dropout=0.)
    kw.update(overrides)
    return MiniMindOmni(OmniConfig(**kw),audio_encoder_path='',vision_model_path=None)
