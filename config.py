"""
Create the model configuration (`config.json`) and the feature extractor configuration
(`preprocessor_config.json`).
"""
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor

model_dir="./"

config = Wav2Vec2Config.from_pretrained(
    "facebook/wav2vec2-base", 
    mask_time_length=10,
    mask_time_prob=0.05,
    diversity_loss_weight=0.1,
    num_negatives=100,
    do_stable_layer_norm=True,
    feat_extract_norm="layer",
)
config.save_pretrained(model_dir)

config = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base", return_attention_mask=True)
config.save_pretrained(model_dir)
