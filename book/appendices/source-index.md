# 附录 C 源码入口索引

本索引按章节列出固定版本的主要实现。先确定仓库，再定位文件和符号；同名文件不表示可混用版本。

## 01 环境、张量与一次最小前向

- [minimind/model/model_minimind.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/model/model_minimind.py)：`MiniMindConfig`、`MiniMindForCausalLM.__init__`、`MiniMindForCausalLM.forward`。
- [minimind/eval_llm.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/eval_llm.py)：`init_model`、`main`。

## 02 从文字到 token

- [minimind/trainer/train_tokenizer.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/train_tokenizer.py)：`get_texts`、`train_tokenizer`、`eval_tokenizer`。
- [minimind/model/model_minimind.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/model/model_minimind.py)：`MiniMindConfig.__init__`。


## 03 搭起语言模型骨架

- [minimind/model/model_minimind.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/model/model_minimind.py)：`MiniMindConfig`、`RMSNorm.norm`、`RMSNorm.forward`、`FeedForward.forward`、`MiniMindBlock.forward`、`MiniMindModel.forward`、`MiniMindForCausalLM.__init__`、`MiniMindForCausalLM.forward`。

## 04 手算一次注意力

- [minimind/model/model_minimind.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/model/model_minimind.py)：`Attention.__init__`、`Attention.forward`、`repeat_kv`。

## 05 位置、缓存与逐字生成

- [minimind/model/model_minimind.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/model/model_minimind.py)：`precompute_freqs_cis`、`apply_rotary_pos_emb`、`MiniMindModel.forward`、`MiniMindForCausalLM.generate`。

## 06 一条训练样本的旅程

- [minimind/dataset/lm_dataset.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/dataset/lm_dataset.py)：`PretrainDataset.__getitem__`、`SFTDataset.__getitem__`、`SFTDataset.generate_labels`、`SFTDataset.create_chat_prompt`、`pre_processing_chat`、`post_processing_chat`。
- [minimind/model/model_minimind.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/model/model_minimind.py)：`MiniMindForCausalLM.forward`。

## 07 预训练循环与断点恢复

- [minimind/trainer/train_pretrain.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/train_pretrain.py)：`train_epoch`。
- [minimind/trainer/train_full_sft.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/train_full_sft.py)：`train_epoch`。
- [minimind/trainer/trainer_utils.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/trainer_utils.py)：`get_lr`、`lm_checkpoint`、`SkipBatchSampler.__iter__`、`init_model`、`init_distributed_mode`。

## 08 SFT、评估与训练排错

- [minimind/trainer/train_full_sft.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/train_full_sft.py)：`train_epoch`。
- [minimind/trainer/trainer_utils.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/trainer_utils.py)：`init_model`。
- [minimind/dataset/lm_dataset.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/dataset/lm_dataset.py)：`SFTDataset`。
- [minimind/model/model_minimind.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/model/model_minimind.py)：`MiniMindForCausalLM.forward`、`MiniMindForCausalLM.generate`。
- [minimind/eval_llm.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/eval_llm.py)：`main`。

## 09 MoE 的路由与负载

- [minimind/model/model_minimind.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/model/model_minimind.py)：`MOEFeedForward.forward`、`MiniMindModel.forward`、`MiniMindBlock.__init__`。

## 10 LoRA 与知识蒸馏

- [minimind/model/model_lora.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/model/model_lora.py)：`LoRA.forward`、`apply_lora`、`save_lora`、`merge_lora`。
- [minimind/trainer/train_lora.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/train_lora.py)：`train_epoch`。
- [minimind/trainer/train_distillation.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/train_distillation.py)：`distillation_loss`、`train_epoch`。

## 11 DPO 与偏好数据

- [minimind/trainer/train_dpo.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/train_dpo.py)：`logits_to_log_probs`、`dpo_loss`、`train_epoch`。
- [minimind/dataset/lm_dataset.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/dataset/lm_dataset.py)：`DPODataset.__getitem__`、`DPODataset.generate_loss_mask`。

## 12 从 PPO 概念到 GRPO

- [minimind/trainer/train_ppo.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/train_ppo.py)：`CriticModel.forward`、`ppo_train_epoch`。
- [minimind/trainer/train_grpo.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/train_grpo.py)：`grpo_train_epoch`、`calculate_rewards`。
- [minimind/trainer/rollout_engine.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/rollout_engine.py)：`compute_per_token_logps`、`TorchRolloutEngine.rollout`、`SGLangRolloutEngine.rollout`、`SGLangRolloutEngine.update_policy`。
- [minimind/dataset/lm_dataset.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/dataset/lm_dataset.py)：`RLAIFDataset.create_chat_prompt`。

## 13 Agent 多轮交互与奖励

- [minimind/trainer/train_agent.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/train_agent.py)：`parse_tool_calls`、`execute_tool`、`rollout_single`、`rollout_batch`、`calculate_rewards`、`validate_gt_in_text`、`rl_train_epoch`。
- [minimind/dataset/lm_dataset.py](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/dataset/lm_dataset.py)：`AgentRLDataset.parse_conversations`。

## 14 图像怎样进入语言模型

- [minimind-v/model/model_vlm.py](https://github.com/jingyaogong/minimind-v/blob/041d528e037a8a373c3c8aa7278a3ba3bd26b6dc/model/model_vlm.py)：`VLMConfig`、`MMVisionProjector`、`MiniMindVLM.forward`、`MiniMindVLM.count_vision_proj`、`MiniMindVLM.get_vision_model`、`MiniMindVLM.get_image_embeddings`。

## 15 视觉训练与评估

- [minimind-v/trainer/trainer_utils.py](https://github.com/jingyaogong/minimind-v/blob/041d528e037a8a373c3c8aa7278a3ba3bd26b6dc/trainer/trainer_utils.py)：`init_vlm_model`、`vlm_checkpoint`、`vlm_collate_fn`。
- [minimind-v/dataset/lm_dataset.py](https://github.com/jingyaogong/minimind-v/blob/041d528e037a8a373c3c8aa7278a3ba3bd26b6dc/dataset/lm_dataset.py)：`VLMDataset.__getitem__`、`VLMDataset.create_chat_prompt`、`VLMDataset.generate_labels`。
- [minimind-v/trainer/train_pretrain_vlm.py](https://github.com/jingyaogong/minimind-v/blob/041d528e037a8a373c3c8aa7278a3ba3bd26b6dc/trainer/train_pretrain_vlm.py)：`train_epoch`。
- [minimind-v/trainer/train_sft_vlm.py](https://github.com/jingyaogong/minimind-v/blob/041d528e037a8a373c3c8aa7278a3ba3bd26b6dc/trainer/train_sft_vlm.py)：`train_epoch`。
- [minimind-v/model/model_vlm.py](https://github.com/jingyaogong/minimind-v/blob/041d528e037a8a373c3c8aa7278a3ba3bd26b6dc/model/model_vlm.py)：`MiniMindVLM.forward`、`MiniMindVLM.get_image_embeddings`。


## 16 音频入门与特征注入

- [minimind-o/model/model_omni.py](https://github.com/jingyaogong/minimind-o/blob/b5d1bc11ece8de17404cbc034c3e09718d0fecdf/model/model_omni.py)：`SenseVoiceAudioProcessor.__call__`、`MiniMindOmni.load_sensevoice`、`MiniMindOmni.encode_audio_inputs`、`MiniMindOmni.inject_audio_features`。
- [minimind-o/dataset/omni_dataset.py](https://github.com/jingyaogong/minimind-o/blob/b5d1bc11ece8de17404cbc034c3e09718d0fecdf/dataset/omni_dataset.py)：`OmniDataset.process_audio`、`OmniDataset.load_audio_inputs`、`OmniDataset.augment_wav`。

## 17 Thinker、Bridge 与 Talker

- [minimind-o/model/model_omni.py](https://github.com/jingyaogong/minimind-o/blob/b5d1bc11ece8de17404cbc034c3e09718d0fecdf/model/model_omni.py)：`OmniConfig`、`TalkerModule`、`MiniMindOmni.__init__`、`MiniMindOmni.forward`。

## 18 八路码本与时间对齐

- [minimind-o/model/model_omni.py](https://github.com/jingyaogong/minimind-o/blob/b5d1bc11ece8de17404cbc034c3e09718d0fecdf/model/model_omni.py)：`TalkerHead.forward`、`TalkerEmbedding.forward`、`MiniMindOmni.generate`、`MiniMindOmni.stream_generate`。
- [minimind-o/dataset/omni_dataset.py](https://github.com/jingyaogong/minimind-o/blob/b5d1bc11ece8de17404cbc034c3e09718d0fecdf/dataset/omni_dataset.py)：`OmniDataset.__getitem__`。
- [minimind-o/eval_omni.py](https://github.com/jingyaogong/minimind-o/blob/b5d1bc11ece8de17404cbc034c3e09718d0fecdf/eval_omni.py)：`eval_sample`。

## 19 Omni 训练流水线

- [minimind-o/trainer/train_sft_omni.py](https://github.com/jingyaogong/minimind-o/blob/b5d1bc11ece8de17404cbc034c3e09718d0fecdf/trainer/train_sft_omni.py)：`omni_collate_fn`、`train_epoch`。
- [minimind-o/trainer/trainer_utils.py](https://github.com/jingyaogong/minimind-o/blob/b5d1bc11ece8de17404cbc034c3e09718d0fecdf/trainer/trainer_utils.py)：`init_omni_model`、`omni_checkpoint`。
- [minimind-o/dataset/omni_dataset.py](https://github.com/jingyaogong/minimind-o/blob/b5d1bc11ece8de17404cbc034c3e09718d0fecdf/dataset/omni_dataset.py)：`OmniDataset.__getitem__`、`OmniDataset.apply_scheduled_sampling`。

## 20 流式交互、打断与综合实战

- [minimind-o/model/model_omni.py](https://github.com/jingyaogong/minimind-o/blob/b5d1bc11ece8de17404cbc034c3e09718d0fecdf/model/model_omni.py)：`SileroVAD`、`RealtimeSession.push_chunk`、`RealtimeSession.get_audio`、`RealtimeSession.reset`。
- [minimind-o/webui/web_demo.py](https://github.com/jingyaogong/minimind-o/blob/b5d1bc11ece8de17404cbc034c3e09718d0fecdf/webui/web_demo.py)：`realtime`、`run_generate`、`prepare_turn`、`stream_pcm`、`pcm_bytes`、`_mimi_decode`。

