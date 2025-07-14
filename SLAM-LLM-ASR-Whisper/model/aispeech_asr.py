import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import logging
import types
from typing import List, Optional, Tuple, Union

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from utils.metric import compute_accuracy
from utils.config_utils import generate_peft_config
from utils.model_utils import print_model_size, print_module_size
from utils.npu_flash_attn import patch_npu_flash_attn

logger = logging.getLogger(__name__)
def extract_variable_length_features(self, x: torch.Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        # x = (x + self.positional_embedding).to(x.dtype)
        x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x

def setup_tokenizer(train_config, model_config, **kwargs):
    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(model_config.llm_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_encoder(train_config, model_config, **kwargs):
    encoder_name = model_config.encoder_name
    import whisper
    encoder = whisper.load_model(name=model_config.encoder_path, device='cpu').encoder
    encoder.extract_variable_length_features = types.MethodType(extract_variable_length_features, encoder)

    if train_config.freeze_encoder:
        for name, param in encoder.named_parameters(): 
            param.requires_grad = False
        encoder.eval()
        print_module_size(encoder, encoder_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    return encoder


def setup_encoder_projector(train_config, model_config, **kwargs):
    if model_config.encoder_projector == "linear":
        from model.projector import EncoderProjectorConcat
        encoder_projector = EncoderProjectorConcat(model_config)
    elif model_config.encoder_projector == "cov1d-linear":
        from model.projector import EncoderProjectorCov1d
        encoder_projector = EncoderProjectorCov1d(model_config)
    elif model_config.encoder_projector == "q-former":
        from model.projector import EncoderProjectorQFormer
        encoder_projector = EncoderProjectorQFormer(model_config)
    else:
         raise ValueError(f"Unknown projector type: {model_config.encoder_projector}")
    print_module_size(encoder_projector, model_config.encoder_projector, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    return encoder_projector


def setup_llm(train_config, model_config, **kwargs):
    use_cache = False if train_config.enable_fsdp or train_config.enable_ddp else None

    model = AutoModelForCausalLM.from_pretrained(
                model_config.llm_path,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
            )

    print_module_size(model, model_config.llm_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    if train_config.freeze_llm: # TODO:to test offical `freeze_layers` and `num_freeze_layers`
        for name, param in model.named_parameters(): 
            param.requires_grad = False
        model.eval()
        
    if kwargs.get("peft_ckpt", None): # (FIX:MZY):reload will get wrong results when decoding
        logger.info("loading peft_ckpt from: {}".format(kwargs.get("peft_ckpt")))
        model = PeftModel.from_pretrained(model=model, model_id=kwargs.get("peft_ckpt"), is_trainable=True)
        model.print_trainable_parameters()
    elif train_config.use_peft:
        logger.info("setup peft...")
        peft_config = generate_peft_config(train_config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    print_module_size(model, model_config.llm_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    return model


def model_factory(train_config, model_config, **kwargs):
    # return necessary components for training
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)
    DEFAULT_SPEECH_TOKEN = "<speech>"
    DEFAULT_IGNORE_TOKEN = -100
    special_tokens_dict = {"additional_special_tokens": [DEFAULT_SPEECH_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.default_ignore_token = DEFAULT_IGNORE_TOKEN
    tokenizer.default_speech_token = tokenizer.convert_tokens_to_ids(
            DEFAULT_SPEECH_TOKEN
        )
    
    # llm
    llm = setup_llm(train_config, model_config, **kwargs)

    # encoder
    encoder = setup_encoder(train_config, model_config, **kwargs)
    
    # projector
    encoder_projector = setup_encoder_projector(
        train_config, model_config, **kwargs
    )

    model = slam_model_asr(
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    )
    patch_npu_flash_attn()
    ckpt_path = kwargs.get( "ckpt_path", None)
    if ckpt_path is not None:
        logger.info("loading other parts from: {}".format(ckpt_path))
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)
        

    print_model_size(
        model,
        train_config,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )
    return model, tokenizer


class slam_model_asr(torch.nn.Module):
    def __init__(
        self,
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    ):
        super().__init__()
        # modality encoder 
        self.encoder = encoder

        # llm
        self.llm = llm

        # projector
        self.encoder_projector = encoder_projector

        # tokenizer
        self.tokenizer = tokenizer
        self.metric = kwargs.get("metric", "acc")

        self.train_config = train_config
        self.model_config = model_config

        if train_config.get("enable_deepspeed", False):
            def new_forward(self, input):
                output = F.layer_norm(
                    input.float(),
                    self.normalized_shape,
                    self.weight.float() if self.weight is not None else None,
                    self.bias.float() if self.bias is not None else None,
                    self.eps,
                )
                return output.type_as(input)
            for item in self.modules():
                if isinstance(item, nn.LayerNorm):
                    item.forward = types.MethodType(new_forward, item)

    def forward(self,
                input_ids: torch.LongTensor = None,
                input_features: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                input_feature_length : Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ):

        encoder_outs = self.encoder(input_features) # bs*seq*dim
        projector_outs = self.encoder_projector(encoder_outs)
        encoder_feature_length = input_feature_length // 2
        projector_feature_length = encoder_feature_length // self.encoder_projector.k
        # print("\n","End",inputs_embeds, attention_mask, labels, position_ids,encoder_feature_length,projector_feature_length)
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds, attention_mask, labels, position_ids, _ = self._merge_input_ids_with_audio_features(
                projector_outs, projector_feature_length, inputs_embeds, input_ids, attention_mask, labels
            )
        # print("\n","Before", projector_outs.shape, input_feature_length,encoder_feature_length,projector_feature_length)
        # exit(1)
        model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels,position_ids=position_ids)
        acc = -1
        if self.metric:
            with torch.no_grad():
                preds = torch.argmax(model_outputs.logits, -1)
                acc = compute_accuracy(preds.detach()[:, :-1], labels.detach()[:, 1:], ignore_label=self.tokenizer.default_ignore_token)

        return model_outputs, acc
    
    @torch.no_grad()
    def generate(self,
                input_ids: torch.LongTensor = None,
                input_features: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                input_feature_length : Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs
                ):
        

        encoder_outs = self.encoder(input_features) # bs*seq*dim
        projector_outs = self.encoder_projector(encoder_outs)
        encoder_feature_length = input_feature_length // 2
        projector_feature_length = encoder_feature_length // self.encoder_projector.k
        # print("\n","End",inputs_embeds, attention_mask, labels, position_ids,encoder_feature_length,projector_feature_length)
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds, attention_mask, labels, position_ids, _ = self._merge_input_ids_with_audio_features(
                projector_outs, projector_feature_length, inputs_embeds, input_ids, attention_mask, labels
            )
        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=kwargs.get("max_new_tokens", 200),
            num_beams=kwargs.get("num_beams", 4),
            do_sample=kwargs.get("do_sample", False),
            min_length=kwargs.get("min_length", 1),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            attention_mask=attention_mask,
            # position_ids=position_ids,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        return model_outputs
    def _merge_input_ids_with_audio_features(
        self, audio_features, num_audio_tokens, inputs_embeds, input_ids, attention_mask, labels
    ):
        """
        Merge input_ids with with audio features into final embeddings
        
        Args:
            audio_features (`torch.Tensor` of shape `(num_audios, max_audio_tokens, embed_dim)`):
                All audio vectors of all audios in the batch
            num_audio_tokens (`torch.LongTensor` of shape `(num_audios)`):
                The length of audio embeddings of each audio as stacked in `audio_features`
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
                Token embeddings before merging with audio embeddings
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input_ids of tokens, possibly filled with audio token
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
                labels need to be recalculated to support training (if provided)
        Returns:
            final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids

        Explanation:
            each audio has variable length embeddings, with length specified by num_audio_tokens
            audio_features is concatenation of all audio embed vectors
            task: fill each <|AUDIO|> with the correct number of audio embeddings
            Example:
                X (5 tokens), Y (3 tokens), Z (8 tokens)
                X, Y are in the same sequence (in-context learning)
            if right padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    o p q r Z s t u v _ _ _ _ _ _
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
                ]
            elif left padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    _ _ _ _ _ _ o p q r Z s t u v
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
                ]
            Edge cases:
                * If tokens are same but audio token sizes are different, then cannot infer left or right padding
                ```python
                url1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
                audio1, _ = librosa.load(BytesIO(urlopen(url1).read()), sr=processor.feature_extractor.sampling_rate)
                url2 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"
                audio2, _ = librosa.load(BytesIO(urlopen(url2).read()), sr=processor.feature_extractor.sampling_rate)
                prompts = [
                    "[INST] <|AUDIO|>\nWhat is that in this audio? [/INST]",
                    "[INST] <|AUDIO|>\nWhat is that in this audio? [/INST]",
                ]
                inputs = processor(text=prompts, audios=[audio1, audio2], return_tensors='pt', padding=True).to("cuda")
                    audio1 has 101 tokens, while audio2 has 72 tokens
                ```

                input_ids: [
                    a b c d X g h
                    i j Y k l m n
                ]
                where X is 3 tokens while Y is 5, this mean after merge
                if left-padding (batched generation)
                    input_ids should be: [
                        _ _ a b c d X X X g h
                        i j Y Y Y Y Y k l m n
                    ]
                elif (right padding) (training)
                    input_ids should be: [
                        a b c d X X X g h _ _
                        i j Y Y Y Y Y k l m n
                    ]
        """
        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_features_mask = torch.arange(max_audio_tokens).expand(num_audios, max_audio_tokens).to(
            num_audio_tokens.device
        ) < num_audio_tokens.unsqueeze(1)
        masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)
        batch_size, sequence_length = input_ids.shape
        _left_padding = torch.any(attention_mask[:, 0] == 0)
        _right_padding = torch.any(attention_mask[:, -1] == 0)

        left_padding = True
        if batch_size > 1:
            if _left_padding and not _right_padding:
                left_padding = True
            elif not _left_padding and _right_padding:
                left_padding = False
            elif not _left_padding and not _right_padding:
                # both side is 1, so cannot tell
                left_padding = True
            else:
                # invalid attention_mask
                raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

        # 1. Create a mask to know where special audio tokens are
        special_audio_token_mask = input_ids == self.tokenizer.default_speech_token
        num_special_audio_tokens = torch.sum(special_audio_token_mask, dim=-1)

        # In case the Audio model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)
        num_audio_tokens = num_audio_tokens.to(target_device)
        batch_indices, non_audio_indices = torch.where(
            (input_ids != self.tokenizer.default_speech_token) & (attention_mask == 1)
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged audio-text sequence.
        # `special_audio_token_mask` identifies audio tokens. Each audio token will be replaced by `audio_feat_lengths - 1` text tokens.
        # `torch.cumsum` computes how each audio token shifts subsequent text token positions.
        token_placeholder_num = torch.zeros_like(input_ids)
        token_placeholder_num[special_audio_token_mask] = num_audio_tokens.long() - 1
        token_placeholder_num = token_placeholder_num + 1
        new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
        max_token_num = token_placeholder_num.sum(-1).max()
        nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_audio_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]
        batch_indices, non_audio_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_audio_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_token_num, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_token_num, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_input_ids = torch.full(
            (batch_size, max_token_num), self.tokenizer.pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
        )

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<audio>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the audio features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_audio_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_audio_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_audio_indices]
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            final_labels = torch.full((batch_size, max_token_num),self.tokenizer.default_ignore_token,dtype=input_ids.dtype, device=inputs_embeds.device).to(torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_audio_indices]
        # 5. Fill the embeddings corresponding to the audios. Anything that is still zeros needs filling
        audio_to_overwrite = torch.full(
            (batch_size, max_token_num), True, dtype=torch.bool, device=inputs_embeds.device
        )
        audio_to_overwrite[batch_indices, text_to_overwrite] = False
        seq_indices = torch.arange(max_token_num).unsqueeze(0).to(target_device)
        seq_indices = seq_indices.expand(batch_size, max_token_num)

        if left_padding:
            # exclude padding on the left
            max_token_num = max_token_num.to(target_device)
            val = (max_token_num - seq_indices) <= (
                token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1)
            )[:, None]
        else:
            # exclude padding on the right
            val = seq_indices < (token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1))[:, None]

        audio_to_overwrite &= val

        if audio_to_overwrite.sum() != num_audio_tokens.sum():
            raise ValueError(
                f"The input provided to the model are wrong. The number of audio tokens is {num_special_audio_tokens} while"
                f" the number of audio given to the model is {num_audios}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[audio_to_overwrite] = (
            masked_audio_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        final_attention_mask |= audio_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        return final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids