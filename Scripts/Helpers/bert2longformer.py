import copy
import torch
from transformers import LongformerSelfAttention

class BertLongSelfAttention(LongformerSelfAttention):
    """
    from https://github.com/allenai/longformer/issues/215
    For transformers=4.12.5
    For transformers=4.26

    From XLMRobertaSelfAttention

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,

    to

    LongformerSelfAttention

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        [`LongformerSelfAttention`] expects *len(hidden_states)* to be multiple of *attention_window*. Padding to
        *attention_window* happens in [`LongformerModel.forward`] to avoid redoing the padding on each layer.

        The *attention_mask* is changed in [`LongformerModel.forward`] from 0, 1, 2 to:

            - -10000: no attention
            - 0: local attention
            - +10000: global attention
    """

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        # is_global_attn = any(is_index_global_attn.flatten()) PR #5811
        is_global_attn = is_index_global_attn.flatten().any().item()
        return super().forward(
            hidden_states,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

def convert_bert_to_longformer(
    bert_model,
    longformer_max_length: int = 4096,
    attention_window: int = 512,
):
    config = bert_model.config
    # extend position embeddings
    (
        current_max_pos,
        embed_size,
    ) = bert_model.bert.embeddings.position_embeddings.weight.shape

    config.max_position_embeddings = longformer_max_length + 2
    # allocate a larger position embedding matrix

    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(bert_model.bert.encoder.layer):
        longformer_self_attn = BertLongSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn

    new_pos_embed = bert_model.bert.embeddings.position_embeddings.weight.new_empty(config.max_position_embeddings, config.hidden_size)
    
    new_pos_embed[0, :] = bert_model.bert.embeddings.position_embeddings.weight[0]
    
    k = 1
    step = current_max_pos - 2
    while k < longformer_max_length + 1 - 1:
        if new_pos_embed[k : (k + step)].shape[0] != step:
            step = longformer_max_length - k -1
        if step >0:
            new_pos_embed[k : (k + step)] = bert_model.bert.embeddings.position_embeddings.weight[1 : (1+step)]
        if step == 0:
            break
        k += step

    bert_model.bert.embeddings.position_embeddings = torch.nn.Embedding.from_pretrained(new_pos_embed,freeze=False)
    bert_model.bert.embeddings.position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))

    #with TemporaryDirectory() as temp_dir:
    #    bert_model.save_pretrained(temp_dir)
    #    longformer_model = LongformerForSequenceClassification.from_pretrained(temp_dir,num_labels = 50)

    return bert_model  # , new_pos_embed