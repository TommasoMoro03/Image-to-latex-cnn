import torch
import torch.nn as nn
import math


class PositionalEncoding2D(nn.Module):
    """
    Implements 2D positional encoding as described in the original Transformer paper,
    adapted for 2D feature maps from a CNN.
    Adds sinusoidal positional encodings to the feature map to inject spatial information.
    """

    def __init__(self, d_model, max_h=256, max_w=1024):
        """
        Args:
            d_model (int): The dimensionality of the feature vectors (channels of CNN output).
            max_h (int): Maximum expected height of the feature map.
            max_w (int): Maximum expected width of the feature map.
        """
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_h, max_w, d_model)

        # Positional encoding for height (dim 0)
        position_h = torch.arange(0, max_h).unsqueeze(1)
        div_term_h = torch.exp(torch.arange(0, d_model // 2, 2) * -(math.log(10000.0) / (d_model // 2)))
        pe_h = torch.zeros(max_h, d_model // 2)
        pe_h[:, 0::2] = torch.sin(position_h * div_term_h)
        pe_h[:, 1::2] = torch.cos(position_h * div_term_h)

        # Positional encoding for width (dim 1)
        position_w = torch.arange(0, max_w).unsqueeze(1)
        div_term_w = torch.exp(torch.arange(0, d_model // 2, 2) * -(math.log(10000.0) / (d_model // 2)))
        pe_w = torch.zeros(max_w, d_model // 2)
        pe_w[:, 0::2] = torch.sin(position_w * div_term_w)
        pe_w[:, 1::2] = torch.cos(position_w * div_term_w)

        # Combine 1D positional encodings for 2D.
        # This assumes d_model is even. If d_model is odd, you might need to adjust.
        # It's common to split d_model into two halves, one for H and one for W.
        pe_h = pe_h.unsqueeze(1).expand(-1, max_w, -1)  # [max_h, max_w, d_model/2]
        pe_w = pe_w.unsqueeze(0).expand(max_h, -1, -1)  # [max_h, max_w, d_model/2]

        pe = torch.cat((pe_h, pe_w), dim=-1)  # [max_h, max_w, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to input feature map.

        Args:
            x (torch.Tensor): Input feature map from CNN encoder.
                              Shape: (B, C, H, W), where C=d_model.
        Returns:
            torch.Tensor: Feature map with positional encoding added.
                          Shape: (B, C, H, W)
        """
        # Slice positional encoding to match input size
        h, w = x.shape[2], x.shape[3]
        # Transpose pe from (H, W, C) to (C, H, W) to match CNN output format
        x = x + self.pe[:h, :w, :].permute(2, 0, 1).unsqueeze(0)
        return x


class Attention(nn.Module):
    """
    Implements a general attention mechanism (e.g., Bahdanau-style or Luong-style).
    This version implements a simple additive (Bahdanau-style) attention,
    which is common in image-to-sequence models with RNNs.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        Args:
            encoder_dim (int): Dimensionality of encoder output features (e.g., d_model from CNN).
            decoder_dim (int): Dimensionality of decoder hidden state (e.g., LSTM hidden size).
            attention_dim (int): Dimensionality of the attention layer.
        """
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # Linear layer for encoder features
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # Linear layer for decoder hidden state
        self.full_att = nn.Linear(attention_dim, 1)  # Linear layer to compute attention scores
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Softmax over the sequence dimension

    def forward(self, encoder_out, decoder_hidden):
        """
        Args:
            encoder_out (torch.Tensor): Encoder output features.
                                        Shape: (B, num_pixels, encoder_dim)
                                        (after flattening 2D map to sequence)
            decoder_hidden (torch.Tensor): Decoder's current hidden state.
                                           Shape: (B, decoder_dim)
        Returns:
            context (torch.Tensor): Context vector. Shape: (B, encoder_dim)
            alpha (torch.Tensor): Attention weights. Shape: (B, num_pixels)
        """
        # Align dimensions for attention
        # decoder_hidden needs to be unsqueezed for broadcasting: (B, 1, decoder_dim)
        att1 = self.encoder_att(encoder_out)  # (B, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # (B, 1, attention_dim)

        # Compute attention scores
        # (B, num_pixels, attention_dim) + (B, 1, attention_dim) -> (B, num_pixels, attention_dim)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # (B, num_pixels)

        alpha = self.softmax(att)  # (B, num_pixels) - attention weights

        # Compute context vector
        # (B, 1, num_pixels) * (B, num_pixels, encoder_dim) -> (B, 1, encoder_dim) -> (B, encoder_dim)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (B, encoder_dim)

        return context, alpha


# --- Test the modules (Optional, for development/debugging) ---
if __name__ == "__main__":
    print("Testing common_layers.py modules...")

    # Test PositionalEncoding2D
    d_model_pe = 512  # Example channel dim
    h_feat, w_feat = 16, 64  # Example feature map dimensions (e.g., 128/8, 512/8)
    pe_layer = PositionalEncoding2D(d_model_pe, max_h=h_feat, max_w=w_feat)

    dummy_input_feat_map = torch.randn(2, d_model_pe, h_feat, w_feat)  # Batch size 2
    output_feat_map = pe_layer(dummy_input_feat_map)
    print(f"PositionalEncoding2D input shape: {dummy_input_feat_map.shape}")
    print(f"PositionalEncoding2D output shape: {output_feat_map.shape}")
    assert output_feat_map.shape == dummy_input_feat_map.shape
    print("PositionalEncoding2D test passed!")

    # Test Attention
    encoder_dim_att = 512  # Matches d_model_pe
    decoder_dim_att = 256  # Example LSTM hidden size
    attention_dim_att = 128  # Attention layer size

    num_pixels = h_feat * w_feat  # Number of feature vectors from encoder
    dummy_encoder_out = torch.randn(2, num_pixels, encoder_dim_att)  # (B, Num_pixels, C)
    dummy_decoder_hidden = torch.randn(2, decoder_dim_att)  # (B, Decoder_dim)

    att_layer = Attention(encoder_dim_att, decoder_dim_att, attention_dim_att)
    context_vector, attention_weights = att_layer(dummy_encoder_out, dummy_decoder_hidden)

    print(f"\nAttention encoder_out shape: {dummy_encoder_out.shape}")
    print(f"Attention decoder_hidden shape: {dummy_decoder_hidden.shape}")
    print(f"Attention context_vector shape: {context_vector.shape}")
    print(f"Attention attention_weights shape: {attention_weights.shape}")

    assert context_vector.shape == (2, encoder_dim_att)
    assert attention_weights.shape == (2, num_pixels)
    print("Attention test passed!")