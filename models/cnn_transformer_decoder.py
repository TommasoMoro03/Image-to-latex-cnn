# models/cnn_transformer_decoder.py

import torch
import torch.nn as nn
import torchvision.models as models
import math # For positional encoding
from models.common_layers import PositionalEncoding2D # Reuse our 2D PE

# --- Re-use the CNNEncoder from cnn_lstm_attention.py ---
# This ensures consistent feature extraction across different decoder types.
class CNNEncoder(nn.Module):
    """
    CNN Encoder that extracts features from the input image.
    Based on VGG-like architecture to achieve 8x downsampling as hinted by the reference paper.
    Includes 2D Positional Encoding.
    """
    def __init__(self, encoded_image_size=(20, 100), encoder_dim=512, dropout_prob=0.5):
        """
        Args:
            encoded_image_size (tuple): (height, width) of the feature map after CNN.
                                        For 160x800 input and 8x downsampling, this is 20x100.
            encoder_dim (int): Dimensionality of the extracted features (channels of CNN output).
            dropout_prob (float): Dropout probability for regularization.
        """
        super().__init__()
        self.encoder_dim = encoder_dim
        self.encoded_image_height = encoded_image_size[0]
        self.encoded_image_width = encoded_image_size[1]

        # --- A simpler VGG-like CNN for 8x downsampling ---
        # Input: 1x160x800
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 1x80x400 (2x downsample)

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 1x40x200 (4x downsample)

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 1x20x100 (8x downsample)

            # Final layer to ensure output channels match encoder_dim
            nn.Conv2d(256, encoder_dim, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        )
        
        self.positional_encoding = PositionalEncoding2D(
            d_model=self.encoder_dim,
            max_h=self.encoded_image_height,
            max_w=self.encoded_image_width
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, images):
        """
        Args:
            images (torch.Tensor): Input images. Shape: (B, 1, H_img, W_img)
        Returns:
            torch.Tensor: Encoder output features. 
                          Shape: (B, num_pixels, encoder_dim) (after flattening and PE)
        """
        out = self.conv_layers(images) # (B, encoder_dim, H_feat, W_feat)
        out = self.positional_encoding(out)

        # Reshape to (B, num_pixels, encoder_dim) for the Transformer Decoder's cross-attention
        out = out.permute(0, 2, 3, 1).contiguous() # (B, H_feat, W_feat, C)
        out = out.view(out.size(0), -1, self.encoder_dim) # (B, H_feat*W_feat, encoder_dim)
        
        return self.dropout(out)


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder that generates LaTeX token sequences.
    """
    def __init__(self, vocab_size, embed_dim, nhead, num_decoder_layers, 
                 dim_feedforward, dropout_prob, max_seq_len):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimensionality of word embeddings (also d_model for Transformer).
            nhead (int): Number of attention heads.
            num_decoder_layers (int): Number of stacked Transformer decoder layers.
            dim_feedforward (int): Dimension of the feedforward network model.
            dropout_prob (float): Dropout probability.
            max_seq_len (int): Maximum expected length of the LaTeX sequence.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim # This is also the d_model for Transformer layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Embedding(max_seq_len, embed_dim) # 1D PE for sequence
        self.dropout = nn.Dropout(dropout_prob)

        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout_prob,
            batch_first=True # Input/output tensors are (batch, seq, feature)
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.fc_out = nn.Linear(embed_dim, vocab_size) # Output layer

    def forward(self, tgt_token_ids, memory, tgt_mask=None, memory_mask=None, 
                tgt_padding_mask=None, memory_padding_mask=None):
        """
        Args:
            tgt_token_ids (torch.Tensor): Input LaTeX token IDs for the decoder. Shape: (B, L_tgt)
            memory (torch.Tensor): Output from the encoder (image features). Shape: (B, L_mem, D_mem)
            tgt_mask (torch.Tensor, optional): Mask to prevent attention to future tokens. Shape: (L_tgt, L_tgt)
            memory_mask (torch.Tensor, optional): Mask for encoder output (not typically used here unless parts are masked).
            tgt_padding_mask (torch.Tensor, optional): Mask for padding in target sequence. Shape: (B, L_tgt)
            memory_padding_mask (torch.Tensor, optional): Mask for padding in encoder output. Shape: (B, L_mem)

        Returns:
            torch.Tensor: Logits for each token in the sequence. Shape: (B, L_tgt, vocab_size)
        """
        batch_size, seq_len = tgt_token_ids.shape

        # Token embeddings
        tgt_embeddings = self.embedding(tgt_token_ids) # (B, L_tgt, embed_dim)

        # Add 1D Positional Encoding to target embeddings
        positions = torch.arange(0, seq_len, device=tgt_token_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.positional_encoding(positions) # (B, L_tgt, embed_dim)
        
        tgt = self.dropout(tgt_embeddings + pos_embeddings)

        # Transformer Decoder forward pass
        # The `memory` (encoder_out) becomes the K, V for cross-attention.
        # `tgt` is the Q for self-attention.
        output = self.transformer_decoder(
            tgt=tgt, 
            memory=memory, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask
        ) # (B, L_tgt, embed_dim)

        logits = self.fc_out(output) # (B, L_tgt, vocab_size)
        return logits


class ImageToLatexTransformerModel(nn.Module):
    """
    Combines the CNN Encoder and Transformer Decoder
    to form the complete Image-to-LaTeX model.
    """
    def __init__(self, vocab_size, embed_dim, encoder_dim=512, 
                 encoded_image_size=(20, 100), dropout_prob=0.5,
                 nhead=8, num_decoder_layers=6, dim_feedforward=2048, max_seq_len=500):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimensionality of word embeddings (also d_model for Transformer).
            encoder_dim (int): Dimensionality of the encoder's output features.
            encoded_image_size (tuple): (height, width) of the feature map from encoder.
            dropout_prob (float): Dropout probability.
            nhead (int): Number of attention heads in Transformer.
            num_decoder_layers (int): Number of stacked Transformer decoder layers.
            dim_feedforward (int): Dimension of the feedforward network model in Transformer.
            max_seq_len (int): Maximum expected length of the LaTeX sequence.
        """
        super().__init__()
        self.encoder = CNNEncoder(encoded_image_size=encoded_image_size, 
                                  encoder_dim=encoder_dim, 
                                  dropout_prob=dropout_prob)
        # For Transformer, decoder_dim should typically be equal to embed_dim (d_model)
        # and also compatible with encoder_dim for cross-attention.
        # We'll use embed_dim as the d_model for the Transformer.
        self.decoder = TransformerDecoder(vocab_size=vocab_size, 
                                          embed_dim=embed_dim, 
                                          nhead=nhead, 
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout_prob=dropout_prob,
                                          max_seq_len=max_seq_len)

        # Make sure encoder_dim matches embed_dim for cross-attention compatibility
        if encoder_dim != embed_dim:
            # Add a linear projection layer if dimensions don't match
            # This is critical for the Transformer's cross-attention `memory` input
            self.encoder_output_projection = nn.Linear(encoder_dim, embed_dim)
        else:
            self.encoder_output_projection = nn.Identity() # No-op if dimensions already match

    # Helper function to generate square subsequent mask for Transformer decoder
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    # Helper function to generate padding mask
    def _create_padding_mask(self, seq_ids, pad_token_id):
        # Returns a boolean mask where True means padding, False means not padding
        return (seq_ids == pad_token_id)

    def forward(self, images, input_token_ids, pad_token_id):
        """
        Args:
            images (torch.Tensor): Input images. Shape: (B, 1, H_img, W_img)
            input_token_ids (torch.Tensor): Input LaTeX token IDs for the decoder (teacher forcing).
                                            Shape: (B, max_seq_len - 1)
            pad_token_id (int): ID of the padding token.
        Returns:
            torch.Tensor: Logits for each token in the sequence. Shape: (B, max_seq_len - 1, vocab_size)
        """
        # Encode image
        # encoder_out shape: (B, num_pixels, encoder_dim)
        encoder_out = self.encoder(images) 
        
        # Project encoder output if dimensions don't match embed_dim (d_model for Transformer)
        encoder_out_projected = self.encoder_output_projection(encoder_out) # (B, num_pixels, embed_dim)

        # Generate masks for Transformer Decoder
        tgt_len = input_token_ids.size(1) # Length of target sequence (max_seq_len - 1)
        
        # 1. Look-ahead mask (upper triangular mask) for self-attention in decoder
        # Prevents decoder from seeing future tokens. Shape: (L_tgt, L_tgt)
        tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(input_token_ids.device)

        # 2. Padding mask for target sequence (input_token_ids)
        # Prevents attention to padding tokens in self-attention. Shape: (B, L_tgt)
        tgt_padding_mask = self._create_padding_mask(input_token_ids, pad_token_id)

        # 3. Padding mask for encoder output (memory)
        # Prevents cross-attention to padding-like positions if memory had padding.
        # Here, encoder_out has fixed size for a given input image size, so no real padding for memory.
        # But for robustness, we could create one if encoder_out could dynamically vary.
        # For now, it's optional, as CNN output is usually fixed by design.
        memory_padding_mask = None # (B, num_pixels) - all False if no padding in encoder_out

        # Forward pass through Transformer Decoder
        # The `memory` parameter to the TransformerDecoder is the encoder's output.
        # It's what the cross-attention mechanism will attend to.
        predictions = self.decoder(
            tgt_token_ids=input_token_ids,
            memory=encoder_out_projected,
            tgt_mask=tgt_mask,
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=memory_padding_mask # Pass if you implement memory padding
        ) # (B, L_tgt, vocab_size)

        return predictions


# --- Test the model (Optional, for development/debugging) ---
if __name__ == "__main__":
    print("Testing cnn_transformer_decoder.py model...")
    
    # Dummy parameters (adjust these based on your actual data and desired model size)
    VOCAB_SIZE_EXAMPLE = 1000 
    EMBED_DIM_EXAMPLE = 512 # For Transformer, embed_dim should typically be encoder_dim (d_model)
    ENCODER_DIM_EXAMPLE = 512 # Matches embed_dim for seamless integration
    TARGET_IMG_HEIGHT_EXAMPLE = 160
    TARGET_IMG_WIDTH_EXAMPLE = 800
    ENCODED_IMAGE_SIZE_EXAMPLE = (20, 100) 
    MAX_SEQ_LEN_EXAMPLE = 568 
    BATCH_SIZE_EXAMPLE = 4
    DROPOUT_PROB_EXAMPLE = 0.1 # Transformers often use lower dropout
    
    NHEADS_EXAMPLE = 8 # Common choice for nhead
    NUM_DECODER_LAYERS_EXAMPLE = 6 # Common choice for number of layers
    DIM_FEEDFORWARD_EXAMPLE = 2048 # Common choice (4x embed_dim)

    # Create dummy input data
    dummy_images = torch.randn(BATCH_SIZE_EXAMPLE, 1, TARGET_IMG_HEIGHT_EXAMPLE, TARGET_IMG_WIDTH_EXAMPLE)
    dummy_input_token_ids = torch.randint(0, VOCAB_SIZE_EXAMPLE, (BATCH_SIZE_EXAMPLE, MAX_SEQ_LEN_EXAMPLE - 1))
    DUMMY_PAD_TOKEN_ID = 1 # Match your PAD_TOKEN_ID

    # Instantiate the model
    model = ImageToLatexTransformerModel(
        vocab_size=VOCAB_SIZE_EXAMPLE,
        embed_dim=EMBED_DIM_EXAMPLE,
        encoder_dim=ENCODER_DIM_EXAMPLE,
        encoded_image_size=ENCODED_IMAGE_SIZE_EXAMPLE,
        dropout_prob=DROPOUT_PROB_EXAMPLE,
        nhead=NHEADS_EXAMPLE,
        num_decoder_layers=NUM_DECODER_LAYERS_EXAMPLE,
        dim_feedforward=DIM_FEEDFORWARD_EXAMPLE,
        max_seq_len=MAX_SEQ_LEN_EXAMPLE
    )

    # Move model to a device
    device = torch.device("cpu") 
    model.to(device)
    dummy_images = dummy_images.to(device)
    dummy_input_token_ids = dummy_input_token_ids.to(device)

    # Forward pass
    print(f"\nModel instantiated. Input image shape: {dummy_images.shape}, Input token IDs shape: {dummy_input_token_ids.shape}")
    
    output_logits = model(dummy_images, dummy_input_token_ids, DUMMY_PAD_TOKEN_ID)

    print(f"Model output (logits) shape: {output_logits.shape}")
    
    expected_shape = (BATCH_SIZE_EXAMPLE, MAX_SEQ_LEN_EXAMPLE - 1, VOCAB_SIZE_EXAMPLE)
    assert output_logits.shape == expected_shape, f"Output shape mismatch! Expected {expected_shape}, got {output_logits.shape}"
    
    print("Transformer model forward pass test successful!")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
