import torch
import torch.nn as nn
import torchvision.models as models
from models.common_layers import PositionalEncoding2D, Attention  # Import our common layers


class CNNEncoder(nn.Module):
    """
    CNN Encoder that extracts features from the input image.
    Based on VGG-like architecture as hinted by the reference paper,
    but adaptable to other backbones like ResNet.
    Includes 2D Positional Encoding.
    """

    def __init__(self, encoded_image_size=(16, 64), encoder_dim=512, dropout_prob=0.5):
        """
        Args:
            encoded_image_size (tuple): (height, width) of the feature map after CNN.
                                        e.g., if input is 160x800 and downsample factor is 10, then 16x80.
                                        The paper uses 8x downsample factor, but typical CNNs may vary.
                                        We'll assume your TARGET_IMG_HEIGHT=160, TARGET_IMG_WIDTH=800 -> 16x80 if downsample 10x.
                                        Let's assume the paper's 8x factor implies 160/8=20, 800/8=100 -> (20, 100)
                                        So, let's set a flexible encoded_image_size or calculate it.
                                        For this example, let's make it fixed as in the paper's derived dimensions.
                                        If input is 160x800 and encoder outputs 1/8th resolution: 20x100.
                                        Let's assume encoded_image_size=(20, 100) for now.
            encoder_dim (int): Dimensionality of the extracted features (channels of CNN output).
            dropout_prob (float): Dropout probability for regularization.
        """
        super().__init__()
        self.encoder_dim = encoder_dim
        self.encoded_image_height = encoded_image_size[0]
        self.encoded_image_width = encoded_image_size[1]

        # Use a pre-trained ResNet-18 as the backbone for feature extraction.
        # ResNet is a common and effective choice.
        # We'll adapt it for single-channel (grayscale) input and remove the classification head.
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Use default weights

        # Modify the first convolutional layer for grayscale input (1 channel instead of 3)
        # Standard resnet.conv1 is Conv2d(3, 64, ...)
        self.resnet_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize with average of pre-trained weights to handle grayscale input
        self.resnet_conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)

        # Remove the classification layer and average pooling from ResNet
        modules = list(resnet.children())[1:-2]  # Exclude conv1, avgpool, fc
        self.resnet_features = nn.Sequential(self.resnet_conv1, *modules)

        # Add a final convolutional layer to adjust output channels to encoder_dim
        # ResNet18 outputs 512 channels before the avgpool.
        # So, the final layer of ResNet18 (layer4) outputs 512 features.
        # We assume encoder_dim is 512 here. If different, an additional conv layer is needed.
        # For simplicity, if encoder_dim != 512, add a 1x1 conv to map channels.
        if self.encoder_dim != 512:
            self.feature_mapper = nn.Conv2d(512, self.encoder_dim, kernel_size=1)
        else:
            self.feature_mapper = nn.Identity()  # No op if already 512

        # Add 2D Positional Encoding
        # Max H/W here should be based on your *input image size* divided by CNN's total stride.
        # E.g., if input is 160x800, ResNet's effective stride is 32 (2x2x2x2 for blocks + 2 for initial conv)
        # So, 160/32=5, 800/32=25. This yields (5, 25) feature map for ResNet18.
        # The paper uses 8x downsample. Let's stick to the paper's implied logic.
        # Let's adjust `encoded_image_size` to be the actual output dimensions of the ResNet.
        # Standard ResNet18 output resolution for 224x224 input is 7x7 (224/32).
        # For 160x800 input, it would be 5x25 approx.
        # Re-evaluating paper: VGG-VeryDeep adapts for OCR to retain spatial locality.
        # The paper states output feature maps are 8 times smaller.
        # Let's define the CNN architecture to achieve that or explicitly adjust.
        # Given your target input image size (160x800), if we aim for 8x downsampling
        # as per the paper, the feature map size would be 20x100.
        # Let's adjust the CNN structure slightly or assume output dimensions.
        # For this setup, we'll ensure the PositionalEncoding matches the expected output of this CNN chain.
        # ResNet will likely give smaller dimensions than 20x100.
        # Let's build a simpler VGG-like CNN for clarity first, consistent with paper's 8x reduction.

        # --- A simpler VGG-like CNN for 8x downsampling ---
        # Input: 1x160x800
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1x80x400 (2x downsample)

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1x40x200 (4x downsample)

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1x20x100 (8x downsample)

            nn.Conv2d(256, encoder_dim, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            # Optional: Another conv layer or just adjust channels to encoder_dim
            # If encoder_dim is different from last conv output (256), add 1x1 conv
            nn.Conv2d(encoder_dim, encoder_dim, kernel_size=1) if encoder_dim != 256 else nn.Identity()
        )
        # The `encoded_image_size` parameter should reflect the output of this CNN
        # With input 160x800 and 3 MaxPool2d(2,2), the output will be 20x100.
        self.encoded_image_height = 20
        self.encoded_image_width = 100

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
                          Shape: (B, num_pixels, encoder_dim) (after flattening)
            torch.Tensor: Flattened feature map before PE, for debugging.
        """
        # Pass through CNN layers
        out = self.conv_layers(images)  # (B, encoder_dim, H_feat, W_feat)

        # Add positional encoding
        out = self.positional_encoding(out)

        # Reshape to (B, num_pixels, encoder_dim) for the Attention mechanism
        # where num_pixels = H_feat * W_feat
        # Permute from (B, C, H, W) to (B, H, W, C) then flatten (B, H*W, C)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.size(0), -1, self.encoder_dim)  # (B, H*W, encoder_dim)

        return self.dropout(out)


class RNNDecoder(nn.Module):
    """
    RNN Decoder with Attention mechanism.
    Generates LaTeX token sequences.
    """

    def __init__(self, vocab_size, embed_dim, decoder_dim, encoder_dim, dropout_prob=0.5):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimensionality of word embeddings.
            decoder_dim (int): Dimensionality of the decoder's LSTM hidden state.
            encoder_dim (int): Dimensionality of the encoder's output features.
            dropout_prob (float): Dropout probability for regularization.
        """
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Embedding layer for tokens
        self.dropout = nn.Dropout(p=dropout_prob)

        # Attention mechanism
        # Here, attention_dim is chosen, common choice is half of decoder_dim or embed_dim
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim=decoder_dim // 2)

        # Initialize LSTM's hidden and cell states from encoder output
        # The paper says "additional layers are added to train these initial states based on the encoder output."
        # A simple approach is a linear layer from mean of encoder output to LSTM states.
        # The mean of encoder output has shape (B, encoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.sigmoid = nn.Sigmoid()  # Use sigmoid for cell state initialization as often done

        # LSTM cell - paper uses Stacked Bidirectional LSTM.
        # For a decoder, a unidirectional LSTM is more common as it generates token by token.
        # Let's simplify to a single-layer unidirectional LSTM first, then consider stacking/bidirectional.
        # Bidirectional LSTMs are usually for encoders.
        # For a decoder, "input-feeding" approach means concatenating context vector with input embedding
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # (embedding + context) as input

        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # Layer to create a 'gate' for context vector
        self.softmax_beta = nn.Sigmoid()  # Sigmoid for the gate

        self.fc = nn.Linear(decoder_dim + encoder_dim, vocab_size)  # Linear layer to output vocabulary scores

    def init_hidden_state(self, encoder_out):
        """
        Initializes the decoder's first hidden and cell states from the encoder output.
        Args:
            encoder_out (torch.Tensor): Output features from the encoder (B, num_pixels, encoder_dim).
        Returns:
            tuple: (h, c) initial hidden and cell states for LSTM.
        """
        # Average encoder output across spatial dimensions to get (B, encoder_dim)
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (B, decoder_dim)
        c = self.init_c(mean_encoder_out)  # (B, decoder_dim)
        return h, c

    def forward(self, encoder_out, input_token_ids, hidden_state=None, cell_state=None):
        """
        Forward pass for the decoder at a single time step (if teacher forcing).
        Or for the full sequence during training if unrolling LSTM.
        For simplicity, this forward expects one time step input_token_ids (B).
        The actual training loop will handle unrolling and teacher forcing.

        Args:
            encoder_out (torch.Tensor): Encoded image features. Shape: (B, num_pixels, encoder_dim)
            input_token_ids (torch.Tensor): Input token IDs for current step. Shape: (B) for single token, or (B, seq_len) for full seq
            hidden_state (torch.Tensor, optional): Previous hidden state. Shape: (B, decoder_dim)
            cell_state (torch.Tensor, optional): Previous cell state. Shape: (B, decoder_dim)
        Returns:
            predictions (torch.Tensor): Logits for next token. Shape: (B, vocab_size)
            hidden_state (torch.Tensor): New hidden state. Shape: (B, decoder_dim)
            cell_state (torch.Tensor): New cell state. Shape: (B, decoder_dim)
            attention_weights (torch.Tensor): Attention weights. Shape: (B, num_pixels)
        """
        batch_size = encoder_out.size(0)

        # Initialize hidden state if first step
        if hidden_state is None and cell_state is None:
            h, c = self.init_hidden_state(encoder_out)
        else:
            h, c = hidden_state, cell_state

        # Unroll the sequence for training (teacher forcing)
        predictions = torch.zeros(batch_size, input_token_ids.size(1), self.vocab_size).to(input_token_ids.device)
        attention_weights_list = []  # For storing attention weights

        for t in range(input_token_ids.size(1)):
            embeddings = self.embedding(input_token_ids[:, t])  # (B, embed_dim)

            # Context vector from attention
            context, alpha = self.attention(encoder_out, h)  # (B, encoder_dim), (B, num_pixels)

            # Input-feeding approach: concatenate context vector with input embeddings
            # This is commonly done to make the LSTM aware of past attention decisions.
            lstm_input = torch.cat((embeddings, context), dim=1)  # (B, embed_dim + encoder_dim)

            h, c = self.lstm(lstm_input, (h, c))  # (B, decoder_dim)

            # Gating mechanism for context vector (f_beta in some implementations)
            # This gate determines how much context to pass to the final classification layer
            gate = self.sigmoid(self.f_beta(h))  # (B, encoder_dim)
            gated_context = gate * context  # (B, encoder_dim)

            # Final prediction layer
            # Concatenate current hidden state and gated context
            score = self.fc(torch.cat((h, gated_context), dim=1))  # (B, vocab_size)
            predictions[:, t, :] = score
            attention_weights_list.append(alpha)  # Store attention weights for visualization/analysis

        # Stack attention weights if you need them for output (e.g., (B, seq_len, num_pixels))
        attention_weights_stacked = torch.stack(attention_weights_list, dim=1) if attention_weights_list else None

        # Return predictions for the full sequence, and final states if needed for inference
        return predictions, h, c, attention_weights_stacked


class ImageToLatexModel(nn.Module):
    """
    Combines the CNN Encoder and RNN Decoder with Attention
    to form the complete Image-to-LaTeX model.
    """

    def __init__(self, vocab_size, embed_dim, encoder_dim=512, decoder_dim=512,
                 encoded_image_size=(20, 100), dropout_prob=0.5):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimensionality of word embeddings.
            encoder_dim (int): Dimensionality of the encoder's output features.
            decoder_dim (int): Dimensionality of the decoder's LSTM hidden state.
            encoded_image_size (tuple): (height, width) of the feature map from encoder.
                                        This should match the CNNEncoder's output dimensions.
            dropout_prob (float): Dropout probability.
        """
        super().__init__()
        self.encoder = CNNEncoder(encoded_image_size=encoded_image_size,
                                  encoder_dim=encoder_dim,
                                  dropout_prob=dropout_prob)
        self.decoder = RNNDecoder(vocab_size=vocab_size,
                                  embed_dim=embed_dim,
                                  decoder_dim=decoder_dim,
                                  encoder_dim=encoder_dim,
                                  dropout_prob=dropout_prob)

    def forward(self, images, input_token_ids):
        """
        Args:
            images (torch.Tensor): Input images. Shape: (B, 1, H_img, W_img)
            input_token_ids (torch.Tensor): Input LaTeX token IDs for the decoder (teacher forcing).
                                            Shape: (B, max_seq_len - 1)
        Returns:
            torch.Tensor: Logits for each token in the sequence. Shape: (B, max_seq_len - 1, vocab_size)
        """
        encoder_out = self.encoder(images)  # (B, num_pixels, encoder_dim)
        predictions, _, _, _ = self.decoder(encoder_out, input_token_ids)
        return predictions


# --- Test the model (Optional, for development/debugging) ---
if __name__ == "__main__":
    print("Testing cnn_lstm_attention.py model...")

    # Dummy parameters (adjust these based on your actual data and desired model size)
    VOCAB_SIZE_EXAMPLE = 1000  # Example vocab size from build_vocab.py
    EMBED_DIM_EXAMPLE = 256
    ENCODER_DIM_EXAMPLE = 512
    DECODER_DIM_EXAMPLE = 512
    TARGET_IMG_HEIGHT_EXAMPLE = 160
    TARGET_IMG_WIDTH_EXAMPLE = 800
    ENCODED_IMAGE_SIZE_EXAMPLE = (20, 100)  # (H/8, W/8) from 160x800 input
    MAX_SEQ_LEN_EXAMPLE = 568  # From your dataset.py (567 + 1 for input_ids/target_ids logic)
    BATCH_SIZE_EXAMPLE = 4
    DROPOUT_PROB_EXAMPLE = 0.5

    # Create dummy input data
    dummy_images = torch.randn(BATCH_SIZE_EXAMPLE, 1, TARGET_IMG_HEIGHT_EXAMPLE, TARGET_IMG_WIDTH_EXAMPLE)
    # input_token_ids will be (max_seq_len - 1)
    dummy_input_token_ids = torch.randint(0, VOCAB_SIZE_EXAMPLE, (BATCH_SIZE_EXAMPLE, MAX_SEQ_LEN_EXAMPLE - 1))

    # Instantiate the model
    model = ImageToLatexModel(
        vocab_size=VOCAB_SIZE_EXAMPLE,
        embed_dim=EMBED_DIM_EXAMPLE,
        encoder_dim=ENCODER_DIM_EXAMPLE,
        decoder_dim=DECODER_DIM_EXAMPLE,
        encoded_image_size=ENCODED_IMAGE_SIZE_EXAMPLE,
        dropout_prob=DROPOUT_PROB_EXAMPLE
    )

    # Move model to a device (CPU for this test, change to 'mps' or 'cuda' for real)
    device = torch.device("cpu")  # For quick test
    model.to(device)
    dummy_images = dummy_images.to(device)
    dummy_input_token_ids = dummy_input_token_ids.to(device)

    # Forward pass
    print(
        f"\nModel instantiated. Input image shape: {dummy_images.shape}, Input token IDs shape: {dummy_input_token_ids.shape}")

    output_logits = model(dummy_images, dummy_input_token_ids)

    print(f"Model output (logits) shape: {output_logits.shape}")

    # Expected output shape: (B, max_seq_len - 1, vocab_size)
    expected_shape = (BATCH_SIZE_EXAMPLE, MAX_SEQ_LEN_EXAMPLE - 1, VOCAB_SIZE_EXAMPLE)
    assert output_logits.shape == expected_shape, f"Output shape mismatch! Expected {expected_shape}, got {output_logits.shape}"

    print("Model forward pass test successful!")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
