import os
import numpy as np
from scipy.fftpack import dct, idct
from scipy.ndimage import zoom
from PIL import Image
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(np.array(img))
    return images

def learn_q_tables(train_images, target_bpp):
    """
    Ajustar heuristicamente as tabelas Q, vindas do JPEG, de acordo com a
    taxa de bit selecionada. Menores taxas de bit equivalem a maiores valores de Q,
    isto é, quantização mais agressiva.
    Isto é apenas uma tentativa de encontrar passos de quantização usando valores já conhecidos.
    Na realidade, como os valores para target_bpp 3.0 são muito pequenos, a tabela Q retornada
    é composta de varios 1 (Ex: 0.04*16 approx 0 -> clipped to 1). O mesmo vale para 0.25, que
    retorna passos de quantização muito grandes. Um trabalho mais adequado pode ser feito no futuro.
    """
    # Padrão tabelas Q do JPEG
    luminance_q_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    chrominance_q_table = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ])

    if target_bpp == 3.0:
        scale_luma = 0.04
        scale_chroma = 0.125
    elif target_bpp == 1.0:
        scale_luma = 0.5  # More aggressive quantization
        scale_chroma = 0.5
    elif target_bpp == 0.25:
        scale_luma = 12.0  # Very aggressive quantization
        scale_chroma = 12.0

    luminance_q_table = np.clip(luminance_q_table * scale_luma, 1, 255).astype(int)
    chrominance_q_table = np.clip(chrominance_q_table * scale_chroma, 1, 255).astype(int)

    return luminance_q_table, chrominance_q_table


def test_q_tables(test_images, luminance_q_table, chrominance_q_table):
    psnrs = []
    for img_array in test_images:
        # Convert to YCbCr
        original_ycbcr = RGB2YCbCr(img_array)

        # Downsample Cb/Cr (4:2:0)
        downsampled_channels = channel_downsampler(original_ycbcr)

        # Split into 8x8 blocks
        blocks = split_into_blocks_with_padding(downsampled_channels)
        shapes = [downsampled_channels[0].shape,
                  downsampled_channels[1].shape,
                  downsampled_channels[2].shape]

        # DCT -> Quantize -> Dequantize -> IDCT
        dct_blocks = DCT_blocks(blocks)
        quantized_blocks = quantize_blocks(dct_blocks, luminance_q_table, chrominance_q_table)
        dequantized_blocks = dequantize_blocks(quantized_blocks, luminance_q_table, chrominance_q_table)
        reconstructed_blocks = IDCT_blocks(dequantized_blocks)

        # Reconstruct channels
        reconstructed_channels = reconstruct_from_blocks(reconstructed_blocks, shapes)
        upsampled_channels = channel_upsampler(reconstructed_channels, original_ycbcr.shape)

        # Combine and convert to RGB
        upsampled_ycbcr = np.zeros_like(original_ycbcr)
        upsampled_ycbcr[:, :, 0] = upsampled_channels[0]
        upsampled_ycbcr[:, :, 1] = upsampled_channels[1]
        upsampled_ycbcr[:, :, 2] = upsampled_channels[2]
        final_rgb = YCbCr2RGB(upsampled_ycbcr)

        # Compute PSNR
        mse_val = np.mean((img_array - final_rgb) ** 2)
        psnr = 10 * np.log10(255 ** 2 / mse_val)
        psnrs.append(psnr)

    return np.mean(psnrs)


# Função para converter RGB para YCbCr
def RGB2YCbCr(img_array):
    """
    Converte uma imagem no formato RGB para o espaço de cores YCbCr.
    Parâmetros:
        img_array (numpy.ndarray): Array de imagem RGB com valores entre 0 e 255.
    Retorna:
        numpy.ndarray: Array de imagem no formato YCbCr com valores entre 0 e 255.
    """
    basisMatrix = np.array([[0.299, 0.587, 0.114],
                            [-0.168736, -0.331264, 0.5],
                            [0.5, -0.418688, -0.081312]])
    offset = np.array([0, 128, 128])
    ycbcr = np.clip(img_array.dot(basisMatrix.T) + offset, 0, 255)
    return ycbcr

# Função para converter YCbCr para RGB
def YCbCr2RGB(img_array):
    """
    Converte uma imagem no formato YCbCr para o espaço de cores RGB.
    """
    basisMatrix = np.array([[1.0, 0.0, 1.402],
                            [1.0, -0.344136, -0.714136],
                            [1.0, 1.772, 0.0]])
    offset = np.array([0, 128, 128])
    rgb = (img_array - offset).dot(basisMatrix.T)
    rgb = np.clip(rgb, 0, 255)
    return rgb.astype(np.uint8)

# Função para realizar downsampling 4:2:0
def channel_downsampler(img_array, channel=(1, 2)):
    """
    Realiza o downsampling dos canais selecionados na imagem.
    """
    downsampled_channels = []
    downsampled_channels.append(img_array[:,:,0])
    for ch in channel:
        downsampled_channels.append(img_array[:, :, ch][::2, ::2])
    return downsampled_channels

# Função para realizar upsampling
def channel_upsampler(downsampled_channels, original_shape, channel=(1, 2)):
    """
    Realiza o upsampling dos canais reduzidos para restaurar o tamanho original.
    """
    upsampled_channels = []

    # Adicionar o canal Y (não reduzido)
    upsampled_channels.append(downsampled_channels[0])

    # Fazer upsampling nos canais Cb e Cr
    for idx, ch in enumerate(channel):
        # Aplicar zoom para restaurar o tamanho original
        upsampled_channel = zoom(downsampled_channels[idx + 1], (original_shape[0] / downsampled_channels[idx + 1].shape[0],
                                                                 original_shape[1] / downsampled_channels[idx + 1].shape[1]),
                                 order=1)  # Interpolação linear
        upsampled_channels.append(upsampled_channel)

    return upsampled_channels

def split_into_blocks_with_padding(img_downsampled):
    """
    Divide os canais de uma imagem em blocos de 8x8, adicionando padding se necessário.
    """
    blocks = []

    for idx, channel in enumerate(img_downsampled):
        height, width = channel.shape

        # Calcular o tamanho necessário de padding
        padded_height = (height + 7) // 8 * 8  # Próximo múltiplo de 8
        padded_width = (width + 7) // 8 * 8    # Próximo múltiplo de 8

        # Criar o canal com padding (valores padrão = 0)
        padded_channel = np.zeros((padded_height, padded_width), dtype=channel.dtype)
        padded_channel[:height, :width] = channel  # Copiar dados originais

        channel_blocks = []

        # Iterar sobre o canal em passos de 8 pixels
        for i in range(0, padded_height, 8):
            for j in range(0, padded_width, 8):
                block = padded_channel[i:i+8, j:j+8]
                channel_blocks.append(block)

        blocks.append(channel_blocks)

    return blocks

def reconstruct_from_blocks(blocks, original_shapes):
    """
    Reconstrói os canais de uma imagem a partir de blocos 8x8 e remove o padding.
    """
    reconstructed_channels = []

    for idx, channel_blocks in enumerate(blocks):
        original_height, original_width = original_shapes[idx]

        # Calcular o tamanho do canal após padding
        padded_height = (original_height + 7) // 8 * 8
        padded_width = (original_width + 7) // 8 * 8

        # Reconstruir o canal com padding
        padded_channel = np.zeros((padded_height, padded_width), dtype=channel_blocks[0].dtype)

        # Preencher o canal com os blocos
        block_idx = 0
        for i in range(0, padded_height, 8):
            for j in range(0, padded_width, 8):
                padded_channel[i:i+8, j:j+8] = channel_blocks[block_idx]
                block_idx += 1

        # Cortar o padding para voltar ao tamanho original
        reconstructed_channel = padded_channel[:original_height, :original_width]
        reconstructed_channels.append(reconstructed_channel)

    return reconstructed_channels

from scipy.fftpack import dct
from scipy.fftpack import idct

def DCT_blocks(blocks):
    """
    Aplica a Transformada Discreta do Cosseno (DCT) em cada bloco 8x8.
    """
    dct_blocks = []

    for channel_blocks in blocks:
        dct_channel_blocks = []
        for block in channel_blocks:
            # Aplicar DCT em 2D no bloco 8x8
            dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
            dct_channel_blocks.append(dct_block)
        dct_blocks.append(dct_channel_blocks)

    return dct_blocks

def IDCT_blocks(dct_blocks):
    """
    Aplica a Transformada Inversa do Cosseno Discreta (IDCT) em blocos 8x8.
    """
    reconstructed_blocks = []
    for channel_blocks in dct_blocks:
        channel_reconstructed = []
        for block in channel_blocks:
            # Aplica a IDCT 2D no bloco
            idct_block = idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
            # Arredonda e converte para valores inteiros no intervalo válido
            idct_block = np.round(idct_block).astype(np.uint8)
            channel_reconstructed.append(idct_block)
        reconstructed_blocks.append(channel_reconstructed)
    return reconstructed_blocks


def quantize_blocks(blocks,luminance_q_table,chrominance_q_table):
  def quantize(block, q_table):
    return np.round(block / q_table)

  quantized_blocks = []
  for idx, channel_blocks in enumerate(blocks):
    channel_quantized = []
    for block in channel_blocks:
      if idx == 0:
        channel_quantized.append(quantize(block, luminance_q_table))
      else:
        channel_quantized.append(quantize(block, chrominance_q_table))
    quantized_blocks.append(channel_quantized)
  return quantized_blocks

def dequantize_blocks(quantized_blocks,luminance_q_table,chrominance_q_table):

  def dequantize(block, q_table):
    return block * q_table

  dequantized_blocks = []
  for idx, channel_blocks in enumerate(quantized_blocks):
    channel_dequantized = []
    for block in channel_blocks:
      if idx == 0:
        channel_dequantized.append(dequantize(block, luminance_q_table))
      else:
        channel_dequantized.append(dequantize(block, chrominance_q_table))
    dequantized_blocks.append(channel_dequantized)
  return dequantized_blocks

# Função para obter o MSE
def mse(original, filtered):
  return np.mean((original - filtered) ** 2)


import numpy as np

def estimate_mean_bpp(train_images, luma_table, chroma_table):
    """Estimate the mean bits per pixel (BPP) across all training images.
    """
    total_bpp = 0.0
    num_images = len(train_images)
    
    for img_array in train_images:
        original_shape = img_array.shape[:2]
        original_ycbcr = RGB2YCbCr(img_array)
        downsampled = channel_downsampler(original_ycbcr)
        blocks = split_into_blocks_with_padding(downsampled)
        dct_blocks = DCT_blocks(blocks)
        quantized_blocks = quantize_blocks(dct_blocks, luma_table, chroma_table)
        
        # After quantization, analyze a block
        sample_block = quantized_blocks[0][0]  # First luminance block
        
        print("Non-zero coeffs:", np.sum(sample_block != 0), "/ 64")
        print("Max coeff:", np.max(np.abs(sample_block)))

        total_pixels = original_shape[0] * original_shape[1]
        total_bits = 0

        for channel_blocks in quantized_blocks:
            for channel in channel_blocks:
                # Count non-zero coeffs + assume 8 bits for DC, 4 for AC
                nz_coeffs = np.sum(channel != 0)
                if nz_coeffs > 0:
                    total_bits += 8  # DC coefficient (higher precision)
                    total_bits += (nz_coeffs - 1) * 4  # AC coefficients (lower precision)

        total_bpp += total_bits / total_pixels
    
    return total_bpp / num_images

if __name__ == "__main__":
    # Load training and test images
    train_images = load_images_from_folder("train")
    test_images = load_images_from_folder("test")

    # Target bit rates
    target_bpps = [3.0, 1.0, 0.25]

    for bpp in target_bpps:
        # Learn Q-tables for this bit rate
        luminance_q_table, chrominance_q_table = learn_q_tables(train_images, bpp)

        avg_psnr = test_q_tables(train_images, luminance_q_table, chrominance_q_table)
        
        actual_bpp = estimate_mean_bpp(train_images, luminance_q_table, chrominance_q_table)

        print(f"Target: {bpp} bpp | Actual: {actual_bpp:.2f} bpp | PSNR: {avg_psnr:.2f} dB")

        sample_psnr = test_q_tables([test_images[0]], luminance_q_table, chrominance_q_table)
        img_array = test_images[0]
        original_ycbcr = RGB2YCbCr(img_array)
        downsampled_channels = channel_downsampler(original_ycbcr)
        blocks = split_into_blocks_with_padding(downsampled_channels)
        shapes = [downsampled_channels[0].shape,
                  downsampled_channels[1].shape,
                  downsampled_channels[2].shape]
        dct_blocks = DCT_blocks(blocks)
        quantized_blocks = quantize_blocks(dct_blocks, luminance_q_table, chrominance_q_table)
        dequantized_blocks = dequantize_blocks(quantized_blocks, luminance_q_table, chrominance_q_table)
        reconstructed_blocks = IDCT_blocks(dequantized_blocks)
        reconstructed_channels = reconstruct_from_blocks(reconstructed_blocks, shapes)
        upsampled_channels = channel_upsampler(reconstructed_channels, original_ycbcr.shape)
        upsampled_ycbcr = np.zeros_like(original_ycbcr)
        upsampled_ycbcr[:, :, 0] = upsampled_channels[0]
        upsampled_ycbcr[:, :, 1] = upsampled_channels[1]
        upsampled_ycbcr[:, :, 2] = upsampled_channels[2]
        final_rgb = YCbCr2RGB(upsampled_ycbcr)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_array)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(final_rgb)
        plt.title(f"Compressed ({bpp} bpp, PSNR: {sample_psnr:.2f} dB)")
        plt.axis("off")
    plt.show()