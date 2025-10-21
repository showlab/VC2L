import numpy as np
from PIL import Image
import os
def get_pp_render_text(image_size: int=224, font_size: int = 16, max_chars=768,
                       background_brightness=127, text_brightness=0,
                       lower=True, monospace=False, spacing=1, min_width=4,
                       resize_method="area"):
    """Renders text as image, using binary Unifont.
    Args:
    image_size: Width/height of output image.
    font_size: Font size to use. Recommended to leave at 16, as this requires
    no resizing, and is safe.
    max_chars: Maximum inpute characters to render, to make faster.
    background_brightness: (r, g, b) of background pixels.
    text_brightness: (r, g, b) of text pixels.
    lower: whether to lowercase.
    monospace: if False, text characters are horizontally trimmed according to
        `spacing` and `minwidth` args.
    spacing: # pixels between each letter.
    min_width: Minimum width of each letter. Useful to make sure e.g. spaces and
        full stops aren't collapsed to nothing.
    resize_method: resize method to use if fontsize != 16.

    Returns:
    Function which renders text as an image.
    """
    bit_embedding = np.zeros((0x200000, 32), dtype=np.uint8)
    colpattern = {64: range(32),
                32: sorted(tuple(range(0, 32, 4)) + tuple(range(2, 32, 4)))}
    unifont_path = "./unifont-9.0.06.hex"
    unifont_upper_path = "./unifont_upper-9.0.06.hex"
    with open(unifont_path, 'r') as f:
        for line in f:
            row = int(line[0:4], 16)
            hexbits = line[5:-1]
            bit_embedding[row, colpattern[len(hexbits)]] = bytearray.fromhex(hexbits)
    with open(unifont_upper_path, 'r') as f:
        for line in f:
            row = int(line[0:6], 16)
            hexbits = line[7:-1]
            bit_embedding[row, colpattern[len(hexbits)]] = bytearray.fromhex(hexbits)
    params = np.array(bit_embedding, dtype=np.uint8)

    def trim_letter_np(letter):
        """Remove white space based on the letter size."""
        v = np.max(letter, axis=0)
        has_pixels = np.where(v)[0]
        no_pixels = np.max(v) == 0
        first = 0 if no_pixels else has_pixels[0]
        last = 0 if no_pixels else has_pixels[-1]
        first = max(first - spacing, 0)
        last = max(last + spacing, first + min_width)
        return np.transpose(letter[:, first:last])

    def to_image_np(rendered, width, height=None):
        """Makes a nice square image from a long string of rendered characters."""
        height = height or width
        padded = np.zeros((width, height), np.uint8)
        max_letter_width = np.max([r.shape[0] for r in rendered])
        row_lengths = np.cumsum([r.shape[0] for r in rendered]).astype(np.float32)
        div = width - max_letter_width  # For rounding errors.
        row_idx = np.floor(row_lengths / div).astype(np.int64)
        trimmed = [[rendered[i] for i in range(len(rendered)) if row_idx[i] == row]
            for row in range(max(row_idx)+1)]
        trimmed = [np.concatenate(line, axis=0).transpose() for line in trimmed]
        for line_id, line in enumerate(trimmed):
            if line_id < (height // 16):
                # print(line_id*16, (line_id+1)*16)
                padded[line_id*16:(line_id+1)*16, 0:line.shape[1]] = line
        return padded, max(row_idx)

    def to_image_np_448(rendered, width, height=None):
        """Makes a nice square image from a long string of rendered characters."""
        height = height or width
        padded = np.zeros((width*4, height), np.uint8)
        max_letter_width = np.max([r.shape[0] for r in rendered])
        row_lengths = np.cumsum([r.shape[0] for r in rendered]).astype(np.float32)
        div = width - max_letter_width  # For rounding errors.
        row_idx = np.floor(row_lengths / div).astype(np.int64)
        # print("row_idx", max(row_idx)+1)
        trimmed = [[rendered[i] for i in range(len(rendered)) if row_idx[i] == row]
            for row in range(max(row_idx)+1)]
        trimmed = [np.concatenate(line, axis=0).transpose() for line in trimmed]
        for line_id, line in enumerate(trimmed):
            if line_id < (height // 16)*4:
                # print(line_id)
                # print(line_id*16, (line_id+1)*16)
                padded[line_id*16:(line_id+1)*16, 0:line.shape[1]] = line
        return padded, max(row_idx)

    def render_np_blend(text, image):
        ids = np.fromiter(list(ord(char) for char in text.lower() ), dtype=np.uint16)
        ids = ids[:max_chars]
        embed = params[ids]  # Get the letters
        vertical = np.reshape(embed, (1, -1))
        repl = np.reshape(np.transpose(np.tile(vertical, (8, 1))), (-1,))
        ones = np.ones_like(repl, dtype=np.uint8)
        index = np.cumsum(ones, dtype=np.uint8) - 1
        sevens = np.full_like(repl, 7, dtype=np.uint8)
        moded = np.bitwise_and(index, sevens)
        shifted = np.right_shift(repl, np.bitwise_xor(moded, sevens))
        anded = np.bitwise_and(shifted, ones)
        letters = np.reshape(anded, (len(ids), 16, 16))
        letters = [trim_letter_np(letter) for letter in letters]
        img, num_row = to_image_np(letters, image_size)  # A nice square image.
        img *= np.array(text_brightness - background_brightness).astype(np.uint8)  # Rescale value range.
        img += np.array(background_brightness).astype(np.uint8)
        img = Image.fromarray(img).convert('RGB')
        # print(type(image))
        image[:(num_row+1)*16] = np.asarray(img)[:(num_row+1)*16]
        return image
        
    def render_np_moscia(text, image=[0,0,0,0]):
        # print(len(text))
        ids = np.fromiter(list(ord(char) for char in text.lower() ), dtype=np.uint16)
        ids = ids[:max_chars]
        embed = params[ids]  # Get the letters
        vertical = np.reshape(embed, (1, -1))
        repl = np.reshape(np.transpose(np.tile(vertical, (8, 1))), (-1,))
        ones = np.ones_like(repl, dtype=np.uint8)
        index = np.cumsum(ones, dtype=np.uint8) - 1
        sevens = np.full_like(repl, 7, dtype=np.uint8)
        moded = np.bitwise_and(index, sevens)
        shifted = np.right_shift(repl, np.bitwise_xor(moded, sevens))
        anded = np.bitwise_and(shifted, ones)
        letters = np.reshape(anded, (len(ids), 16, 16))
        letters = [trim_letter_np(letter) for letter in letters]
        img, num_row = to_image_np_448(letters, image_size)  # A nice square image.
        # print(num_row)
        img *= np.array(text_brightness - background_brightness).astype(np.uint8)  # Rescale value range.
        img += np.array(background_brightness).astype(np.uint8)
        img = np.array(Image.fromarray(img).convert('RGB'))
        # print(type(image))
        full_image = np.array(Image.new(mode="RGB", size=(448, 448)))
        curr_row = 0
        if image[0] is None:
            full_image[:224, :224] = img[curr_row*16:(curr_row+14)*16]
            curr_row += 14
        else:
            full_image[:224, :224] = np.array(image[0])
        if image[1] is None:
            full_image[:224, 224:] = img[curr_row*16:(curr_row+14)*16]
            curr_row += 14
        else:
            full_image[:224, 224:] = np.array(image[1])
        if image[2] is None:
            full_image[224:, :224] = img[curr_row*16:(curr_row+14)*16]
            curr_row += 14
        else:
            full_image[224:, :224] = np.array(image[2])
        if image[3] is None:
            full_image[224:, 224:] = img[curr_row*16:(curr_row+14)*16]
            curr_row += 14
        else:
            full_image[224:, 224:] = np.array(image[3])
        return full_image
    return render_np_moscia
