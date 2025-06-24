import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):
    batch_size, in_channels, input_height, input_width = X.shape
    # Let's assume W is transposed before passed into this function
    # out_channels, in_channels_, filter_height, filter_width = W.shape 
    # Weights can completely fit inside SBUF, so maybe load transpose?
    #filter_height, filter_width, in_channels_, out_channels  = W.shape
    #out_channels, in_channels_, filter_height, filter_width = W.shape
    # Filter Height, Filter Weight, Input Channels, Output Channels
    out_channels_ = bias.shape[0]
    out_channels, in_channels_, filter_height, filter_width = W.shape
    # reshaped = W.reshape([out_channels*in_channels_, filter_height*filter_width])
    # transposed = nl.load_transpose2d(reshaped)
    # final_weights = transposed.reshape([filter_height, filter_width, out_channels, in_channels_])
    # nl.device_print("W:"+str(W.shape)+":", W)
    # nl.device_print("transposed filter:"+str(final.shape)+":", final)
    # for i in nl.affine_range(filter_height):
    #     for j in nl.affine_range(filter_width):
    #         W_re = nl.load(W[:,:,i,j])

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    #assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax # max partition dimension (128)
    #c_in_pmax = 2 # temporary example 
    n_tiles_c_in = in_channels // c_in_pmax # this will should be the tiling loop which is 256/128
    n_tiles_c_out = out_channels // c_in_pmax # this will should be the tiling loop which is 256/128
    free_dim_max = 512
    # (22*22 = 484) but the large image is 224*224 (50176) that's input, what will be the output size?
    # It will be 222*222 if the filter size is 3x3 (49284). So it is equally divided by 18. So even though max can be (22*22 < 512)
    # let's start with 18, this is wrong 222/18 is not real. What if I just tile the height?  
    # Let's start with picking 2 rows of 222 (444) at a time. 
    free_dim_max_rows = 2
    #out_channels, in_channels_, filter_height, filter_width = W.shape
  
    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )
    # nl.device_print("starting input:"+str(X.shape)+":", X)
    # nl.device_print("starting filter:"+str(W.shape)+":", W)
    W_re = W.reshape((out_channels, in_channels, filter_height*filter_width))

    X_re = X.reshape((batch_size, in_channels, input_height*input_width))
    W_trans = nl.ndarray((c_in_pmax, n_tiles_c_in, out_channels, filter_height, filter_width), dtype=X.dtype)
    for tile_out in nl.affine_range(n_tiles_c_out): # For each channel tile (0-127, 128-255)
        start_idx_out = tile_out*c_in_pmax
        end_idx_out = tile_out*c_in_pmax + c_in_pmax
        for tile_inner in nl.affine_range(n_tiles_c_in):
                start_idx_inner = tile_inner*c_in_pmax
                end_idx_inner = tile_inner*c_in_pmax + c_in_pmax 
                W_in = nl.load(W[start_idx_out:end_idx_out, start_idx_inner:end_idx_inner, :, :])
                for i in nl.affine_range(filter_height): # row 0-1, 2-3 etc        
                    for j in nl.affine_range(filter_width):
                         filter = nl.transpose(W_in[:, :, i, j])
                         W_trans[:,tile_inner, start_idx_out:end_idx_out, i, j] = filter

    for batch in nl.affine_range(batch_size): # For Each image
        for tile_out in nl.affine_range(n_tiles_c_out): # For each channel tile (0-127, 128-255)
            start_idx_out = tile_out*c_in_pmax
            end_idx_out = tile_out*c_in_pmax + c_in_pmax
            for row in nl.affine_range(out_height // free_dim_max_rows):
                row_start = free_dim_max_rows * row
                row_end = row_start + free_dim_max_rows
                output_3d = nl.zeros((c_in_pmax, free_dim_max_rows, out_width), dtype=nl.float32, buffer=nl.psum) 
                for tile_inner in nl.affine_range(n_tiles_c_in):
                    start_idx_inner = tile_inner*c_in_pmax
                    end_idx_inner = tile_inner*c_in_pmax + c_in_pmax 

                    total_start = row_start * input_width
                    total_end = (row_start + filter_height - 1 + free_dim_max_rows - 1) * input_width + filter_width + out_width - 1
                    im2col_all = nl.load(X_re[batch, start_idx_inner:end_idx_inner, total_start:total_end])
                    for i in nl.affine_range(filter_height): # row 0-1, 2-3 etc        
                        for y in nl.affine_range(free_dim_max_rows):
                            offset = (i+y) * input_width
                            im2col_mega = im2col_all[:, offset:offset + filter_width + out_width]
                            for j in nl.affine_range(filter_width):
                                filter = W_trans[:,tile_inner, start_idx_out:end_idx_out, i, j]
                                im2col = im2col_mega[:,j:j+out_width]
                                output_3d[:,y,:] += nl.matmul(filter, im2col, transpose_x = True) # accumulate to PSUM
                bias_sbm = nl.load(bias[start_idx_out:end_idx_out])
                result_biased = nl.add(output_3d, bias_sbm)
                nl.store(X_out[batch, start_idx_out:end_idx_out, row_start:row_end, :], value=result_biased) # SBUF -> PSUM
    return X_out 
                             


@nki.jit
def fused_conv2d_maxpool_with_max_pool(X, W, bias, pool_size=1):
    batch_size, in_channels, input_height, input_width = X.shape
    # Let's assume W is transposed before passed into this function
    # out_channels, in_channels_, filter_height, filter_width = W.shape 
    # Weights can completely fit inside SBUF, so maybe load transpose?
    #filter_height, filter_width, in_channels_, out_channels  = W.shape
    #out_channels, in_channels_, filter_height, filter_width = W.shape
    # Filter Height, Filter Weight, Input Channels, Output Channels
    out_channels_ = bias.shape[0]
    out_channels, in_channels_, filter_height, filter_width = W.shape
    # reshaped = W.reshape([out_channels*in_channels_, filter_height*filter_width])
    # transposed = nl.load_transpose2d(reshaped)
    # final_weights = transposed.reshape([filter_height, filter_width, out_channels, in_channels_])
    # nl.device_print("W:"+str(W.shape)+":", W)
    # nl.device_print("transposed filter:"+str(final.shape)+":", final)
    # for i in nl.affine_range(filter_height):
    #     for j in nl.affine_range(filter_width):
    #         W_re = nl.load(W[:,:,i,j])

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    #assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax # max partition dimension (128)
    #c_in_pmax = 2 # temporary example 
    n_tiles_c_in = in_channels // c_in_pmax # this will should be the tiling loop which is 256/128
    n_tiles_c_out = out_channels // c_in_pmax # this will should be the tiling loop which is 256/128
    free_dim_max = 512
    # (22*22 = 484) but the large image is 224*224 (50176) that's input, what will be the output size?
    # It will be 222*222 if the filter size is 3x3 (49284). So it is equally divided by 18. So even though max can be (22*22 < 512)
    # let's start with 18, this is wrong 222/18 is not real. What if I just tile the height?  
    # Let's start with picking 2 rows of 222 (444) at a time. 
    free_dim_max_rows = 2
    #out_channels, in_channels_, filter_height, filter_width = W.shape
  
    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )
    # nl.device_print("starting input:"+str(X.shape)+":", X)
    # nl.device_print("starting filter:"+str(W.shape)+":", W)
    W_re = W.reshape((out_channels, in_channels, filter_height*filter_width))

    X_re = X.reshape((batch_size, in_channels, input_height*input_width))
    W_trans = nl.ndarray((c_in_pmax, n_tiles_c_in, out_channels, filter_height, filter_width), dtype=X.dtype)
    for tile_out in nl.affine_range(n_tiles_c_out): # For each channel tile (0-127, 128-255)
        start_idx_out = tile_out*c_in_pmax
        end_idx_out = tile_out*c_in_pmax + c_in_pmax
        for tile_inner in nl.affine_range(n_tiles_c_in):
                start_idx_inner = tile_inner*c_in_pmax
                end_idx_inner = tile_inner*c_in_pmax + c_in_pmax 
                W_in = nl.load(W[start_idx_out:end_idx_out, start_idx_inner:end_idx_inner, :, :])
                for i in nl.affine_range(filter_height): # row 0-1, 2-3 etc        
                    for j in nl.affine_range(filter_width):
                         filter = nl.transpose(W_in[:, :, i, j])
                         W_trans[:,tile_inner, start_idx_out:end_idx_out, i, j] = filter
    
    for batch in nl.affine_range(batch_size): # For Each image
        for tile_out in nl.affine_range(n_tiles_c_out): # For each channel tile (0-127, 128-255)
            start_idx_out = tile_out*c_in_pmax
            end_idx_out = tile_out*c_in_pmax + c_in_pmax
            bias_sbm = nl.load(bias[start_idx_out:end_idx_out])
            for row in nl.affine_range(out_height // free_dim_max_rows):
                row_start = free_dim_max_rows * row
                row_end = row_start + free_dim_max_rows
                output_3d = nl.zeros((c_in_pmax, free_dim_max_rows, out_width), dtype=nl.float32, buffer=nl.psum) 
                for tile_inner in nl.affine_range(n_tiles_c_in):
                    start_idx_inner = tile_inner*c_in_pmax
                    end_idx_inner = tile_inner*c_in_pmax + c_in_pmax 

                    total_start = row_start * input_width
                    total_end = (row_start + filter_height - 1 + free_dim_max_rows - 1) * input_width + filter_width + out_width - 1
                    im2col_all = nl.load(X_re[batch, start_idx_inner:end_idx_inner, total_start:total_end])
                    for i in nl.affine_range(filter_height): # row 0-1, 2-3 etc        
                        for y in nl.affine_range(free_dim_max_rows):
                            offset = (i+y) * input_width
                            im2col_mega = im2col_all[:, offset:offset + filter_width + out_width]
                            for j in nl.affine_range(filter_width):
                                filter = W_trans[:,tile_inner, start_idx_out:end_idx_out, i, j]
                                im2col = im2col_mega[:,j:j+out_width]
                                output_3d[:,y,:] += nl.matmul(filter, im2col, transpose_x = True) # accumulate to PSUM
                result = nl.add(output_3d, bias_sbm)
                result = result.reshape((c_in_pmax, free_dim_max_rows, out_width // 2, 2))
                result = nl.max(result, axis=[1,3]) # collapse the height & width into 1
                nl.store(X_out[batch, start_idx_out:end_idx_out, row , :], value=result) # SBUF -> PSUM
    return X_out 
                           