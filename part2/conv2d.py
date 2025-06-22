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
    filter_height, filter_width, in_channels_, out_channels  = W.shape

    out_channels_ = bias.shape[0]

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
    c_in_pmax = 2 # temporary example 
    n_tiles_c_in = in_channels // c_in_pmax # this will should be the tiling loop which is 256/128
    free_dim_max = 512
    # (22*22 = 484) but the large image is 224*224 (50176) that's input, what will be the output size?
    # It will be 222*222 if the filter size is 3x3 (49284). So it is equally divided by 18. So even though max can be (22*22 < 512)
    # let's start with 18, this is wrong 222/18 is not real. What if I just tile the height?  
    # Let's start with picking 2 rows of 222 (444) at a time. 
    free_dim_max_rows = 2


    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )
    nl.device_print("starting input:"+str(X.shape)+":", X)
    nl.device_print("starting filter:"+str(W.shape)+":", W)


    X_re = X.reshape((batch_size, in_channels, input_height*input_width))
    
    # # Process the images in batches
    # for batch in nl.affine_range(batch_size): # For Each image
    #     for tile in nl.affine_range(n_tiles_c_in): # For each channel tile (0-127, 128-255)
    #         start_idx_out = tile*c_in_pmax
    #         end_idx_out = tile*c_in_pmax + c_in_pmax
    #         for i in nl.affine_range(filter_height):
    #             # Algorithm that I am trying to implement is load for each row of output, load 
    #             # one input row and then slide the filter
    #             start_1d_index_row = i*input_width
    #             end_1d_index_row = start_1d_index_row+input_width
    #             output_3d = nl.zeros((c_in_pmax, 1, out_width), dtype=X.dtype, buffer=nl.psum)
    #             for tile_inner in nl.affine_range(n_tiles_c_in):
    #                 start_idx_inner = tile_inner*c_in_pmax
    #                 end_idx_inner = tile_inner*c_in_pmax + c_in_pmax
    #                 im2col = nl.load(X_re[batch, start_idx_inner:end_idx_inner, start_1d_index_row:end_1d_index_row, ])
    #                 #filter_load = nl.load(W[i,:,start_idx_inner:end_idx_inner,start_idx_out:end_idx_out])
    #                 for j in nl.affine_range(filter_width):
    #                     filter = nl.load(W[i,j,start_idx_inner:end_idx_inner,start_idx_out:end_idx_out])

    #                     # im2col = nl.ndarray((c_in_pmax, out_height, out_width), dtype=X.dtype, buffer=nl.sbuf)
    #                     # for y in nl.affine_range(out_height):
    #                     #     for x in nl.affine_range(out_width):
    #                     #         in_y = y + i
    #                     #         in_x = x + j
    #                     #         input_pos = in_y * input_width + in_x 
    #                     #         im2col[:,y,x] = nl.load(X_re[batch,start_idx_inner:end_idx_inner,input_pos]) # loads to SBUF
    #                     nl.device_print("Filter:"+str(filter.shape)+":", filter)
    #                     nl.device_print("im2col:"+str(im2col.shape)+":", im2col)
    #                     im2col_sub = im2col[:, j:j+filter_width]
    #                     nl.device_print("im2col-Sub:"+str(im2col_sub.shape)+":", im2col_sub)
    #                     # Now I need to make a mask of im2col so that I only multiply the needed thing
    #                     # What part of input DO I Need here? 
    #                     output_3d += nl.matmul(filter, im2col[:, ], transpose_x = True) # accumulate to PSUM
    #             # Need to copy from psum to sbuf before we can copy it to HBM
    #             #result_sbuf = nl.copy(output_3d, dtype=X.dtype) # PSUM -> SBUF
    #             #nl.store(X_out[batch, start_idx_out:end_idx_out, start_1d_index_row:start_1d_index_row+1, :], value=result_sbuf) # SBUF -> PSUM
    # test = nl.load(X[0,0,0,0])
    # nl.store(X_out[0,0,0,0],test)
    # return X_out 
    

    # Middle of the filter is processed multiple times because filter overlap

    # Process the images in batches
    for batch in nl.affine_range(batch_size): # For Each image
        for tile in nl.affine_range(n_tiles_c_in): # For each channel tile (0-127, 128-255)
            start_idx_out = tile*c_in_pmax
            end_idx_out = tile*c_in_pmax + c_in_pmax
            for row in nl.affine_range(out_height // free_dim_max_rows):
                row_start = free_dim_max_rows * row
                row_end = row_start + free_dim_max_rows
                result_sbuf = nl.zeros((c_in_pmax, free_dim_max_rows, out_width), dtype=X.dtype, buffer=nl.sbuf)
                for i in nl.affine_range(filter_height):
                    for y in nl.affine_range(free_dim_max_rows): # LOOP 1
                        output_3d = nl.zeros((c_in_pmax, free_dim_max_rows, 1), dtype=X.dtype, buffer=nl.psum)
                        for tile_inner in nl.affine_range(n_tiles_c_in):
                            start_idx_inner = tile_inner*c_in_pmax
                            end_idx_inner = tile_inner*c_in_pmax + c_in_pmax
                            for j in nl.affine_range(out_width): # LOOP 2
                                #im2col = nl.ndarray((c_in_pmax, 1, out_width), dtype=X.dtype, buffer=nl.sbuf)
                                filter = nl.load(W[i,j,start_idx_inner:end_idx_inner,start_idx_out:end_idx_out])
                                in_y = y + row_start + i
                                input_pos = in_y * input_width + j
                                im2col = nl.load(X_re[batch,start_idx_inner:end_idx_inner,input_pos:input_pos+out_width]) # loads to SBUF
                                nl.device_print("im2col:"+str(im2col.shape)+":", im2col)
                                nl.device_print("filter:"+str(filter.shape)+":", filter)
                                output_3d += nl.matmul(filter, im2col, transpose_x = True) # accumulate to PSUM
                                nl.device_print("output_3d:"+str(output_3d.shape)+":", output_3d)
                        # Need to copy from psum to sbuf before we can copy it to HBM
                        nl.device_print("output_3d before add:"+str(output_3d.shape)+":", output_3d)
                        result_sbuf[:,:,y] += output_3d # PSUM -> SBUF
                        nl.device_print("waahm7 result_sbuf:"+str(result_sbuf.shape)+":", result_sbuf)
                nl.store(X_out[batch, start_idx_out:end_idx_out, row_start:row_end, :], value=result_sbuf) # SBUF -> PSUM
    # test = nl.load(X[0,0,0,0])
    # nl.store(X_out[0,0,0,0],test)
    return X_out 


