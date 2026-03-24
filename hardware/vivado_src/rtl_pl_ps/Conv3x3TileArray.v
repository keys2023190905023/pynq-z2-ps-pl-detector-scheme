`timescale 1ns / 1ps

module Conv3x3TileArray #(
    parameter OUT_CHANNELS_PAR = 4
)(
    input  wire signed [71:0]                              act_window,
    input  wire signed [OUT_CHANNELS_PAR*72-1:0]           weight_tile,
    input  wire signed [OUT_CHANNELS_PAR*32-1:0]           partial_sums_in,
    output wire signed [OUT_CHANNELS_PAR*32-1:0]           partial_sums_out
);
    genvar idx;
    generate
        for (idx = 0; idx < OUT_CHANNELS_PAR; idx = idx + 1) begin : gen_output_pe
            Conv3x3OutputPE u_pe (
                .act_window(act_window),
                .weight_window(weight_tile[idx*72 +: 72]),
                .partial_sum_in(partial_sums_in[idx*32 +: 32]),
                .partial_sum_out(partial_sums_out[idx*32 +: 32])
            );
        end
    endgenerate

endmodule
