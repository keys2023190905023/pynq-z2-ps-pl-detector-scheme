`timescale 1ns / 1ps

module PlPsConvOperatorTop #(
    parameter OUT_CHANNELS_PAR = 4
)(
    input  wire                                     clk,
    input  wire                                     rst_n,
    input  wire                                     ap_start,
    input  wire                                     frame_last,
    input  wire                                     pixel_valid,
    input  wire                                     window_valid,
    input  wire                                     channel_first,
    input  wire                                     channel_last,
    input  wire                                     relu_enable,
    input  wire signed [71:0]                       act_window,
    input  wire signed [OUT_CHANNELS_PAR*72-1:0]    weight_tile,
    input  wire signed [OUT_CHANNELS_PAR*32-1:0]    bias_tile,
    input  wire signed [OUT_CHANNELS_PAR*32-1:0]    accum_read_data,
    input  wire signed [15:0]                       quant_scale,
    input  wire [4:0]                               quant_shift,
    input  wire signed [7:0]                        output_zp,
    output reg                                      ap_done,
    output reg                                      ap_idle,
    output reg                                      accum_write_en,
    output reg  signed [OUT_CHANNELS_PAR*32-1:0]    accum_write_data,
    output reg                                      ofm_valid,
    output reg  signed [OUT_CHANNELS_PAR*8-1:0]     ofm_write_data
);
    wire signed [OUT_CHANNELS_PAR*32-1:0] base_partial_sums = channel_first ? {OUT_CHANNELS_PAR{32'sd0}} : accum_read_data;
    wire signed [OUT_CHANNELS_PAR*32-1:0] next_partial_sums;

    reg busy;
    integer idx;
    reg signed [31:0] partial_value;
    reg signed [31:0] biased_value;
    reg signed [63:0] scaled_value;
    reg signed [31:0] quantized_value;

    function signed [7:0] clamp_i8;
        input signed [31:0] value;
        begin
            if (value > 127) begin
                clamp_i8 = 8'sd127;
            end else if (value < -128) begin
                clamp_i8 = -8'sd128;
            end else begin
                clamp_i8 = value[7:0];
            end
        end
    endfunction

    Conv3x3TileArray #(
        .OUT_CHANNELS_PAR(OUT_CHANNELS_PAR)
    ) u_tile_array (
        .act_window(act_window),
        .weight_tile(weight_tile),
        .partial_sums_in(base_partial_sums),
        .partial_sums_out(next_partial_sums)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy            <= 1'b0;
            ap_done         <= 1'b0;
            ap_idle         <= 1'b1;
            accum_write_en  <= 1'b0;
            accum_write_data <= {OUT_CHANNELS_PAR{32'sd0}};
            ofm_valid       <= 1'b0;
            ofm_write_data  <= {OUT_CHANNELS_PAR{8'sd0}};
        end else begin
            ap_done        <= 1'b0;
            accum_write_en <= 1'b0;
            ofm_valid      <= 1'b0;

            if (ap_start) begin
                busy    <= 1'b1;
                ap_idle <= 1'b0;
            end

            if (busy && pixel_valid && window_valid) begin
                if (channel_last) begin
                    for (idx = 0; idx < OUT_CHANNELS_PAR; idx = idx + 1) begin
                        partial_value = next_partial_sums[idx*32 +: 32];
                        biased_value = partial_value + bias_tile[idx*32 +: 32];
                        scaled_value = biased_value * quant_scale;
                        if (quant_shift != 0) begin
                            quantized_value = (scaled_value + (64'sd1 <<< (quant_shift - 1'b1))) >>> quant_shift;
                        end else begin
                            quantized_value = scaled_value[31:0];
                        end
                        quantized_value = quantized_value + output_zp;
                        if (relu_enable && quantized_value < output_zp) begin
                            quantized_value = output_zp;
                        end
                        ofm_write_data[idx*8 +: 8] <= clamp_i8(quantized_value);
                    end
                    ofm_valid <= 1'b1;
                    if (frame_last) begin
                        busy    <= 1'b0;
                        ap_done <= 1'b1;
                        ap_idle <= 1'b1;
                    end
                end else begin
                    accum_write_en   <= 1'b1;
                    accum_write_data <= next_partial_sums;
                end
            end
        end
    end

endmodule
