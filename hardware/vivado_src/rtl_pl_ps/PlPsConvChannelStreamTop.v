`timescale 1ns / 1ps

module PlPsConvChannelStreamTop #(
    parameter OUT_CHANNELS_PAR = 4,
    parameter MAX_IMG_WIDTH    = 640
)(
    input  wire                                     clk,
    input  wire                                     rst_n,
    input  wire                                     ap_start,
    input  wire                                     frame_start,
    input  wire                                     frame_last,
    input  wire                                     pixel_valid,
    input  wire [15:0]                              img_width,
    input  wire                                     channel_first,
    input  wire                                     channel_last,
    input  wire                                     relu_enable,
    input  wire signed [7:0]                        pixel_in,
    input  wire signed [OUT_CHANNELS_PAR*72-1:0]    weight_tile,
    input  wire signed [OUT_CHANNELS_PAR*32-1:0]    bias_tile,
    input  wire signed [OUT_CHANNELS_PAR*32-1:0]    accum_read_data,
    input  wire signed [15:0]                       quant_scale,
    input  wire [4:0]                               quant_shift,
    input  wire signed [7:0]                        output_zp,
    output wire                                     ap_done,
    output wire                                     ap_idle,
    output wire                                     accum_write_en,
    output wire signed [OUT_CHANNELS_PAR*32-1:0]    accum_write_data,
    output wire                                     ofm_valid,
    output wire signed [OUT_CHANNELS_PAR*8-1:0]     ofm_write_data,
    output wire                                     window_valid_dbg
);
    wire signed [71:0] act_window;
    wire window_valid;

    assign window_valid_dbg = window_valid;

    StreamLineBuffer3x3 #(
        .DATA_WIDTH(8),
        .MAX_IMG_WIDTH(MAX_IMG_WIDTH)
    ) u_line_buffer (
        .clk(clk),
        .rst_n(rst_n),
        .frame_start(frame_start),
        .pixel_valid(pixel_valid),
        .img_width(img_width),
        .pixel_in(pixel_in),
        .act_window(act_window),
        .window_valid(window_valid)
    );

    PlPsConvOperatorTop #(
        .OUT_CHANNELS_PAR(OUT_CHANNELS_PAR)
    ) u_operator (
        .clk(clk),
        .rst_n(rst_n),
        .ap_start(ap_start),
        .frame_last(frame_last),
        .pixel_valid(pixel_valid),
        .window_valid(window_valid),
        .channel_first(channel_first),
        .channel_last(channel_last),
        .relu_enable(relu_enable),
        .act_window(act_window),
        .weight_tile(weight_tile),
        .bias_tile(bias_tile),
        .accum_read_data(accum_read_data),
        .quant_scale(quant_scale),
        .quant_shift(quant_shift),
        .output_zp(output_zp),
        .ap_done(ap_done),
        .ap_idle(ap_idle),
        .accum_write_en(accum_write_en),
        .accum_write_data(accum_write_data),
        .ofm_valid(ofm_valid),
        .ofm_write_data(ofm_write_data)
    );

endmodule
