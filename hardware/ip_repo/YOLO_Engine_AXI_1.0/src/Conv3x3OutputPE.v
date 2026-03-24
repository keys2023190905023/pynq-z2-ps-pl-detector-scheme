`timescale 1ns / 1ps

module Conv3x3OutputPE (
    input  wire signed [71:0] act_window,
    input  wire signed [71:0] weight_window,
    input  wire signed [31:0] partial_sum_in,
    output wire signed [31:0] partial_sum_out
);
    wire signed [7:0] a00 = act_window[7:0];
    wire signed [7:0] a01 = act_window[15:8];
    wire signed [7:0] a02 = act_window[23:16];
    wire signed [7:0] a10 = act_window[31:24];
    wire signed [7:0] a11 = act_window[39:32];
    wire signed [7:0] a12 = act_window[47:40];
    wire signed [7:0] a20 = act_window[55:48];
    wire signed [7:0] a21 = act_window[63:56];
    wire signed [7:0] a22 = act_window[71:64];

    wire signed [7:0] w00 = weight_window[7:0];
    wire signed [7:0] w01 = weight_window[15:8];
    wire signed [7:0] w02 = weight_window[23:16];
    wire signed [7:0] w10 = weight_window[31:24];
    wire signed [7:0] w11 = weight_window[39:32];
    wire signed [7:0] w12 = weight_window[47:40];
    wire signed [7:0] w20 = weight_window[55:48];
    wire signed [7:0] w21 = weight_window[63:56];
    wire signed [7:0] w22 = weight_window[71:64];

    wire signed [31:0] mac_sum =
        ($signed(a00) * $signed(w00)) +
        ($signed(a01) * $signed(w01)) +
        ($signed(a02) * $signed(w02)) +
        ($signed(a10) * $signed(w10)) +
        ($signed(a11) * $signed(w11)) +
        ($signed(a12) * $signed(w12)) +
        ($signed(a20) * $signed(w20)) +
        ($signed(a21) * $signed(w21)) +
        ($signed(a22) * $signed(w22));

    assign partial_sum_out = partial_sum_in + mac_sum;

endmodule
