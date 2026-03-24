`timescale 1ns / 1ps

module YOLOv8_Top #(
    parameter DATA_WIDTH = 8,
    parameter IMG_WIDTH  = 640,
    parameter MAX_OUTPUT_PIXELS = 8192
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  ap_start,
    output reg                   ap_done,
    output reg                   ap_idle,
    input  wire [15:0]           img_w,
    input  wire [15:0]           img_h,
    input  wire [15:0]           total_channels,
    input  wire signed [31:0]    layer_bias,
    input  wire [4:0]            quant_shift,
    input  wire signed [15:0]    quant_scale,
    input  wire signed [7:0]     output_zp,

    output wire [31:0]           ifm_read_addr,
    output wire                  ifm_read_en,
    input  wire [DATA_WIDTH-1:0] ifm_read_data,

    input  wire signed [7:0]     w_00, w_01, w_02,
    input  wire signed [7:0]     w_10, w_11, w_12,
    input  wire signed [7:0]     w_20, w_21, w_22,

    output wire [31:0]           ofm_write_addr,
    output wire                  ofm_write_en,
    output wire signed [7:0]     ofm_write_data
);
    wire channel_first = total_channels[0];
    wire channel_last  = total_channels[1];
    wire relu_enable   = total_channels[2];

    wire [31:0] total_input_pixels  = img_w * img_h;
    wire [31:0] output_width        = (img_w >= 16'd2) ? (img_w - 16'd2) : 32'd0;
    wire [31:0] output_height       = (img_h >= 16'd2) ? (img_h - 16'd2) : 32'd0;
    wire [31:0] total_output_pixels = output_width * output_height;

    reg busy;
    reg ap_start_d1;
    reg inner_start;
    reg frame_init_pending;
    reg [31:0] input_pixel_counter;
    reg [31:0] window_counter;

    (* ram_style = "distributed" *) reg signed [31:0] accum_mem [0:MAX_OUTPUT_PIXELS-1];
    wire signed [31:0] accum_read_word =
        (window_counter < MAX_OUTPUT_PIXELS) ? accum_mem[window_counter] : 32'sd0;

    wire run_pixel = busy && !frame_init_pending && (input_pixel_counter < total_input_pixels);
    wire frame_start = busy && frame_init_pending;

    wire inner_done;
    wire inner_idle;
    wire inner_accum_write_en;
    wire signed [31:0] inner_accum_write_data;
    wire inner_ofm_valid;
    wire signed [7:0] inner_ofm_write_data;
    wire valid_window_event = inner_accum_write_en || inner_ofm_valid;

    wire signed [71:0] weight_tile = {
        w_22, w_21, w_20,
        w_12, w_11, w_10,
        w_02, w_01, w_00
    };

    assign ifm_read_en = run_pixel;
    assign ifm_read_addr = input_pixel_counter;
    assign ofm_write_addr = window_counter;
    assign ofm_write_en = inner_ofm_valid;
    assign ofm_write_data = inner_ofm_write_data;

    PlPsConvChannelStreamTop #(
        .OUT_CHANNELS_PAR(1),
        .MAX_IMG_WIDTH(IMG_WIDTH)
    ) u_stream_operator (
        .clk(clk),
        .rst_n(rst_n),
        .ap_start(inner_start),
        .frame_start(frame_start),
        .frame_last(1'b0),
        .pixel_valid(run_pixel),
        .img_width(img_w),
        .channel_first(channel_first),
        .channel_last(channel_last),
        .relu_enable(relu_enable),
        .pixel_in(ifm_read_data),
        .weight_tile(weight_tile),
        .bias_tile(layer_bias),
        .accum_read_data(accum_read_word),
        .quant_scale(quant_scale),
        .quant_shift(quant_shift),
        .output_zp(output_zp),
        .ap_done(inner_done),
        .ap_idle(inner_idle),
        .accum_write_en(inner_accum_write_en),
        .accum_write_data(inner_accum_write_data),
        .ofm_valid(inner_ofm_valid),
        .ofm_write_data(inner_ofm_write_data),
        .window_valid_dbg()
    );

    always @(posedge clk) begin
        if (busy && valid_window_event && !channel_last && (window_counter < MAX_OUTPUT_PIXELS)) begin
            accum_mem[window_counter] <= inner_accum_write_data;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;
            ap_done <= 1'b0;
            ap_idle <= 1'b1;
            ap_start_d1 <= 1'b0;
            inner_start <= 1'b0;
            frame_init_pending <= 1'b0;
            input_pixel_counter <= 32'd0;
            window_counter <= 32'd0;
        end else begin
            ap_done <= 1'b0;
            inner_start <= 1'b0;
            ap_start_d1 <= ap_start;

            if (!busy && ap_start && !ap_start_d1) begin
                busy <= 1'b1;
                ap_idle <= 1'b0;
                inner_start <= 1'b1;
                frame_init_pending <= 1'b1;
                input_pixel_counter <= 32'd0;
                window_counter <= 32'd0;
            end else if (busy) begin
                if (frame_init_pending) begin
                    frame_init_pending <= 1'b0;
                end else if (run_pixel) begin
                    input_pixel_counter <= input_pixel_counter + 32'd1;
                end

                if (valid_window_event) begin
                    if (window_counter + 32'd1 >= total_output_pixels) begin
                        busy <= 1'b0;
                        ap_idle <= 1'b1;
                        ap_done <= 1'b1;
                        frame_init_pending <= 1'b0;
                        window_counter <= 32'd0;
                    end else begin
                        window_counter <= window_counter + 32'd1;
                    end
                end
            end
        end
    end

endmodule
