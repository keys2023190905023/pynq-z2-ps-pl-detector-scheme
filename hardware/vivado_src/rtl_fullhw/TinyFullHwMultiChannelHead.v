`timescale 1ns / 1ps

// A tiny folded multi-channel 3x3 detection head.
//
// Input tensor layout on s_axis:
//   channel-major, raster-scan within each channel
// Output record layout on m_record:
//   row-major cells, one record per cell
//   record bytes = [x, y, w, h, obj, cls0, cls1, ...]
//
// The arithmetic is intentionally tiny and deterministic so the design stays
// easy to verify in RTL simulation while still exercising true channel folding
// and partial-sum accumulation.
module TinyFullHwMultiChannelHead #(
    parameter integer IMG_WIDTH        = 4,
    parameter integer IMG_HEIGHT       = 4,
    parameter integer IN_CHANNELS      = 2,
    parameter integer NUM_ANCHORS      = 1,
    parameter integer BOX_PARAMS       = 5,
    parameter integer NUM_CLASSES      = 3,
    parameter integer RAM_ADDR_WIDTH   = 16,
    parameter integer INPUT_RAM_DEPTH  = 65536,
    parameter integer OUTPUT_RAM_DEPTH = 65536
)(
    input  wire                                  clk,
    input  wire                                  rst_n,
    input  wire                                  ap_start,
    output reg                                   ap_done,
    output reg                                   ap_idle,
    input  wire signed [7:0]                     s_axis_tdata,
    input  wire                                  s_axis_tvalid,
    output wire                                  s_axis_tready,
    output wire [NUM_ANCHORS*(BOX_PARAMS+NUM_CLASSES)*8-1:0] m_record_tdata,
    output wire                                  m_record_tvalid,
    input  wire                                  m_record_tready,
    output wire                                  m_record_tlast
);
    localparam integer HEAD_CHANNELS = NUM_ANCHORS * (BOX_PARAMS + NUM_CLASSES);
    localparam integer INPUT_PIXELS  = IMG_WIDTH * IMG_HEIGHT;
    localparam integer OUTPUT_WIDTH  = (IMG_WIDTH  >= 3) ? (IMG_WIDTH  - 2) : 0;
    localparam integer OUTPUT_HEIGHT = (IMG_HEIGHT >= 3) ? (IMG_HEIGHT - 2) : 0;
    localparam integer OUTPUT_PIXELS = OUTPUT_WIDTH * OUTPUT_HEIGHT;
    localparam integer INPUT_VALUES  = INPUT_PIXELS * IN_CHANNELS;

    localparam [2:0] ST_IDLE   = 3'd0;
    localparam [2:0] ST_LOAD   = 3'd1;
    localparam [2:0] ST_KICK   = 3'd2;
    localparam [2:0] ST_RUN    = 3'd3;
    localparam [2:0] ST_STREAM = 3'd4;

    reg [2:0] state;
    reg ap_start_d1;
    wire start_edge = ap_start && !ap_start_d1;

    reg [RAM_ADDR_WIDTH-1:0] load_addr;
    reg [15:0] current_channel;
    reg [31:0] pixel_feed_count;
    reg [31:0] issue_count;
    reg [31:0] output_count;
    reg [31:0] stream_record_addr;
    reg [HEAD_CHANNELS*8-1:0] stream_record_data;
    reg stream_record_valid;
    reg conv_ap_start;

    wire run_active = (state == ST_RUN);
    wire feeding_pixels = run_active && (pixel_feed_count < INPUT_PIXELS);
    wire current_channel_first = (current_channel == 0);
    wire current_channel_last = (current_channel == IN_CHANNELS - 1);
    wire [RAM_ADDR_WIDTH-1:0] input_base_addr = current_channel * INPUT_PIXELS;
    wire [RAM_ADDR_WIDTH-1:0] input_rd_addr = input_base_addr + pixel_feed_count[RAM_ADDR_WIDTH-1:0];

    wire signed [7:0] input_rd_data;
    wire signed [HEAD_CHANNELS*32-1:0] accum_rd_data;
    wire [HEAD_CHANNELS*8-1:0] output_rd_data;

    wire conv_ap_done;
    wire conv_ap_idle;
    wire conv_accum_write_en;
    wire signed [HEAD_CHANNELS*32-1:0] conv_accum_write_data;
    wire conv_ofm_valid;
    wire signed [HEAD_CHANNELS*8-1:0] conv_ofm_write_data;
    wire conv_window_valid_dbg;

    integer idx;
    reg [HEAD_CHANNELS*8-1:0] output_wr_data;
    reg output_wr_en;
    reg [RAM_ADDR_WIDTH-1:0] output_wr_addr;

    function signed [7:0] head_center_coeff;
        input integer out_channel;
        input integer in_channel;
        begin
            case (out_channel)
                0: head_center_coeff = (in_channel == 0) ? 8'sd1 : 8'sd0;   // x
                1: head_center_coeff = (in_channel == 0) ? 8'sd0 : 8'sd1;   // y
                2: head_center_coeff = 8'sd1;                                // w
                3: head_center_coeff = (in_channel == 0) ? 8'sd1 : 8'sd2;   // h
                4: head_center_coeff = (in_channel == 0) ? -8'sd1 : 8'sd1;  // obj
                5: head_center_coeff = (in_channel == 0) ? 8'sd1 : -8'sd1;  // cls0
                6: head_center_coeff = (in_channel == 0) ? 8'sd0 : 8'sd2;   // cls1
                7: head_center_coeff = (in_channel == 0) ? 8'sd2 : 8'sd0;   // cls2
                default: head_center_coeff = 8'sd0;
            endcase
        end
    endfunction

    function signed [31:0] head_bias_value;
        input integer out_channel;
        begin
            case (out_channel)
                0: head_bias_value = 32'sd10;
                1: head_bias_value = 32'sd20;
                2: head_bias_value = 32'sd0;
                3: head_bias_value = -32'sd10;
                4: head_bias_value = 32'sd64;
                5: head_bias_value = 32'sd32;
                6: head_bias_value = 32'sd1;
                7: head_bias_value = 32'sd2;
                default: head_bias_value = 32'sd0;
            endcase
        end
    endfunction

    function [HEAD_CHANNELS*72-1:0] build_weight_tile;
        input integer in_channel;
        integer out_channel;
        reg signed [7:0] coeff;
        begin
            build_weight_tile = {HEAD_CHANNELS*72{1'b0}};
            for (out_channel = 0; out_channel < HEAD_CHANNELS; out_channel = out_channel + 1) begin
                coeff = head_center_coeff(out_channel, in_channel);
                build_weight_tile[out_channel*72 +: 72] = {
                    8'sd0, 8'sd0, 8'sd0,
                    8'sd0, coeff, 8'sd0,
                    8'sd0, 8'sd0, 8'sd0
                };
            end
        end
    endfunction

    function [HEAD_CHANNELS*32-1:0] build_bias_tile;
        input integer unused_select;
        integer out_channel;
        begin
            build_bias_tile = {HEAD_CHANNELS*32{1'b0}};
            for (out_channel = 0; out_channel < HEAD_CHANNELS; out_channel = out_channel + 1) begin
                build_bias_tile[out_channel*32 +: 32] = head_bias_value(out_channel);
            end
        end
    endfunction

    wire [HEAD_CHANNELS*72-1:0] weight_tile = build_weight_tile(current_channel);
    wire [HEAD_CHANNELS*32-1:0] bias_tile = build_bias_tile(0);

    assign s_axis_tready = (state == ST_LOAD);
    assign m_record_tdata = stream_record_data;
    assign m_record_tvalid = (state == ST_STREAM) && stream_record_valid;
    assign m_record_tlast = (state == ST_STREAM) && stream_record_valid && (stream_record_addr == OUTPUT_PIXELS - 1);

    FeatureMapDualPortRam #(
        .DATA_WIDTH(8),
        .ADDR_WIDTH(RAM_ADDR_WIDTH),
        .DEPTH(INPUT_RAM_DEPTH)
    ) u_input_ram (
        .clk(clk),
        .wr_en((state == ST_LOAD) && s_axis_tvalid && s_axis_tready),
        .wr_addr(load_addr),
        .wr_data(s_axis_tdata),
        .rd_addr_a(input_rd_addr),
        .rd_data_a(input_rd_data),
        .rd_addr_b({RAM_ADDR_WIDTH{1'b0}}),
        .rd_data_b()
    );

    FeatureMapDualPortRam #(
        .DATA_WIDTH(HEAD_CHANNELS*32),
        .ADDR_WIDTH(RAM_ADDR_WIDTH),
        .DEPTH(OUTPUT_RAM_DEPTH)
    ) u_accum_ram (
        .clk(clk),
        .wr_en((state == ST_RUN) && conv_accum_write_en),
        .wr_addr(output_count[RAM_ADDR_WIDTH-1:0]),
        .wr_data(conv_accum_write_data),
        .rd_addr_a(issue_count[RAM_ADDR_WIDTH-1:0]),
        .rd_data_a(accum_rd_data),
        .rd_addr_b({RAM_ADDR_WIDTH{1'b0}}),
        .rd_data_b()
    );

    FeatureMapDualPortRam #(
        .DATA_WIDTH(HEAD_CHANNELS*8),
        .ADDR_WIDTH(RAM_ADDR_WIDTH),
        .DEPTH(OUTPUT_RAM_DEPTH)
    ) u_output_ram (
        .clk(clk),
        .wr_en(output_wr_en),
        .wr_addr(output_wr_addr),
        .wr_data(output_wr_data),
        .rd_addr_a(stream_record_addr[RAM_ADDR_WIDTH-1:0]),
        .rd_data_a(output_rd_data),
        .rd_addr_b({RAM_ADDR_WIDTH{1'b0}}),
        .rd_data_b()
    );

    PlPsConvChannelStreamTop #(
        .OUT_CHANNELS_PAR(HEAD_CHANNELS),
        .MAX_IMG_WIDTH(IMG_WIDTH)
    ) u_detect_head (
        .clk(clk),
        .rst_n(rst_n),
        .ap_start(conv_ap_start),
        .frame_start(feeding_pixels && (pixel_feed_count == 0)),
        .frame_last(feeding_pixels && (pixel_feed_count == INPUT_PIXELS - 1)),
        .pixel_valid(feeding_pixels),
        .img_width(IMG_WIDTH[15:0]),
        .channel_first(current_channel_first),
        .channel_last(current_channel_last),
        .relu_enable(1'b0),
        .pixel_in(input_rd_data),
        .weight_tile(weight_tile),
        .bias_tile(bias_tile),
        .accum_read_data(accum_rd_data),
        .quant_scale(16'sd1),
        .quant_shift(5'd0),
        .output_zp(8'sd0),
        .ap_done(conv_ap_done),
        .ap_idle(conv_ap_idle),
        .accum_write_en(conv_accum_write_en),
        .accum_write_data(conv_accum_write_data),
        .ofm_valid(conv_ofm_valid),
        .ofm_write_data(conv_ofm_write_data),
        .window_valid_dbg(conv_window_valid_dbg)
    );

    always @(*) begin
        output_wr_en = 1'b0;
        output_wr_addr = output_count[RAM_ADDR_WIDTH-1:0];
        output_wr_data = {HEAD_CHANNELS*8{1'b0}};
        if ((state == ST_RUN) && conv_ofm_valid) begin
            output_wr_en = 1'b1;
            output_wr_addr = output_count[RAM_ADDR_WIDTH-1:0];
            output_wr_data = conv_ofm_write_data;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            ap_start_d1 <= 1'b0;
            ap_done <= 1'b0;
            ap_idle <= 1'b1;
            load_addr <= {RAM_ADDR_WIDTH{1'b0}};
            current_channel <= 16'd0;
            pixel_feed_count <= 32'd0;
            issue_count <= 32'd0;
            output_count <= 32'd0;
            stream_record_addr <= 32'd0;
            stream_record_data <= {HEAD_CHANNELS*8{1'b0}};
            stream_record_valid <= 1'b0;
            conv_ap_start <= 1'b0;
        end else begin
            ap_start_d1 <= ap_start;
            ap_done <= 1'b0;
            conv_ap_start <= 1'b0;

            case (state)
                ST_IDLE: begin
                    ap_idle <= 1'b1;
                    load_addr <= {RAM_ADDR_WIDTH{1'b0}};
                    current_channel <= 16'd0;
                    pixel_feed_count <= 32'd0;
                    issue_count <= 32'd0;
                    output_count <= 32'd0;
                    stream_record_addr <= 32'd0;
                    stream_record_valid <= 1'b0;
                    if (start_edge) begin
                        ap_idle <= 1'b0;
                        state <= ST_LOAD;
                    end
                end

                ST_LOAD: begin
                    if (s_axis_tvalid && s_axis_tready) begin
                        if (load_addr == INPUT_VALUES - 1) begin
                            load_addr <= {RAM_ADDR_WIDTH{1'b0}};
                            current_channel <= 16'd0;
                            pixel_feed_count <= 32'd0;
                            issue_count <= 32'd0;
                            output_count <= 32'd0;
                            stream_record_valid <= 1'b0;
                            state <= ST_KICK;
                        end else begin
                            load_addr <= load_addr + 1'b1;
                        end
                    end
                end

                ST_KICK: begin
                    conv_ap_start <= 1'b1;
                    state <= ST_RUN;
                end

                ST_RUN: begin
                    if (pixel_feed_count < INPUT_PIXELS) begin
                        pixel_feed_count <= pixel_feed_count + 1'b1;
                    end

                    if (feeding_pixels && conv_window_valid_dbg) begin
                        if (issue_count == OUTPUT_PIXELS - 1) begin
                            issue_count <= 32'd0;
                        end else begin
                            issue_count <= issue_count + 1'b1;
                        end
                    end

                    if (conv_accum_write_en || conv_ofm_valid) begin
                        if (output_count == OUTPUT_PIXELS - 1) begin
                            issue_count <= 32'd0;
                            output_count <= 32'd0;
                            pixel_feed_count <= 32'd0;
                            if (current_channel_last) begin
                                stream_record_addr <= 32'd0;
                                stream_record_valid <= 1'b0;
                                state <= ST_STREAM;
                            end else begin
                                current_channel <= current_channel + 1'b1;
                                state <= ST_KICK;
                            end
                        end else begin
                            output_count <= output_count + 1'b1;
                        end
                    end
                end

                ST_STREAM: begin
                    if (!stream_record_valid) begin
                        stream_record_data <= output_rd_data;
                        stream_record_valid <= 1'b1;
                    end else if (m_record_tready) begin
                        if (stream_record_addr == OUTPUT_PIXELS - 1) begin
                            ap_done <= 1'b1;
                            ap_idle <= 1'b1;
                            stream_record_valid <= 1'b0;
                            state <= ST_IDLE;
                        end else begin
                            stream_record_addr <= stream_record_addr + 1'b1;
                            stream_record_valid <= 1'b0;
                        end
                    end
                end

                default: begin
                    state <= ST_IDLE;
                    ap_idle <= 1'b1;
                end
            endcase
        end
    end

endmodule
