`timescale 1ns / 1ps

// PL-only demo detector:
//   RGB input -> PL stem feature stage -> PL detect head -> DET1 packet
//
// PS only needs to move the input frame, pulse ap_start and consume the final
// detector packet. Layer sequencing stays entirely inside PL.
module TinyFullHwPlOnlyDemoDetectorTop #(
    parameter integer IMG_WIDTH             = 8,
    parameter integer IMG_HEIGHT            = 8,
    parameter integer RAM_ADDR_WIDTH        = 16,
    parameter integer STEM_INPUT_RAM_DEPTH  = 65536,
    parameter integer STEM_OUTPUT_RAM_DEPTH = 65536,
    parameter integer HEAD_INPUT_RAM_DEPTH  = 65536,
    parameter integer HEAD_ACCUM_RAM_DEPTH  = 65536,
    parameter integer HEAD_OUTPUT_RAM_DEPTH = 65536
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  ap_start,
    output reg                   ap_done,
    output reg                   ap_idle,
    input  wire signed [7:0]     s_axis_tdata,
    input  wire                  s_axis_tvalid,
    output wire                  s_axis_tready,
    output wire signed [7:0]     m_axis_tdata,
    output wire                  m_axis_tvalid,
    input  wire                  m_axis_tready,
    output wire                  m_axis_tlast
);
    localparam integer STEM_OUT_WIDTH  = (IMG_WIDTH  >= 3) ? (IMG_WIDTH  - 2) : 0;
    localparam integer STEM_OUT_HEIGHT = (IMG_HEIGHT >= 3) ? (IMG_HEIGHT - 2) : 0;

    reg ap_start_d1;
    wire start_edge = ap_start && !ap_start_d1;

    reg stem_start;
    reg head_start;

    wire stem_ap_done;
    wire stem_ap_idle;
    wire stem_input_tready;
    wire signed [7:0] stem_tdata;
    wire stem_tvalid;
    wire head_input_tready;
    wire stem_tlast;

    wire head_ap_done;
    wire head_ap_idle;
    wire signed [7:0] m_axis_tdata_int;
    wire m_axis_tvalid_int;
    wire m_axis_tlast_int;

    assign s_axis_tready = stem_input_tready;
    assign m_axis_tdata = m_axis_tdata_int;
    assign m_axis_tvalid = m_axis_tvalid_int;
    assign m_axis_tlast = m_axis_tlast_int;

    TinyFullHwMultiChannelFeatureStage #(
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .IN_CHANNELS(3),
        .OUT_CHANNELS(4),
        .STAGE_KIND(0),
        .RAM_ADDR_WIDTH(RAM_ADDR_WIDTH),
        .INPUT_RAM_DEPTH(STEM_INPUT_RAM_DEPTH),
        .OUTPUT_RAM_DEPTH(STEM_OUTPUT_RAM_DEPTH)
    ) u_stem_stage (
        .clk(clk),
        .rst_n(rst_n),
        .ap_start(stem_start),
        .ap_done(stem_ap_done),
        .ap_idle(stem_ap_idle),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(stem_input_tready),
        .m_axis_tdata(stem_tdata),
        .m_axis_tvalid(stem_tvalid),
        .m_axis_tready(head_input_tready),
        .m_axis_tlast(stem_tlast)
    );

    TinyFullHwDetectorTop #(
        .IMG_WIDTH(STEM_OUT_WIDTH),
        .IMG_HEIGHT(STEM_OUT_HEIGHT),
        .IN_CHANNELS(4),
        .NUM_ANCHORS(1),
        .BOX_PARAMS(5),
        .NUM_CLASSES(3),
        .RAM_ADDR_WIDTH(RAM_ADDR_WIDTH),
        .INPUT_RAM_DEPTH(HEAD_INPUT_RAM_DEPTH),
        .OUTPUT_RAM_DEPTH(HEAD_OUTPUT_RAM_DEPTH)
    ) u_detect_head (
        .clk(clk),
        .rst_n(rst_n),
        .ap_start(head_start),
        .ap_done(head_ap_done),
        .ap_idle(head_ap_idle),
        .s_axis_tdata(stem_tdata),
        .s_axis_tvalid(stem_tvalid),
        .s_axis_tready(head_input_tready),
        .m_axis_tdata(m_axis_tdata_int),
        .m_axis_tvalid(m_axis_tvalid_int),
        .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast_int)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ap_start_d1 <= 1'b0;
            ap_done <= 1'b0;
            ap_idle <= 1'b1;
            stem_start <= 1'b0;
            head_start <= 1'b0;
        end else begin
            ap_start_d1 <= ap_start;
            ap_done <= 1'b0;
            stem_start <= 1'b0;
            head_start <= 1'b0;

            if (start_edge) begin
                ap_idle <= 1'b0;
                stem_start <= 1'b1;
                head_start <= 1'b1;
            end

            if (m_axis_tvalid_int && m_axis_tready && m_axis_tlast_int) begin
                ap_done <= 1'b1;
                ap_idle <= 1'b1;
            end
        end
    end

endmodule
