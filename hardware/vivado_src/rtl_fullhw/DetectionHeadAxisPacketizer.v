`timescale 1ns / 1ps

// Serializes fixed-width detection records into a byte AXI stream with a
// compact detector-oriented header:
//   DET1 | grid_w | grid_h | anchors | box_params | num_classes | record_bytes
module DetectionHeadAxisPacketizer #(
    parameter integer RECORD_BYTES = 8,
    parameter integer HEADER_BYTES = 16
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire [15:0]                   grid_width,
    input  wire [15:0]                   grid_height,
    input  wire [15:0]                   num_anchors,
    input  wire [15:0]                   box_params,
    input  wire [15:0]                   num_classes,
    input  wire [15:0]                   record_bytes,
    input  wire [RECORD_BYTES*8-1:0]     s_record_tdata,
    input  wire                          s_record_tvalid,
    output wire                          s_record_tready,
    input  wire                          s_record_tlast,
    output reg  [7:0]                    m_axis_tdata,
    output reg                           m_axis_tvalid,
    input  wire                          m_axis_tready,
    output reg                           m_axis_tlast
);
    localparam [1:0] ST_IDLE   = 2'd0;
    localparam [1:0] ST_HEADER = 2'd1;
    localparam [1:0] ST_RECORD = 2'd2;

    reg [1:0] state;
    reg [4:0] header_index;
    reg [4:0] record_index;
    reg [RECORD_BYTES*8-1:0] record_buffer;
    reg record_buffer_valid;
    reg record_buffer_last;

    assign s_record_tready = (state == ST_RECORD) && !record_buffer_valid;

    function [7:0] header_byte;
        input [4:0] index;
        begin
            case (index)
                5'd0:  header_byte = 8'h44; // D
                5'd1:  header_byte = 8'h45; // E
                5'd2:  header_byte = 8'h54; // T
                5'd3:  header_byte = 8'h31; // 1
                5'd4:  header_byte = grid_width[7:0];
                5'd5:  header_byte = grid_width[15:8];
                5'd6:  header_byte = grid_height[7:0];
                5'd7:  header_byte = grid_height[15:8];
                5'd8:  header_byte = num_anchors[7:0];
                5'd9:  header_byte = num_anchors[15:8];
                5'd10: header_byte = box_params[7:0];
                5'd11: header_byte = box_params[15:8];
                5'd12: header_byte = num_classes[7:0];
                5'd13: header_byte = num_classes[15:8];
                5'd14: header_byte = record_bytes[7:0];
                5'd15: header_byte = record_bytes[15:8];
                default: header_byte = 8'h00;
            endcase
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            header_index <= 5'd0;
            record_index <= 5'd0;
            record_buffer <= {RECORD_BYTES*8{1'b0}};
            record_buffer_valid <= 1'b0;
            record_buffer_last <= 1'b0;
            m_axis_tdata <= 8'h00;
            m_axis_tvalid <= 1'b0;
            m_axis_tlast <= 1'b0;
        end else begin
            m_axis_tvalid <= 1'b0;
            m_axis_tlast <= 1'b0;

            if (s_record_tvalid && s_record_tready) begin
                record_buffer <= s_record_tdata;
                record_buffer_valid <= 1'b1;
                record_buffer_last <= s_record_tlast;
                record_index <= 5'd0;
            end

            case (state)
                ST_IDLE: begin
                    header_index <= 5'd0;
                    record_index <= 5'd0;
                    record_buffer_valid <= 1'b0;
                    if (s_record_tvalid) begin
                        state <= ST_HEADER;
                    end
                end

                ST_HEADER: begin
                    m_axis_tdata <= header_byte(header_index);
                    m_axis_tvalid <= 1'b1;
                    if (m_axis_tready) begin
                        if (header_index == HEADER_BYTES - 1) begin
                            state <= ST_RECORD;
                        end else begin
                            header_index <= header_index + 1'b1;
                        end
                    end
                end

                ST_RECORD: begin
                    if (record_buffer_valid) begin
                        m_axis_tdata <= record_buffer[record_index*8 +: 8];
                        m_axis_tvalid <= 1'b1;
                        m_axis_tlast <= record_buffer_last && (record_index == RECORD_BYTES - 1);
                        if (m_axis_tready) begin
                            if (record_index == RECORD_BYTES - 1) begin
                                record_index <= 5'd0;
                                record_buffer_valid <= 1'b0;
                                if (record_buffer_last) begin
                                    state <= ST_IDLE;
                                end
                            end else begin
                                record_index <= record_index + 1'b1;
                            end
                        end
                    end
                end

                default: begin
                    state <= ST_IDLE;
                end
            endcase
        end
    end

endmodule
