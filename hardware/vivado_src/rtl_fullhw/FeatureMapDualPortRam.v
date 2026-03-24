`timescale 1ns / 1ps

module FeatureMapDualPortRam #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 16,
    parameter DEPTH      = 65536
)(
    input  wire                             clk,
    input  wire                             wr_en,
    input  wire [ADDR_WIDTH-1:0]            wr_addr,
    input  wire signed [DATA_WIDTH-1:0]     wr_data,
    input  wire [ADDR_WIDTH-1:0]            rd_addr_a,
    output wire signed [DATA_WIDTH-1:0]     rd_data_a,
    input  wire [ADDR_WIDTH-1:0]            rd_addr_b,
    output wire signed [DATA_WIDTH-1:0]     rd_data_b
);
    (* ram_style = "block" *) reg signed [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (wr_en && (wr_addr < DEPTH)) begin
            mem[wr_addr] <= wr_data;
        end
    end

    assign rd_data_a = (rd_addr_a < DEPTH) ? mem[rd_addr_a] : {DATA_WIDTH{1'b0}};
    assign rd_data_b = (rd_addr_b < DEPTH) ? mem[rd_addr_b] : {DATA_WIDTH{1'b0}};

endmodule
