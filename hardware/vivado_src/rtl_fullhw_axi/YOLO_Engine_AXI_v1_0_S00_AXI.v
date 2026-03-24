`timescale 1 ns / 1 ps

`include "FeatureMapDualPortRam.v"
`include "DetectionHeadAxisPacketizer.v"
`include "TinyFullHwMultiChannelHead.v"
`include "TinyFullHwDetectorTop.v"

module YOLO_Engine_AXI_v1_0_S00_AXI #
(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 5
)
(
    input  wire [7:0] S_AXIS_TDATA,
    input  wire       S_AXIS_TVALID,
    output wire       S_AXIS_TREADY,
    output wire       M_AXIS_TKEEP,
    output wire       M_AXIS_TLAST,
    output wire [7:0] M_AXIS_TDATA,
    output wire       M_AXIS_TVALID,
    input  wire       M_AXIS_TREADY,

    input  wire                              S_AXI_ACLK,
    input  wire                              S_AXI_ARESETN,
    input  wire [C_S_AXI_ADDR_WIDTH-1 : 0]   S_AXI_AWADDR,
    input  wire [2 : 0]                      S_AXI_AWPROT,
    input  wire                              S_AXI_AWVALID,
    output wire                              S_AXI_AWREADY,
    input  wire [C_S_AXI_DATA_WIDTH-1 : 0]   S_AXI_WDATA,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0] S_AXI_WSTRB,
    input  wire                              S_AXI_WVALID,
    output wire                              S_AXI_WREADY,
    output wire [1 : 0]                      S_AXI_BRESP,
    output wire                              S_AXI_BVALID,
    input  wire                              S_AXI_BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1 : 0]   S_AXI_ARADDR,
    input  wire [2 : 0]                      S_AXI_ARPROT,
    input  wire                              S_AXI_ARVALID,
    output wire                              S_AXI_ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1 : 0]   S_AXI_RDATA,
    output wire [1 : 0]                      S_AXI_RRESP,
    output wire                              S_AXI_RVALID,
    input  wire                              S_AXI_RREADY
);
    localparam integer ADDR_LSB = (C_S_AXI_DATA_WIDTH / 32) + 1;
    localparam integer OPT_MEM_ADDR_BITS = 2;
    localparam integer IMG_WIDTH = 4;
    localparam integer IMG_HEIGHT = 4;
    localparam integer IN_CHANNELS = 2;
    localparam integer NUM_ANCHORS = 1;
    localparam integer BOX_PARAMS = 5;
    localparam integer NUM_CLASSES = 3;
    localparam integer RECORD_BYTES = NUM_ANCHORS * (BOX_PARAMS + NUM_CLASSES);
    localparam integer INPUT_BYTES = IMG_WIDTH * IMG_HEIGHT * IN_CHANNELS;
    localparam integer OUTPUT_BYTES = 16 + ((IMG_WIDTH - 2) * (IMG_HEIGHT - 2) * RECORD_BYTES);

    reg [C_S_AXI_ADDR_WIDTH-1 : 0] axi_awaddr;
    reg axi_awready;
    reg axi_wready;
    reg [1 : 0] axi_bresp;
    reg axi_bvalid;
    reg [C_S_AXI_ADDR_WIDTH-1 : 0] axi_araddr;
    reg axi_arready;
    reg [C_S_AXI_DATA_WIDTH-1 : 0] axi_rdata;
    reg [1 : 0] axi_rresp;
    reg axi_rvalid;
    reg aw_en;

    reg [31:0] slv_reg0;
    reg [31:0] slv_reg1;
    reg [31:0] slv_reg2;
    reg [31:0] slv_reg3;
    reg [31:0] slv_reg4;
    reg [31:0] slv_reg5;
    reg [31:0] slv_reg6;
    reg [31:0] slv_reg7;

    wire slv_reg_wren;
    wire slv_reg_rden;
    reg [31:0] reg_data_out;
    integer byte_index;

    wire soft_reset = slv_reg0[1];
    wire detector_resetn = S_AXI_ARESETN & ~soft_reset;
    reg start_d1;
    wire start_rise = slv_reg0[0] && !start_d1;

    wire detector_ap_done;
    wire detector_ap_idle;

    reg [31:0] in_cnt;
    reg [31:0] out_cnt;
    reg done_seen;

    assign S_AXI_AWREADY = axi_awready;
    assign S_AXI_WREADY = axi_wready;
    assign S_AXI_BRESP = axi_bresp;
    assign S_AXI_BVALID = axi_bvalid;
    assign S_AXI_ARREADY = axi_arready;
    assign S_AXI_RDATA = axi_rdata;
    assign S_AXI_RRESP = axi_rresp;
    assign S_AXI_RVALID = axi_rvalid;

    assign M_AXIS_TKEEP = 1'b1;

    TinyFullHwDetectorTop #(
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .IN_CHANNELS(IN_CHANNELS),
        .NUM_ANCHORS(NUM_ANCHORS),
        .BOX_PARAMS(BOX_PARAMS),
        .NUM_CLASSES(NUM_CLASSES),
        .RAM_ADDR_WIDTH(8),
        .INPUT_RAM_DEPTH(256),
        .OUTPUT_RAM_DEPTH(256)
    ) u_fullhw_detector (
        .clk(S_AXI_ACLK),
        .rst_n(detector_resetn),
        .ap_start(slv_reg0[0]),
        .ap_done(detector_ap_done),
        .ap_idle(detector_ap_idle),
        .s_axis_tdata(S_AXIS_TDATA),
        .s_axis_tvalid(S_AXIS_TVALID),
        .s_axis_tready(S_AXIS_TREADY),
        .m_axis_tdata(M_AXIS_TDATA),
        .m_axis_tvalid(M_AXIS_TVALID),
        .m_axis_tready(M_AXIS_TREADY),
        .m_axis_tlast(M_AXIS_TLAST)
    );

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN || soft_reset) begin
            start_d1 <= 1'b0;
            in_cnt <= 32'd0;
            out_cnt <= 32'd0;
            done_seen <= 1'b0;
        end else begin
            start_d1 <= slv_reg0[0];
            if (start_rise) begin
                in_cnt <= 32'd0;
                out_cnt <= 32'd0;
                done_seen <= 1'b0;
            end else begin
                if (S_AXIS_TVALID && S_AXIS_TREADY) begin
                    in_cnt <= in_cnt + 32'd1;
                end
                if (M_AXIS_TVALID && M_AXIS_TREADY) begin
                    out_cnt <= out_cnt + 32'd1;
                end
                if (detector_ap_done) begin
                    done_seen <= 1'b1;
                end
            end
        end
    end

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_awready <= 1'b0;
            aw_en <= 1'b1;
        end else if (!axi_awready && S_AXI_AWVALID && S_AXI_WVALID && aw_en) begin
            axi_awready <= 1'b1;
            aw_en <= 1'b0;
        end else if (S_AXI_BREADY && axi_bvalid) begin
            aw_en <= 1'b1;
            axi_awready <= 1'b0;
        end else begin
            axi_awready <= 1'b0;
        end
    end

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_awaddr <= {C_S_AXI_ADDR_WIDTH{1'b0}};
        end else if (!axi_awready && S_AXI_AWVALID && S_AXI_WVALID && aw_en) begin
            axi_awaddr <= S_AXI_AWADDR;
        end
    end

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_wready <= 1'b0;
        end else if (!axi_wready && S_AXI_WVALID && S_AXI_AWVALID && aw_en) begin
            axi_wready <= 1'b1;
        end else begin
            axi_wready <= 1'b0;
        end
    end

    assign slv_reg_wren = axi_wready && S_AXI_WVALID && axi_awready && S_AXI_AWVALID;

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            slv_reg0 <= 32'd0;
            slv_reg1 <= 32'd0;
            slv_reg2 <= 32'd0;
            slv_reg3 <= 32'd0;
            slv_reg4 <= 32'd0;
            slv_reg5 <= 32'd0;
            slv_reg6 <= 32'd0;
            slv_reg7 <= 32'd0;
        end else if (slv_reg_wren) begin
            case (axi_awaddr[ADDR_LSB + OPT_MEM_ADDR_BITS : ADDR_LSB])
                3'h0: for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
                    if (S_AXI_WSTRB[byte_index]) slv_reg0[(byte_index*8) +: 8] <= S_AXI_WDATA[(byte_index*8) +: 8];
                3'h1: for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
                    if (S_AXI_WSTRB[byte_index]) slv_reg1[(byte_index*8) +: 8] <= S_AXI_WDATA[(byte_index*8) +: 8];
                3'h2: for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
                    if (S_AXI_WSTRB[byte_index]) slv_reg2[(byte_index*8) +: 8] <= S_AXI_WDATA[(byte_index*8) +: 8];
                3'h3: for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
                    if (S_AXI_WSTRB[byte_index]) slv_reg3[(byte_index*8) +: 8] <= S_AXI_WDATA[(byte_index*8) +: 8];
                3'h4: for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
                    if (S_AXI_WSTRB[byte_index]) slv_reg4[(byte_index*8) +: 8] <= S_AXI_WDATA[(byte_index*8) +: 8];
                3'h5: for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
                    if (S_AXI_WSTRB[byte_index]) slv_reg5[(byte_index*8) +: 8] <= S_AXI_WDATA[(byte_index*8) +: 8];
                3'h6: for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
                    if (S_AXI_WSTRB[byte_index]) slv_reg6[(byte_index*8) +: 8] <= S_AXI_WDATA[(byte_index*8) +: 8];
                3'h7: for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
                    if (S_AXI_WSTRB[byte_index]) slv_reg7[(byte_index*8) +: 8] <= S_AXI_WDATA[(byte_index*8) +: 8];
                default: begin end
            endcase
        end
    end

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_bvalid <= 1'b0;
            axi_bresp <= 2'b00;
        end else if (axi_awready && S_AXI_AWVALID && !axi_bvalid && axi_wready && S_AXI_WVALID) begin
            axi_bvalid <= 1'b1;
            axi_bresp <= 2'b00;
        end else if (S_AXI_BREADY && axi_bvalid) begin
            axi_bvalid <= 1'b0;
        end
    end

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_arready <= 1'b0;
            axi_araddr <= {C_S_AXI_ADDR_WIDTH{1'b0}};
        end else if (!axi_arready && S_AXI_ARVALID) begin
            axi_arready <= 1'b1;
            axi_araddr <= S_AXI_ARADDR;
        end else begin
            axi_arready <= 1'b0;
        end
    end

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_rvalid <= 1'b0;
            axi_rresp <= 2'b00;
        end else if (axi_arready && S_AXI_ARVALID && !axi_rvalid) begin
            axi_rvalid <= 1'b1;
            axi_rresp <= 2'b00;
        end else if (axi_rvalid && S_AXI_RREADY) begin
            axi_rvalid <= 1'b0;
        end
    end

    assign slv_reg_rden = axi_arready && S_AXI_ARVALID && !axi_rvalid;

    always @(*) begin
        case (axi_araddr[ADDR_LSB + OPT_MEM_ADDR_BITS : ADDR_LSB])
            3'h0: reg_data_out = slv_reg0;
            3'h1: reg_data_out = {IMG_HEIGHT[15:0], IMG_WIDTH[15:0]};
            3'h2: reg_data_out = {16'd0, OUTPUT_BYTES[15:0]};
            3'h3: reg_data_out = {16'd0, INPUT_BYTES[15:0]};
            3'h4: reg_data_out = {16'd0, RECORD_BYTES[15:0]};
            3'h5: reg_data_out = {8'hD1, detector_ap_idle, done_seen, detector_ap_done, M_AXIS_TVALID, S_AXIS_TREADY, 19'd0};
            3'h6: reg_data_out = in_cnt;
            3'h7: reg_data_out = out_cnt;
            default: reg_data_out = 32'd0;
        endcase
    end

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_rdata <= 32'd0;
        end else if (slv_reg_rden) begin
            axi_rdata <= reg_data_out;
        end
    end

endmodule
