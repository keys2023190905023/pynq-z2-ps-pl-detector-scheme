`timescale 1 ns / 1 ps

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

    wire slv_reg_wren;
    wire slv_reg_rden;
    reg [31:0] reg_data_out;
    integer byte_index;

    wire soft_reset = slv_reg0[1];
    wire [31:0] padded_input_pixels = {16'd0, slv_reg1[15:0]} * {16'd0, slv_reg1[31:16]};
    wire [31:0] output_pixels =
        ((slv_reg1[15:0] >= 16'd2) && (slv_reg1[31:16] >= 16'd2))
        ? ({16'd0, (slv_reg1[15:0] - 16'd2)} * {16'd0, (slv_reg1[31:16] - 16'd2)})
        : 32'd0;

    reg [31:0] in_cnt;
    reg [31:0] out_cnt;
    wire frame_fed = (in_cnt >= padded_input_pixels);

    wire core_read_en;
    wire [31:0] core_read_addr;
    wire [31:0] core_write_addr;
    wire raw_ofm_write_en;
    wire [7:0] raw_ofm_write_data;

    wire engine_starved = core_read_en && !S_AXIS_TVALID && !frame_fed;
    wire engine_blocked = raw_ofm_write_en && !M_AXIS_TREADY;
    reg engine_ce_reg;
    wire engine_clk;

    wire [31:0] status_reg = {
        8'hA5,
        out_cnt[15:0],
        in_cnt[5:0],
        raw_ofm_write_en,
        core_read_en
    };

    assign S_AXI_AWREADY = axi_awready;
    assign S_AXI_WREADY = axi_wready;
    assign S_AXI_BRESP = axi_bresp;
    assign S_AXI_BVALID = axi_bvalid;
    assign S_AXI_ARREADY = axi_arready;
    assign S_AXI_RDATA = axi_rdata;
    assign S_AXI_RRESP = axi_rresp;
    assign S_AXI_RVALID = axi_rvalid;

    assign M_AXIS_TKEEP = 1'b1;
    assign S_AXIS_TREADY = core_read_en && engine_ce_reg && !frame_fed;
    assign M_AXIS_TDATA = raw_ofm_write_data;
    assign M_AXIS_TVALID = raw_ofm_write_en && (out_cnt < output_pixels);
    assign M_AXIS_TLAST = M_AXIS_TVALID && (out_cnt == output_pixels - 1'b1);

    BUFGCE u_engine_clk_gate (
        .I(S_AXI_ACLK),
        .CE(engine_ce_reg),
        .O(engine_clk)
    );

    YOLOv8_Top #(
        .DATA_WIDTH(8),
        .IMG_WIDTH(642),
        .MAX_OUTPUT_PIXELS(8192)
    ) u_yolo_engine (
        .clk(engine_clk),
        .rst_n(S_AXI_ARESETN & ~soft_reset),
        .ap_start(slv_reg0[0]),
        .ap_done(),
        .ap_idle(),
        .img_w(slv_reg1[15:0]),
        .img_h(slv_reg1[31:16]),
        .total_channels({13'd0, slv_reg0[4], slv_reg0[6], slv_reg0[5]}),
        .layer_bias(slv_reg2),
        .quant_shift(slv_reg3[20:16]),
        .quant_scale(slv_reg3[15:0]),
        .output_zp(slv_reg0[31:24]),
        .ifm_read_addr(core_read_addr),
        .ifm_read_en(core_read_en),
        .ifm_read_data(S_AXIS_TDATA),
        .w_00(slv_reg4[7:0]),
        .w_01(slv_reg4[15:8]),
        .w_02(slv_reg4[23:16]),
        .w_10(slv_reg4[31:24]),
        .w_11(slv_reg5[7:0]),
        .w_12(slv_reg5[15:8]),
        .w_20(slv_reg5[23:16]),
        .w_21(slv_reg5[31:24]),
        .w_22(slv_reg6[7:0]),
        .ofm_write_addr(core_write_addr),
        .ofm_write_en(raw_ofm_write_en),
        .ofm_write_data(raw_ofm_write_data)
    );

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN || soft_reset) begin
            in_cnt <= 32'd0;
            out_cnt <= 32'd0;
        end else begin
            if (S_AXIS_TVALID && S_AXIS_TREADY) begin
                in_cnt <= in_cnt + 32'd1;
            end
            if (M_AXIS_TVALID && M_AXIS_TREADY) begin
                if (out_cnt == output_pixels - 1'b1) begin
                    out_cnt <= 32'd0;
                end else begin
                    out_cnt <= out_cnt + 32'd1;
                end
            end
            if (slv_reg0[0] && (in_cnt >= padded_input_pixels)) begin
                in_cnt <= 32'd0;
            end
        end
    end

    always @(negedge S_AXI_ACLK or negedge S_AXI_ARESETN) begin
        if (!S_AXI_ARESETN || soft_reset) begin
            engine_ce_reg <= 1'b1;
        end else begin
            engine_ce_reg <= !(engine_starved || engine_blocked);
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
            3'h1: reg_data_out = slv_reg1;
            3'h2: reg_data_out = slv_reg2;
            3'h3: reg_data_out = slv_reg3;
            3'h4: reg_data_out = slv_reg4;
            3'h5: reg_data_out = slv_reg5;
            3'h6: reg_data_out = slv_reg6;
            3'h7: reg_data_out = status_reg;
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
