`timescale 1ns / 1ps

module tb_TinyFullHwPlOnlyDemoDetectorTop;
    localparam integer IMG_WIDTH = 6;
    localparam integer IMG_HEIGHT = 6;
    localparam integer INPUT_VALUES = IMG_WIDTH * IMG_HEIGHT * 3;
    localparam integer OUTPUT_BYTES = 48;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg ap_start = 1'b0;
    reg signed [7:0] s_axis_tdata = 8'sd0;
    reg s_axis_tvalid = 1'b0;
    wire s_axis_tready;
    wire signed [7:0] m_axis_tdata;
    wire m_axis_tvalid;
    reg m_axis_tready = 1'b1;
    wire m_axis_tlast;
    wire ap_done;
    wire ap_idle;

    integer idx;
    integer out_count;
    reg [7:0] expected [0:OUTPUT_BYTES-1];

    always #5 clk = ~clk;

    TinyFullHwPlOnlyDemoDetectorTop #(
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .RAM_ADDR_WIDTH(8),
        .STEM_INPUT_RAM_DEPTH(256),
        .STEM_OUTPUT_RAM_DEPTH(256),
        .HEAD_INPUT_RAM_DEPTH(256),
        .HEAD_ACCUM_RAM_DEPTH(256),
        .HEAD_OUTPUT_RAM_DEPTH(256)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .ap_start(ap_start),
        .ap_done(ap_done),
        .ap_idle(ap_idle),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast)
    );

    initial begin
        expected[0] = 8'd68;   expected[1] = 8'd69;   expected[2] = 8'd84;   expected[3] = 8'd49;
        expected[4] = 8'd2;    expected[5] = 8'd0;    expected[6] = 8'd2;    expected[7] = 8'd0;
        expected[8] = 8'd1;    expected[9] = 8'd0;    expected[10] = 8'd5;   expected[11] = 8'd0;
        expected[12] = 8'd3;   expected[13] = 8'd0;   expected[14] = 8'd8;   expected[15] = 8'd0;
        expected[16] = 8'd24;  expected[17] = 8'd127; expected[18] = 8'd127; expected[19] = 8'd127;
        expected[20] = 8'd127; expected[21] = 8'd166; expected[22] = 8'd127; expected[23] = 8'd30;
        expected[24] = 8'd25;  expected[25] = 8'd127; expected[26] = 8'd127; expected[27] = 8'd127;
        expected[28] = 8'd127; expected[29] = 8'd165; expected[30] = 8'd127; expected[31] = 8'd32;
        expected[32] = 8'd30;  expected[33] = 8'd127; expected[34] = 8'd127; expected[35] = 8'd127;
        expected[36] = 8'd127; expected[37] = 8'd160; expected[38] = 8'd127; expected[39] = 8'd42;
        expected[40] = 8'd31;  expected[41] = 8'd127; expected[42] = 8'd127; expected[43] = 8'd127;
        expected[44] = 8'd127; expected[45] = 8'd159; expected[46] = 8'd127; expected[47] = 8'd44;

        repeat (5) @(posedge clk);
        rst_n <= 1'b1;
        @(posedge clk);
        ap_start <= 1'b1;
        @(posedge clk);
        ap_start <= 1'b0;

        idx = 0;
        while (idx < INPUT_VALUES) begin
            @(posedge clk);
            if (s_axis_tready) begin
                s_axis_tvalid <= 1'b1;
                s_axis_tdata <= idx[7:0];
                idx = idx + 1;
            end
        end
        @(posedge clk);
        s_axis_tvalid <= 1'b0;
        s_axis_tdata <= 8'sd0;
    end

    initial begin
        out_count = 0;
        wait(rst_n);
        while (!ap_done) begin
            @(posedge clk);
            if (m_axis_tvalid && m_axis_tready) begin
                if (m_axis_tdata !== expected[out_count]) begin
                    $fatal(1, "TB_FAIL pl_only_demo expected=%0d got=%0d index=%0d", expected[out_count], m_axis_tdata, out_count);
                end
                out_count = out_count + 1;
            end
        end

        if (out_count != OUTPUT_BYTES) begin
            $fatal(1, "TB_FAIL pl_only_demo wrong output count=%0d expected=%0d", out_count, OUTPUT_BYTES);
        end
        $display("TB_PASS tb_TinyFullHwPlOnlyDemoDetectorTop bytes=%0d", out_count);
        $finish(0);
    end

    initial begin
        #300000;
        $fatal(1, "TB_FAIL pl_only_demo timeout");
    end
endmodule
