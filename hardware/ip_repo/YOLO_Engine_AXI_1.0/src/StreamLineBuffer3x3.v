`timescale 1ns / 1ps

module StreamLineBuffer3x3 #(
    parameter DATA_WIDTH    = 8,
    parameter MAX_IMG_WIDTH = 640
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         frame_start,
    input  wire                         pixel_valid,
    input  wire [15:0]                  img_width,
    input  wire [DATA_WIDTH-1:0]        pixel_in,
    output wire signed [71:0]           act_window,
    output wire                         window_valid
);
    reg [DATA_WIDTH-1:0] line0_ram [0:MAX_IMG_WIDTH-1];
    reg [DATA_WIDTH-1:0] line1_ram [0:MAX_IMG_WIDTH-1];
    wire [DATA_WIDTH-1:0] line0_tap = line0_ram[x_count];
    wire [DATA_WIDTH-1:0] line1_tap = line1_ram[x_count];

    reg [DATA_WIDTH-1:0] s0_0, s0_1, s0_2;
    reg [DATA_WIDTH-1:0] s1_0, s1_1, s1_2;
    reg [DATA_WIDTH-1:0] s2_0, s2_1, s2_2;
    reg [15:0] x_count;
    reg [15:0] y_count;
    integer idx;

    wire row_start = (x_count == 16'd0);
    wire [DATA_WIDTH-1:0] s0_2_next = row_start ? {DATA_WIDTH{1'b0}} : s0_1;
    wire [DATA_WIDTH-1:0] s0_1_next = row_start ? {DATA_WIDTH{1'b0}} : s0_0;
    wire [DATA_WIDTH-1:0] s0_0_next = line1_tap;
    wire [DATA_WIDTH-1:0] s1_2_next = row_start ? {DATA_WIDTH{1'b0}} : s1_1;
    wire [DATA_WIDTH-1:0] s1_1_next = row_start ? {DATA_WIDTH{1'b0}} : s1_0;
    wire [DATA_WIDTH-1:0] s1_0_next = line0_tap;
    wire [DATA_WIDTH-1:0] s2_2_next = row_start ? {DATA_WIDTH{1'b0}} : s2_1;
    wire [DATA_WIDTH-1:0] s2_1_next = row_start ? {DATA_WIDTH{1'b0}} : s2_0;
    wire [DATA_WIDTH-1:0] s2_0_next = pixel_in;
    wire valid_window_next = (y_count >= 16'd2) && (x_count >= 16'd1) && (x_count < (img_width - 16'd1));

    assign window_valid = valid_window_next;
    // Conv3x3OutputPE interprets the lowest byte as a00/top-left and the highest
    // byte as a22/bottom-right, so the concatenation order here must be the
    // reverse of the visual 3x3 window.
    assign act_window = {
        s2_0_next, s2_1_next, s2_2_next,
        s1_0_next, s1_1_next, s1_2_next,
        s0_0_next, s0_1_next, s0_2_next
    };

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_count      <= 16'd0;
            y_count      <= 16'd0;
            for (idx = 0; idx < MAX_IMG_WIDTH; idx = idx + 1) begin
                line0_ram[idx] <= {DATA_WIDTH{1'b0}};
                line1_ram[idx] <= {DATA_WIDTH{1'b0}};
            end
            s0_0 <= 0; s0_1 <= 0; s0_2 <= 0;
            s1_0 <= 0; s1_1 <= 0; s1_2 <= 0;
            s2_0 <= 0; s2_1 <= 0; s2_2 <= 0;
        end else if (frame_start) begin
            x_count      <= 16'd0;
            y_count      <= 16'd0;
            for (idx = 0; idx < MAX_IMG_WIDTH; idx = idx + 1) begin
                line0_ram[idx] <= {DATA_WIDTH{1'b0}};
                line1_ram[idx] <= {DATA_WIDTH{1'b0}};
            end
            s0_0 <= 0; s0_1 <= 0; s0_2 <= 0;
            s1_0 <= 0; s1_1 <= 0; s1_2 <= 0;
            s2_0 <= 0; s2_1 <= 0; s2_2 <= 0;
        end else if (pixel_valid) begin
            line1_ram[x_count] <= line0_tap;
            line0_ram[x_count] <= pixel_in;
            s0_2 <= s0_2_next;
            s0_1 <= s0_1_next;
            s0_0 <= s0_0_next;
            s1_2 <= s1_2_next;
            s1_1 <= s1_1_next;
            s1_0 <= s1_0_next;
            s2_2 <= s2_2_next;
            s2_1 <= s2_1_next;
            s2_0 <= s2_0_next;

            if (x_count == img_width - 1'b1) begin
                x_count <= 16'd0;
                y_count <= y_count + 1'b1;
            end else begin
                x_count <= x_count + 1'b1;
            end
        end
    end

endmodule
