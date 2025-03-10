module labeight1(SW,KEY,LEDR);
	input [1:0] SW;
	input [0:0] KEY;
	output [9:0]LEDR;
	wire [8:0]out;
	wire w;
	
	assign w=SW[1];
	assign LEDR[8:0]=out;
	assign LEDR[9]=(out[4]|out[8]);
	
	init a(~KEY[0],SW[0],out[0]);
	d_f_f d1(~KEY[0],SW[0],(~w)&(out[0]|out[5]|out[6]|out[7]|out[8]),out[1]);
	d_f_f d2(~KEY[0],SW[0],(~w)&(out[1]),out[2]);
	d_f_f d3(~KEY[0],SW[0],(~w)&(out[2]),out[3]);
	d_f_f d4(~KEY[0],SW[0],(~w)&(out[3]|out[4]),out[4]);
	d_f_f d5(~KEY[0],SW[0],(w)&(out[0]|out[1]|out[2]|out[3]|out[4]),out[5]);
	d_f_f d6(~KEY[0],SW[0],(w)&(out[5]),out[6]);
	d_f_f d7(~KEY[0],SW[0],(w)&(out[6]),out[7]);
	d_f_f d8(~KEY[0],SW[0],(w)&(out[7]|out[8]),out[8]);
endmodule


module init(clk,reset,Q);
	input clk;
	input reset;
	output reg Q;
	always@(posedge clk) begin
		if(~reset) begin
			Q<=1;
		end
		else begin
			Q<=0;
		end
	end
	
endmodule


module d_f_f(clk,reset,D,Q);
	input clk;
	input reset;
	input D;
	output reg Q;
	
	always@(posedge clk ) begin
		if(~reset) begin
			Q<=0;
		end
		else begin
			Q<=D;
		end
	end
	
endmodule
