`timescale 1ns/100ps
module testbanch1();

reg clock;
reg r;
reg s;
wire q;

labfour1 u1(
.Clk(clock),
.R(r),
.S(s),
.Q(q)
);

always #2 clock=~clock;

initial begin
	#0 clock=1'b0;
	#0 r=1'b1;
	#0 s=1'b1;
	#3 r=1'b0;
	#3 s=1'b0;	
end
endmodule