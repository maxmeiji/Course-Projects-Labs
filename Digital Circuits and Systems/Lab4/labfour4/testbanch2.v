`timescale 1ns/100ps
module testbanch2();

reg clock;
reg d;
wire qa,qb,qc;

labfour4 u1(
.Clk(clock),
.D(d),
.Qa(qa),
.Qb(qb),
.Qc(qc)
);

always #2 clock=~clock;

initial begin
	#0 clock=1'b0;
	#0 d=1'b1;
	#1 d=1'b0;
	#2 d=1'b1;
	#2 d=1'b0;
	#2 d=1'b1;
	
end
endmodule