module labseven4(SW,HEX5,HEX4,HEX1,HEX0);
	input [8:0]SW;
	output [6:0]HEX0;
	output [6:0]HEX1;

	output [6:0]HEX4;
	output [6:0]HEX5;

	wire [10:0]c;
	wire [7:0]s;
	wire [23:0]w;
	
	hexto7segment s1(SW[7:4],HEX5);
	hexto7segment s2(SW[3:0],HEX4);
	
	and(w[0],SW[0],SW[4]);
	and(w[1],SW[0],SW[5]);
	and(w[2],SW[1],SW[4]);
	and(w[3],SW[0],SW[6]);
	and(w[4],SW[1],SW[5]);
	and(w[5],SW[2],SW[4]);
	and(w[6],SW[0],SW[7]);
	and(w[7],SW[1],SW[6]);
	and(w[8],SW[2],SW[5]);
	and(w[9],SW[3],SW[4]);
	and(w[10],SW[1],SW[7]);
	and(w[11],SW[2],SW[6]);
	and(w[12],SW[3],SW[5]);
	and(w[13],SW[2],SW[7]);
	and(w[14],SW[3],SW[6]);
	and(w[15],SW[3],SW[7]);
	
	assign s[0]=w[0];
	
	full_adder f2(w[1], w[2], 0, c[1], s[1]);
	
	full_adder f3(w[3], w[4], c[1], c[2], w[16]);
	full_adder f4(w[6], w[7], c[2], c[3], w[17]);
	full_adder f5(0, w[10], c[3], w[18], w[19]);
	full_adder f6(w[16], w[5], 0, c[4], s[2]);
	
	full_adder f7(w[17], w[8], c[4], c[5], w[20]);
	full_adder f8(w[19], w[11], c[5], c[6], w[21]);
	full_adder f9(w[18], w[13], c[6], w[22], w[23]);
	full_adder f10(w[20], w[9], 0, c[7], s[3]);
	
	full_adder f11(w[21], w[12], c[7], c[8], s[4]);
	full_adder f12(w[23], w[14], c[8], c[9], s[5]);
	full_adder f13(w[22], w[15], c[9], s[7], s[6]);
	
	hexto7segment s5(s[7:4],HEX1);
	hexto7segment s6(s[3:0],HEX0);
	
endmodule

module full_adder(a,b,cin,cout,sout);
input a;
input b;
input cin;
output cout;
output sout;
wire C1, C2, S1;

process(a,b,C1,S1);
process(S1,cin,C2,sout);
assign cout=C1|C2;

endmodule

module process(x,y,C,S);
input x,y;
output C, S;

assign C=(x&y);
assign S=(~x&y)|(x&~y);

endmodule


module hexto7segment (
   input [3:0] iDIG,
   output reg [6:0] oSEG
	 );
 
 always@(iDIG) begin
   case(iDIG)
     4'h1: oSEG = 7'b1111001;
	4'h2: oSEG = 7'b0100100;  // ---t----     4'h2: oSEG = 7'b0100100;  // |      |
     4'h3: oSEG = 7'b0110000;  // lt    rt
     4'h4: oSEG = 7'b0011001;  // |      |
     4'h5: oSEG = 7'b0010010;  // ---m----
     4'h6: oSEG = 7'b0000010;  // |      |
     4'h7: oSEG = 7'b1111000;  // lb    rb
     4'h8: oSEG = 7'b0000000;  // |      |
     4'h9: oSEG = 7'b0011000;  // ---b----
     4'ha: oSEG = 7'b0001000;
     4'hb: oSEG = 7'b0000011;
     4'hc: oSEG = 7'b1000110;
     4'hd: oSEG = 7'b0100001;
     4'he: oSEG = 7'b0000110;
     4'hf: oSEG = 7'b0001110;
     4'h0: oSEG = 7'b1000000;
   endcase
 end
 
endmodule