module labseven5(SW,HEX5,HEX4,HEX3,HEX2,HEX1,HEX0);
	input [8:0]SW;
	output [6:0]HEX0;
	output [6:0]HEX1;
	output [6:0]HEX2;
	output [6:0]HEX3;
	output [6:0]HEX4;
	output [6:0]HEX5;
	reg [7:0]A,B,C;
	wire [15:0]ans;
	wire [7:0] w1,w2,w3,w4,w5,w6,w7,w8,w9;
	wire c1,c2,c3;
	always@ (SW[8]) begin
		if(SW[8]==1) begin
			A<=SW[7:0];
			C<=SW[7:0];
		end
		else begin
			B<=SW[7:0];
			C<=SW[7:0];
		end
	end
	
	m4 mm1(A[3:0],B[3:0],w1[7:0]);
	m4 mm2(A[3:0],B[7:4],w2[7:0]);
	m4 mm3(A[7:4],B[3:0],w3[7:0]);
	m4 mm4(A[7:4],B[7:4],w4[7:0]);
	
	assign ans[3:0]=w1[3:0];
	
	full_8 fff1(w2[7:0],w3[7:0],0,c1,w5[7:0]);
	assign w6[3:0]=w1[7:4];
	assign w6[7:4]={4'b0000};
	full_8 fff2(w6[7:0],w5[7:0],0,c2,w7[7:0]);
	assign ans[7:4]=w7[3:0];
	
	assign w8[3:0]=w7[7:4];
	assign w8[7:5]={3'b0000};
	assign w8[4]=c1;
	full_8 fff3(w8[7:0],w4[7:0],0,c3,w9[7:0]);
	assign ans[15:8]=w9[7:0];

	hexto7segment s1(C[7:4],HEX5);
	hexto7segment s2(C[3:0],HEX4);
	hexto7segment s3(ans[15:12],HEX3);
	hexto7segment s4(ans[11:8],HEX2);
	hexto7segment s5(ans[7:4],HEX1);
	hexto7segment s6(ans[3:0],HEX0);
	
endmodule

module m4(a,b,s);
	input [3:0]a;
	input [3:0]b;
	output [7:0]s;
	wire [10:0] c;
	wire [23:0]w;
	and(w[0],a[0],b[0]);
	and(w[1],a[0],b[1]);
	and(w[2],a[1],b[0]);
	and(w[3],a[0],b[2]);
	and(w[4],a[1],b[1]);
	and(w[5],a[2],b[0]);
	and(w[6],a[0],b[3]);
	and(w[7],a[1],b[2]);
	and(w[8],a[2],b[1]);
	and(w[9],a[3],b[0]);
	and(w[10],a[1],b[3]);
	and(w[11],a[2],b[2]);
	and(w[12],a[3],b[1]);
	and(w[13],a[2],b[3]);
	and(w[14],a[3],b[2]);
	and(w[15],a[3],b[3]);
	
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


module full_8(a,b,cin,cout,sout);
	input [7:0]a,b;
	input cin;
	output cout;
	output [7:0]sout;
	
	wire[7:0]c;
	full_adder ff1(a[0],b[0],cin,c[0],sout[0]);
	full_adder ff2(a[1],b[1],c[0],c[1],sout[1]);
	full_adder ff3(a[2],b[2],c[1],c[2],sout[2]);
	full_adder ff4(a[3],b[3],c[2],c[3],sout[3]);
	full_adder ff5(a[4],b[4],c[3],c[4],sout[4]);
	full_adder ff6(a[5],b[5],c[4],c[5],sout[5]);
	full_adder ff7(a[6],b[6],c[5],c[6],sout[6]);
	full_adder ff8(a[7],b[7],c[6],c[7],sout[7]);
	assign cout=c[7];
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