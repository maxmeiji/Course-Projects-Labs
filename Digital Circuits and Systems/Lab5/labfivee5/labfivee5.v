module labfivee5(KEY,SW,HEX3,HEX2,HEX1,HEX0);
   input [0:0]KEY;
	input [1:0]SW;
   wire [15:0]Q;
	output [6:0]HEX3,HEX2,HEX1,HEX0;

	ha(SW[1],KEY[0],SW[0],Q);

	hexto7segment e(Q[3:0],HEX0);
	hexto7segment f(Q[7:4],HEX1);
	hexto7segment g(Q[11:8],HEX2);
	hexto7segment h(Q[15:12],HEX3);

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