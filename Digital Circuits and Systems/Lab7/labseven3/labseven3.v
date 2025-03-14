module labseven3(KEY,SW,LEDR,HEX5,HEX4,HEX3,HEX2,HEX1,HEX0);
	input [1:0]KEY;
	input [8:0]SW;
	output [8:0]LEDR;
	output [6:0]HEX0;
	output [6:0]HEX1;
	output [6:0]HEX2;
	output [6:0]HEX3;
	output [6:0]HEX4;
	output [6:0]HEX5;
	reg [7:0] a;
	reg [7:0] b;
	wire [0:0]c;
	wire [7:0]s;
	
	always @(negedge KEY[1],negedge KEY[0]) begin
		if (KEY[0]==0) begin
			a<=0;
			b<=0;
		end
		
		else  begin
			if(SW[8]==1) begin
				b<=SW[7:0];
			end
			else begin
				a<=SW[7:0];
			end
		end
	end
	
	hexto7segment s1(a[7:4],HEX5);
	hexto7segment s2(a[3:0],HEX4);
	hexto7segment s3(b[7:4],HEX3);
	hexto7segment s4(b[3:0],HEX2);
	test(a[7:0],b[7:0],c,s[7:0]);
	
	assign LEDR[7:0]=s[7:0];
	assign LEDR[8]=c;
	hexto7segment s5(s[7:4],HEX1);
	hexto7segment s6(s[3:0],HEX0);

	
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