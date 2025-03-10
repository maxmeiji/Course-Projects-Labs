module labsix1(KEY,CLOCK_50,HEX0,HEX1,HEX2);
	input [0:0]KEY;
	input CLOCK_50;
	output [6:0]HEX0;
	output [6:0]HEX1;
	output [6:0]HEX2;
	reg [30:0]Q;
	reg [3:0]R1,R2,R3;

	
	always @(posedge CLOCK_50)begin
		if(Q == 50000000)begin
			Q <= 0;
			if(R1 == 9)begin
				R1 <= 0;
				if(R2 == 9)begin
				R2<=0;
				R3<=R3+1;
				end
				
				else
					R2<=R2+1;
					
			end
			else
				R1 <= R1+1;
		end
		else begin
			Q <= Q+1;
			if(KEY[0]==0) begin
				R1<=0;
				R2<=0;
				R3<=0;
				Q<=0;
	end
	end		
	end


	hexto7segment s(R1,HEX0);
	hexto7segment ss(R2,HEX1);
	hexto7segment sss(R3,HEX2);

	endmodule

module hexto7segment (
   input [3:0] iDIG,
   output reg [6:0] oSEG
	 );
 
 always@(iDIG) begin
   case(iDIG)
     4'b0001: oSEG = 7'b1111001;
	  4'b0010: oSEG = 7'b0100100;  // ---t----     4'h2: oSEG = 7'b0100100;  // |      |
     4'b0011: oSEG = 7'b0110000;  // lt    rt
     4'b0100: oSEG = 7'b0011001;  // |      |
     4'b0101: oSEG = 7'b0010010;  // ---m----
     4'b0110: oSEG = 7'b0000010;  // |      |
     4'b0111: oSEG = 7'b1111000;  // lb    rb
     4'b1000: oSEG = 7'b0000000;  // |      |
     4'b1001: oSEG = 7'b0011000;  // ---b----
     4'b0000: oSEG = 7'b1000000;
   endcase
 end
 
endmodule