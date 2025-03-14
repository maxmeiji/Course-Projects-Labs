module labsix2(SW,KEY,CLOCK_50,HEX0,HEX1,HEX2,HEX3,HEX4,HEX5);
	input [0:0]KEY;
	input CLOCK_50;
	output [6:0]HEX0;
	output [6:0]HEX1;
	output [6:0]HEX2;
	output [6:0]HEX3;
	output [6:0]HEX4;
	output [6:0]HEX5;
	reg [30:0]Q;
	reg [3:0]R1,R2,R3,R4,R5,R6;
	input [9:0]SW;
	reg temp;
	

	always @(posedge CLOCK_50)begin
		if(Q == 50000000)begin
			Q <= 0;
			if(R6==9) begin
				R6<=0;
				if(R5==5)begin
					R5<=0;
					if(R4==9) begin
						R4<=0;
						if(R3==5)begin
							R3<=0;
							if(R2==9 || (R2==3 & R1==2)) begin
								R2<=0;
								if(R2==3 & R1==2) begin
									R1=0;
								end
								else begin
									R1<=R1+1;
								end
							end
							else begin
								R2<=R2+1;
							end
						end
						else begin
							R3<=R3+1;
						end
					end
					else begin 
						R4<=R4+1;
					end
				end
				else begin
					R5<=R5+1;
				end
			end
		
			else begin
				R6<=R6+1;
			end
		end
		else begin
			Q <= Q+1;
			if(temp & SW[8]==1) begin
				if(SW[9]==1) begin
					temp<=0;
					R2<=SW[3:0];
					R1<=SW[7:4];
					R5<=0;
					R6<=0;
					Q<=0;
				end
				else begin
					temp<=0;
					R4<=SW[3:0];
					R3<=SW[7:4];
					R5<=0;
					R6<=0;
					Q<=0;
				end
			end
			
			else begin
			if(~temp & ~SW[8]) begin
				temp<=1;
			end
			end
		end
			
end

	hexto7segment s(R6,HEX0);
	hexto7segment s1(R5,HEX1);
	hexto7segment s2(R4,HEX2);
	hexto7segment s3(R3,HEX3);
	hexto7segment s4(R2,HEX4);
	hexto7segment s5(R1,HEX5);

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