module labeight4(SW,KEY,HEX0);
	input [2:0]SW;
	input [1:0]KEY;
	output [6:0]HEX0;
	reg [3:0] num;
	reg w0,w1;
	parameter c1=2'b00,c2=2'b10,c3=2'b01,c4=2'b11;
	integer flag=0;
	
	always @(negedge KEY[0]) begin
		w0=SW[1];
		w1=SW[2];
		if(~SW[0]) begin
			num=0;
			w0=0;
			w1=0;
		end
		else begin
			case({w0,w1})
				c1: begin
					num=num;
				end
				c2: begin
					num=num+1;
					if(num>=10) begin
						num=0;
					end
				end
				c3: begin
					num=num+2;
					if(num==10) begin
						num=0;
					end
					else if(num==11) begin
						num=1;
					end
				end
				
				c4: begin
					if(flag==1) begin
						num=10;
						flag=0;
					end
					if(num>=1)
						if(flag==0)
							num=num-1;
								if(num==0)
									flag=1;
					end
					
				default: begin 
					num=0;
				end	
			endcase
		end
	end
	
	display7seg seg1(num,HEX0);
	
endmodule

module display7seg(num,HEX);
	input [3:0]num;
	output reg[6:0]HEX;
	always@(num) begin
		case(num)
			0:HEX=7'b1000000;
			1:HEX=7'b1111001;
			2:HEX=7'b0100100;
			3:HEX=7'b0110000;
			4:HEX=7'b0011001;
			5:HEX=7'b0010010;
			6:HEX=7'b0000010;
			7:HEX=7'b1111000;
			8:HEX=7'b0000000;
			9:HEX=7'b0010000;
			10:HEX=7'b0001000;
			11:HEX=7'b0000011;
			12:HEX=7'b1000110;
			13:HEX=7'b0100001;
			14:HEX=7'b0000110;
			15:HEX=7'b0001110;
		endcase
	end
endmodule