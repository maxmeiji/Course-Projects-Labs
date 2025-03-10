module labnine3(SW, KEY, LEDR, HEX3, HEX2, HEX1, HEX0);
	input [9:0]SW;
	input [0:0]KEY;
	output [0:0]LEDR;
	output [6:0]HEX3;
	output [6:0]HEX2;
	output [6:0]HEX1;
	output [6:0]HEX0;
	wire [3:0]out;
	wire [3:0]in;
	wire [3:0]fake;
	reg [4:0]addr;
	
	reg[3:0] mem[31:0];
	
	always@(negedge KEY[0]) begin
		addr[4:0]=SW[8:4];
		if(SW[9]==1) begin
			mem[addr]=SW[3:0];
		end
	end
	
	assign out=mem[addr];
	assign LEDR[0]=SW[9];
	assign fake[0]=SW[8];
	assign fake[3:1]=3'b000;
	display7seg seg1(SW[3:0],HEX1);
	display7seg seg2(out[3:0],HEX0);
	display7seg seg3(SW[7:4],HEX2);
	display7seg seg4(fake[3:0],HEX3);
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