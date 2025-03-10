module labnine5(CLOCK_50,SW,KEY,LEDR,HEX3,HEX2,HEX0);
	input CLOCK_50;
	input [0:0]KEY;
	input[9:0] SW;
	output [0:0]LEDR;
	output [6:0]HEX0;
	output [6:0]HEX2;
	output [6:0]HEX3;
	
	wire [3:0]out;
	reg [4:0]addr,address;
	reg [30:0]count;
	reg temp;
	wire [3:0]fake;
	reg [4:0]rec;
	reg [3:0]change,ans;
	
	
	test3(address[4:0],temp,SW[8:5],change,out);
	
	always@(posedge CLOCK_50) begin
	
	if(~KEY[0]) begin
		count=0;
		temp=0;
		addr=0;
		address=0;
		temp=1;
	end
	

	if(count==50000000) begin

		count=0;
		ans=out;
		rec[4:0]=addr[4:0];
		if(addr==5'b11111)begin
			addr=0;
		end
		else begin
			addr=addr+1;
		end
	end
	
	//前 1 step 先更新
	else if(count==49999999) begin
		change=0;
		temp=0;
		address=addr;
		temp=1;
		count=count+1;
	end
	
	//input
	else if(count==25000000 & SW[9]==1)begin 
		change=1;
		temp=1;
		count=count+1;
		address=SW[4:0];
	end
	else begin
		change=0;
		count=count+1;	
		temp=0;
	end
	end
		
	assign LEDR[0]=SW[9];
	assign fake[0]=rec[4];
	assign fake[3:1]=3'b000;
	
	display7seg a(ans[3:0],HEX0);
	display7seg b(rec[3:0],HEX2);
	display7seg c(fake[3:0],HEX3);

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