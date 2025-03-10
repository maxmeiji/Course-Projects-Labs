module labeight3(SW, KEY, LEDR);
	input [1:0]SW;
	input [1:0]KEY;
	output [9:0]LEDR;
	reg [3:0]shift1;
	reg [3:0]shift2;
	reg pre,now;
	reg z;
	
	always@(posedge KEY[0]) begin
		if(~SW[0]) begin
			shift1 = 4'b1111;
			shift2 = 0;
			pre = 0;
			now = 0;
			z = 0;
		end
		else begin
			pre = now;
			now = SW[1];
			if(pre==now) begin
				if(SW[1]) begin
					shift2=shift2<<1;
					shift2[0] = 1;
				end
				else begin
					shift1 = shift1<<1;
					shift1[0] = 0;
				end
			end
			
			else begin
				if(now) begin	
					shift1 = 4'b1111;
					shift2 = 4'b0001;
				end
				else begin
					shift2 = 0;	
	            shift1 = 4'b1110;
				end
			end
			
			if((shift2==4'b1111)|(shift1==4'b0000)) begin
				z = 1;
			end
			else begin
				z = 0;
			end
		end
	end	
	
	assign LEDR[9] = z;
	assign LEDR[7:4] = shift2[3:0];
	assign LEDR[3:0] = shift1[3:0];
	
endmodule