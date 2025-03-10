module labeight2(SW,KEY,LEDR);
	input [1:0]SW;
	input [0:0]KEY;
	output [9:0]LEDR;
	reg [3:0] Y;
	reg r;
	parameter A=4'b0000,B=4'b0001,C=4'b0010,D=4'b0011,E=4'b0100,F=4'b0101,G=4'b0110,H=4'b0111,I=4'b1000;
	
	always @(posedge KEY[0]) begin
		if(KEY[0]==1) begin
		case(Y)
			A: if(SW[1]==0) begin 
					Y=B; 
					r=0; 
				end
				else begin 
					Y=F; 
					r=0; 
				end
			B: if(SW[1]==0) begin 
					Y=C; 
					r=0; 
				end
				else begin 
					Y=F; 
					r=0; 
				end
			C: if(SW[1]==0) begin 
					Y=D; 
					r=0; 
				end
				else begin 
					Y=F; 
					r=0; 
				end
			D: if(SW[1]==0) begin 
					Y=E; 
					r=1; 
				end
				else begin 
					Y=F; 
					r=0; 
				end
			E: if(SW[1]==0) begin 
					Y=E; 
					r=1; 
				end
				else begin 
					Y=F; 
					r=0; 
				end
			F: if(SW[1]==0) begin 
					Y=B; 
					r=0; 
				end
				else begin 
					Y=G; 
					r=0; 
				end
			G: if(SW[1]==0) begin 
					Y=B; 
					r=0; 
				end
				else begin 
					Y=H; 
					r=0; 
				end
			H: if(SW[1]==0) begin 
					Y=B; 
					r=0; 
				end
				else begin 
					Y=I; 
					r=1; 
				end
			I: if(SW[1]==0) begin 
					Y=B; 
					r=0; 
				end
				else begin 
					Y=I; 
					r=1; 
				end
			default: begin 
				Y=A; 
				r=0; 
			end
		endcase
		end
		
	if(SW[0]==0) begin
		Y=A;
		r=0;
	end
	
	end
	
	assign LEDR[3:0]=Y;
	assign LEDR[9]=r;
	
endmodule