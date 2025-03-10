module labseven6(SW,KEY,HEX3,HEX2,HEX1,HEX0,LEDR);
	input [9:0] SW;
	input [3:0] KEY;
	output [6:0] HEX3, HEX2, HEX1, HEX0;
	output [9:9] LEDR;

	wire cout;
	reg [7:0] a,b,c,d;
	wire [15:0] ab,cd,out;
	
	reg [15:0] ans;
	
	always@(posedge KEY[1], negedge KEY[0]) begin
	if(KEY[0]==0) begin
		a<=0;
		b<=0;
		c<=0;
		d<=0;
	end
	else begin
	if(SW[9] == 1) begin
		if(SW[8] == 0) begin
			if( KEY[2] == 0)
				a[7:0] <= SW[7:0];
			else
				b[7:0] <= SW[7:0];
		end
		else begin
			if(KEY[2] == 1)
				c[7:0] <= SW[7:0];
			else
				d[7:0] <= SW[7:0];
		end
		end
	end
	end
	
	
	always@(KEY[3]) begin
	if(SW[8]==0 && KEY[3]==1) begin
		ans[15:8]=a[7:0];
		ans[7:0]=b[7:0];
	end
	else if(SW[8]==1 && KEY[3]==1) begin
		ans[15:8]=c[7:0];
		ans[7:0]=d[7:0];
	end
	else if(KEY[3] == 0) begin
		ans[15:0] = out;
	end
	end
	mul(a,b,ab);
	mul(c,d,cd);
	add (ab,cd,cout,out);

	hexto7segment h1(ans[15:12],HEX3);
	hexto7segment h2(ans[11:8],HEX2);
	hexto7segment h3(ans[7:4],HEX1);
	hexto7segment h4(ans[3:0],HEX0);
	assign LEDR[9] = cout;
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