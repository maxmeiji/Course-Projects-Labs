module labfive1(KEY,SW,HEX3,HEX2,HEX1,HEX0);
   input [0:0]KEY;
	input [1:0]SW;
   wire [15:0]Q;
	wire [27:0]w;
	output [6:0]HEX3,HEX2,HEX1,HEX0;
		
   t_ff t1(SW[1],KEY[0],SW[0],Q[0]);
	t_ff t2((SW[1] & Q[0]),KEY[0],SW[0],Q[1]);
   t_ff t3((SW[1] & Q[1] & Q[0]),KEY[0],SW[0],Q[2]); 
	t_ff t4((SW[1] & Q[1] & Q[0] & Q[2]),KEY[0],SW[0],Q[3]); 
	t_ff t5((SW[1] & Q[1] & Q[0] & Q[2] & Q[3]),KEY[0],SW[0],Q[4]); 
	t_ff t6((SW[1] & Q[1] & Q[0] & Q[2] & Q[3] & Q[4]),KEY[0],SW[0],Q[5]); 
	t_ff t7((SW[1] & Q[1] & Q[0] & Q[2] & Q[3] & Q[4] & Q[5]),KEY[0],SW[0],Q[6]); 
	t_ff t8((SW[1] & Q[1] & Q[0] & Q[2] & Q[3] & Q[4] & Q[5] & Q[6]),KEY[0],SW[0],Q[7]); 
	t_ff t9((SW[1] & Q[1] & Q[0] & Q[2] & Q[3] & Q[4] & Q[5] & Q[6] & Q[7]),KEY[0],SW[0],Q[8]); 
	t_ff t10((SW[1] & Q[1] & Q[0] & Q[2] & Q[3] & Q[4] & Q[5] & Q[6] & Q[7] & Q[8]),KEY[0],SW[0],Q[9]); 
	t_ff t11((SW[1] & Q[1] & Q[0] & Q[2] & Q[3] & Q[4] & Q[5] & Q[6] & Q[7] & Q[8] & Q[9]),KEY[0],SW[0],Q[10]); 
	t_ff t12((SW[1] & Q[1] & Q[0] & Q[2] & Q[3] & Q[4] & Q[5] & Q[6] & Q[7] & Q[8] & Q[9] & Q[10]),KEY[0],SW[0],Q[11]); 
	t_ff t13((SW[1] & Q[1] & Q[0] & Q[2] & Q[3] & Q[4] & Q[5] & Q[6] & Q[7] & Q[8] & Q[9] & Q[10] & Q[11]),KEY[0],SW[0],Q[12]);  
	t_ff t14((SW[1] & Q[1] & Q[0] & Q[2] & Q[3] & Q[4] & Q[5] & Q[6] & Q[7] & Q[8] & Q[9] & Q[10] & Q[11] & Q[12]),KEY[0],SW[0],Q[13]); 
	t_ff t15((SW[1] & Q[1] & Q[0] & Q[2] & Q[3] & Q[4] & Q[5] & Q[6] & Q[7] & Q[8] & Q[9] & Q[10] & Q[11] & Q[12] & Q[13]),KEY[0],SW[0],Q[14]); 
	t_ff t16((SW[1] & Q[1] & Q[0] & Q[2] & Q[3] & Q[4] & Q[5] & Q[6] & Q[7] & Q[8] & Q[9] &Q[10] & Q[11] & Q[12] & Q[13] & Q[14]),KEY[0],SW[0],Q[15]);  
	

	hexto7segment a(Q[3:0],HEX0);
	hexto7segment b(Q[7:4],HEX1);
	hexto7segment c(Q[11:8],HEX2);
	hexto7segment d(Q[15:12],HEX3);
	

endmodule
       
      
module t_ff(t, clk, reset, q);

	input t;
	input clk;
	input reset;
	wire w;
	output reg q;
				

				
always@(posedge clk)
begin

	 if (reset==0)
			q=0;
    else begin
 
	if(t==0)
		q=q;
	else
		q=~q;
		
end
end	 
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


