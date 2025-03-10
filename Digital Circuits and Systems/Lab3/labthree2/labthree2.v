module labthree2(SW,HEX1,HEX0);
input [3:0]SW;
output [6:0]HEX1,HEX0;
wire sign ;

wire [3:0]y, v; 

 

comparator comp(SW[3],SW[2],SW[1],SW[0],sign);
circuitA a(SW[3:0],sign,v[3:0]);

circuitB b(sign,HEX1);
char_7seg c(v,HEX0 );

endmodule 

module comparator(a,b,c,d,sign);
input a,b,c,d;
output sign ;
wire  s5,s4,s3,s2,s1,s0;
assign sign = s5|s4|s3|s2|s1|s0;
assign s5 = a&~b&c&~d;  //10
assign s4 = a&~b&c&d;    //11
assign s3 = a&b&~c&~d;  //12
assign s2 = a&b&~c&d;  //13
assign s1 = a&b&c&~d;  //14
assign s0 = a&b&c&d;    //15
endmodule

module circuitA(SW,sign,v);
input [3:0]SW;
input sign;
wire [2:0]x;
output [3:0]v;
								  
assign x[0]=(~SW[2]&SW[1]&~SW[0]&0)|(~SW[2]&SW[1]&SW[0]&1)|(SW[2]&~SW[1]&~SW[0]&0)|(SW[2]&~SW[1]&SW[0]&1)|(SW[2]&SW[1]&~SW[0]&0)|(SW[2]&SW[1]&SW[0]&1);
assign x[1]=(~SW[2]&SW[1]&~SW[0]&0)|(~SW[2]&SW[1]&SW[0]&0)|(SW[2]&~SW[1]&~SW[0]&1)|(SW[2]&~SW[1]&SW[0]&1)|(SW[2]&SW[1]&~SW[0]&0)|(SW[2]&SW[1]&SW[0]&0);
assign x[2]=(~SW[2]&SW[1]&~SW[0]&0)|(~SW[2]&SW[1]&SW[0]&0)|(SW[2]&~SW[1]&~SW[0]&0)|(SW[2]&~SW[1]&SW[0]&0)|(SW[2]&SW[1]&~SW[0]&1)|(SW[2]&SW[1]&SW[0]&1);
	
assign v[0]=(~sign&SW[0])|(sign&x[0]);
assign v[1]=(~sign&SW[1])|(sign&x[1]);
assign v[2]=(~sign&SW[2])|(sign&x[2]);
assign v[3]=(~sign&SW[3])|(sign&0);
								  

endmodule

module circuitB(a,hex);
input a;
output [6:0]hex;


assign hex[0]=(a&1);
assign hex[1]=(a&0);
assign hex[2]=(a&0);
assign hex[3]=(a&1);
assign hex[4]=(a&1);
assign hex[5]=(a&1);
assign hex[6]=(a&1)|(~a&1);
endmodule


module char_7seg(SW,HEX);
input [3:0]SW;
output [6:0]HEX;

assign HEX[0]=(~SW[3]&~SW[2]&~SW[1]&~SW[0]&0)|(~SW[3]&~SW[2]&~SW[1]&SW[0])|(~SW[3]&~SW[2]&SW[1]&~SW[0]&0)|(~SW[3]&~SW[2]&SW[1]&SW[0]&0)|(~SW[3]&SW[2]&~SW[1]&~SW[0])|(~SW[3]&SW[2]&~SW[1]&SW[0]&0)|(~SW[3]&SW[2]&SW[1]&~SW[0]&0)|(~SW[3]&SW[2]&SW[1]&SW[0]&0)|(SW[3]&~SW[2]&~SW[1]&~SW[0]&0)|(SW[3]&~SW[2]&~SW[1]&SW[0]&0)|(SW[3]&(SW[2]|SW[1]));
assign HEX[1]=(~SW[3]&~SW[2]&~SW[1]&~SW[0]&0)|(~SW[3]&~SW[2]&~SW[1]&SW[0]&0)|(~SW[3]&~SW[2]&SW[1]&~SW[0]&0)|(~SW[3]&~SW[2]&SW[1]&SW[0]&0)|(~SW[3]&SW[2]&~SW[1]&~SW[0]&0)|(~SW[3]&SW[2]&~SW[1]&SW[0])|(~SW[3]&SW[2]&SW[1]&~SW[0])|(~SW[3]&SW[2]&SW[1]&SW[0]&0)|(SW[3]&~SW[2]&~SW[1]&~SW[0]&0)|(SW[3]&~SW[2]&~SW[1]&SW[0]&0)|(SW[3]&(SW[2]|SW[1]));
assign HEX[2]=(~SW[3]&~SW[2]&~SW[1]&~SW[0]&0)|(~SW[3]&~SW[2]&~SW[1]&SW[0]&0)|(~SW[3]&~SW[2]&SW[1]&~SW[0])|(~SW[3]&~SW[2]&SW[1]&SW[0]&0)|(~SW[3]&SW[2]&~SW[1]&~SW[0]&0)|(~SW[3]&SW[2]&~SW[1]&SW[0]&0)|(~SW[3]&SW[2]&SW[1]&~SW[0]&0)|(~SW[3]&SW[2]&SW[1]&SW[0]&0)|(SW[3]&~SW[2]&~SW[1]&~SW[0]&0)|(SW[3]&~SW[2]&~SW[1]&SW[0]&0)|(SW[3]&(SW[2]|SW[1]));
assign HEX[3]=(~SW[3]&~SW[2]&~SW[1]&~SW[0]&0)|(~SW[3]&~SW[2]&~SW[1]&SW[0])|(~SW[3]&~SW[2]&SW[1]&~SW[0]&0)|(~SW[3]&~SW[2]&SW[1]&SW[0]&0)|(~SW[3]&SW[2]&~SW[1]&~SW[0])|(~SW[3]&SW[2]&~SW[1]&SW[0]&0)|(~SW[3]&SW[2]&SW[1]&~SW[0]&0)|(~SW[3]&SW[2]&SW[1]&SW[0])|(SW[3]&~SW[2]&~SW[1]&~SW[0]&0)|(SW[3]&~SW[2]&~SW[1]&SW[0])|(SW[3]&(SW[2]|SW[1]));
assign HEX[4]=(~SW[3]&~SW[2]&~SW[1]&~SW[0]&0)|(~SW[3]&~SW[2]&~SW[1]&SW[0])|(~SW[3]&~SW[2]&SW[1]&~SW[0]&0)|(~SW[3]&~SW[2]&SW[1]&SW[0])|(~SW[3]&SW[2]&~SW[1]&~SW[0])|(~SW[3]&SW[2]&~SW[1]&SW[0])|(~SW[3]&SW[2]&SW[1]&~SW[0]&0)|(~SW[3]&SW[2]&SW[1]&SW[0])|(SW[3]&~SW[2]&~SW[1]&~SW[0]&0)|(SW[3]&~SW[2]&~SW[1]&SW[0])|(SW[3]&(SW[2]|SW[1]));
assign HEX[5]=(~SW[3]&~SW[2]&~SW[1]&~SW[0]&0)|(~SW[3]&~SW[2]&~SW[1]&SW[0])|(~SW[3]&~SW[2]&SW[1]&~SW[0])|(~SW[3]&~SW[2]&SW[1]&SW[0])|(~SW[3]&SW[2]&~SW[1]&~SW[0]&0)|(~SW[3]&SW[2]&~SW[1]&SW[0]&0)|(~SW[3]&SW[2]&SW[1]&~SW[0]&0)|(~SW[3]&SW[2]&SW[1]&SW[0])|(SW[3]&~SW[2]&~SW[1]&~SW[0]&0)|(SW[3]&~SW[2]&~SW[1]&SW[0]&0)|(SW[3]&(SW[2]|SW[1]));
assign HEX[6]=(~SW[3]&~SW[2]&~SW[1]&~SW[0])|(~SW[3]&~SW[2]&~SW[1]&SW[0])|(~SW[3]&~SW[2]&SW[1]&~SW[0]&0)|(~SW[3]&~SW[2]&SW[1]&SW[0]&0)|(~SW[3]&SW[2]&~SW[1]&~SW[0]&0)|(~SW[3]&SW[2]&~SW[1]&SW[0]&0)|(~SW[3]&SW[2]&SW[1]&~SW[0]&0)|(~SW[3]&SW[2]&SW[1]&SW[0])|(SW[3]&~SW[2]&~SW[1]&~SW[0]&0)|(SW[3]&~SW[2]&~SW[1]&SW[0]&0)|(SW[3]&(SW[2]|SW[1]));
endmodule