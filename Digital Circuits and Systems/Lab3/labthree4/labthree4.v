module labthree4(SW,HEX5,HEX4,HEX3,HEX2,HEX1,HEX0,LEDR);
input [8:0]SW;
output [6:0]HEX5,HEX4,HEX3,HEX2,HEX1,HEX0;
wire signa, signb,signc,signd,signe,signf ;

wire [3:0]v,y,z; 
output [9:0]LEDR;
comparator compa(0,SW[7],SW[6],SW[5],SW[4],signa,signb);
circuitA a(SW[7:4],signa,signb,v[3:0]);
circuitB b(signa,signb,HEX3);
char_7seg c(v,HEX2);



comparator compb(0,SW[3],SW[2],SW[1],SW[0],signc,signd);
circuitA d(SW[3:0],signc,signd,y[3:0]);

circuitB e(signc,signd,HEX1);
char_7seg f(y,HEX0);

assign LEDR[9]=(signa|signc);
wire [3:0]ss;
wire [3:0]cc;


full_adder f1(SW[0], SW[4], SW[8], cc[0], ss[0]);
full_adder f2(SW[1], SW[5], cc[0], cc[1], ss[1]);
full_adder f3(SW[2], SW[6], cc[1], cc[2], ss[2]);
full_adder f4(SW[3], SW[7], cc[2], cc[3], ss[3]);

comparator compc(cc[3],ss[3],ss[2],ss[1],ss[0],signe,signf);
circuitA g(ss[3:0],signe,signf,z[3:0]);

circuitB h(signe,signf,HEX5);
char_7seg i(z,HEX4);
endmodule 

module comparator(top,a,b,c,d,sign,signn);
input a,b,c,d,top;
output sign,signn ;
wire  s5,s4,s3,s2,s1,s0;
assign sign = s5|s4|s3|s2|s1|s0;
assign signn=top;
assign s5 = a&~b&c&~d;  
assign s4 = a&~b&c&d;   
assign s3 = a&b&~c&~d;  
assign s2 = a&b&~c&d;  
assign s1 = a&b&c&~d;  
assign s0 = a&b&c&d;    
endmodule

module circuitA(SS,sign,signn,v);
input [3:0]SS;
input sign,signn;
wire [3:0]x,y;
output [3:0]v;
								  
assign x[0]=(~SS[2]&SS[1]&~SS[0]&0)|(~SS[2]&SS[1]&SS[0]&1)|(SS[2]&~SS[1]&~SS[0]&0)|(SS[2]&~SS[1]&SS[0]&1)|(SS[2]&SS[1]&~SS[0]&0)|(SS[2]&SS[1]&SS[0]&1);
assign x[1]=(~SS[2]&SS[1]&~SS[0]&0)|(~SS[2]&SS[1]&SS[0]&0)|(SS[2]&~SS[1]&~SS[0]&1)|(SS[2]&~SS[1]&SS[0]&1)|(SS[2]&SS[1]&~SS[0]&0)|(SS[2]&SS[1]&SS[0]&0);
assign x[2]=(~SS[2]&SS[1]&~SS[0]&0)|(~SS[2]&SS[1]&SS[0]&0)|(SS[2]&~SS[1]&~SS[0]&0)|(SS[2]&~SS[1]&SS[0]&0)|(SS[2]&SS[1]&~SS[0]&1)|(SS[2]&SS[1]&SS[0]&1);

assign y[0]=(~SS[2]&~SS[1]&~SS[0]&0)|(~SS[2]&~SS[1]&SS[0]&1)|(~SS[2]&SS[1]&~SS[0]&0)|(~SS[2]&SS[1]&SS[0]&1);
assign y[1]=(~SS[2]&~SS[1]&~SS[0]&1)|(~SS[2]&~SS[1]&SS[0]&1)|(~SS[2]&SS[1]&~SS[0]&0)|(~SS[2]&SS[1]&SS[0]&0);
assign y[2]=(~SS[2]&~SS[1]&~SS[0]&1)|(~SS[2]&~SS[1]&SS[0]&1)|(~SS[2]&SS[1]&~SS[0]&0)|(~SS[2]&SS[1]&SS[0]&0);
assign y[3]=(~SS[2]&~SS[1]&~SS[0]&0)|(~SS[2]&~SS[1]&SS[0]&0)|(~SS[2]&SS[1]&~SS[0]&1)|(~SS[2]&SS[1]&SS[0]&1);

assign v[0]=(~sign&~signn&SS[0])|(sign&~signn&x[0])|(~sign&signn&y[0]);
assign v[1]=(~sign&~signn&SS[1])|(sign&~signn&x[1])|(~sign&signn&y[1]);
assign v[2]=(~sign&~signn&SS[2])|(sign&~signn&x[2])|(~sign&signn&y[2]);
assign v[3]=(~sign&~signn&SS[3])|(sign&~signn&0)|(~sign&signn&y[3]);
								  

endmodule

module circuitB(a,b,hex);
input a,b;
output [6:0]hex;
wire c;
assign c=a|b;

assign hex[0]=(c&1);
assign hex[1]=(c&0);
assign hex[2]=(c&0);
assign hex[3]=(c&1);
assign hex[4]=(c&1);
assign hex[5]=(c&1);
assign hex[6]=(c&1)|(~c&1);
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

module full_adder(a,b,cin,cout,sout);
input a;
input b;
input cin;
output cout;
output sout;
wire C1, C2, S1;

process(a,b,C1,S1);
process(S1,cin,C2,sout);
assign cout=C1|C2;

endmodule

module process(x,y,C,S);
input x,y;
output C, S;

assign C=(x&y);
assign S=(~x&y)|(x&~y);

endmodule