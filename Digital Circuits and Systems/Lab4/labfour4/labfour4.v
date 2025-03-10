module labfour4(Clk,D,Qa,Qb,Qc);
input Clk,D;
output reg Qa,Qb,Qc;

always@(Clk,D) begin
	if(Clk)begin
		Qa=D;
	end
end

always@(posedge Clk) begin
		Qb=D;

end

always@(negedge Clk) begin
		Qc=D;

end


endmodule