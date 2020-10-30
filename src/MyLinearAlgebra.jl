module MyLinearAlgebra

export poly

using LinearAlgebra


function poly(x)

	m,n = size(x);
	if m == n
	   #Characteristic polynomial (square x)
	   e = eigen(x).values;
	elseif (m==1) || (n==1)
	   e = x;
	end

	#Strip out infinities
	e = e[isfinite.(e)];

	#Expand recursion formula
	n = length(e);
	c = [1 zeros(1,n)];
	for j=1:n
	    c[2:(j+1)] = c[2:(j+1)] - e[j].*c[1:j];
	end

	return c

end






end
