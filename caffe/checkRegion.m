function p = checkRegion(proposals, minw, minh)
	w = proposals(:, 3) - proposals(:, 1);
	h = proposals(:, 4) - proposals(:, 2);
	I = find(w >= minw & h >= minh);
	p = proposals(I, :);
end