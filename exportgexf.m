function exportgexf(Z, filename)

T = size(Z, 1);
K = size(Z, 2);

[ind1, ind2] = find(tril(squeeze(sum(Z,1))));

fid = fopen(filename, 'w');
fprintf(fid,'<gexf xmlns="http://www.gexf.net/1.2draft" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd" version="1.2">\n');
fprintf(fid, '<graph mode="dynamic" defaultedgetype="undirected" timeformat="double">\n\n');
fprintf(fid, '# Definition of nodes\n');
fprintf(fid, '# --------------------\n');
fprintf(fid, '<nodes>\n');
fprintf(fid, '<node id="%d" />\n', 1:K);
fprintf(fid, '</nodes>\n\n');
fprintf(fid, '# Definition of edges\n');
fprintf(fid, '# --------------------\n');
fprintf(fid, '<edges>\n');
for i=1:length(ind1)
    fprintf(fid, '<edge source="%d" target="%d">\n', ind1(i), ind2(i));
    fprintf(fid, '<spells>\n');
    fprintf(fid, '<spell start="%d" end="%d"/>\n', [find(Z(:,ind1(i),ind2(i))), find(Z(:,ind1(i),ind2(i)))+1]');%intervals when edge active
    fprintf(fid, '</spells>\n');
    fprintf(fid, '</edge>\n\n')  ;  
end
fprintf(fid, '</edges>\n');
fprintf(fid, '</graph>\n');
fprintf(fid, '</gexf>\n');
fclose(fid)