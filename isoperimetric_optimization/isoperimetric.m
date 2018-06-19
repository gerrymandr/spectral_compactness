clear
nTests = 10;
image = double(rgb2gray(imread('cheese.jpg'))<.9);
m = size(image,1);
n = size(image,2);
imagesc(image)

%%

frac = linspace(0,1,nTests);
results = cell(nTests,1);
objectives = zeros(nTests,1);

for i=1:nTests
    fprintf('ITERATION %d OF %d...\n',i,nTests);
    cvx_begin
        cvx_solver mosek
        variable chi(m,n)

        dx = chi(:,2:n) - chi(:,1:(n-1));
        dx(:,end+1) = -chi(:,end);
        dy = chi(2:m,:) - chi(1:(m-1),:);
        dy(end+1,:) = -chi(end,:);
        objectiveTerms = [dx(:) dy(:)];
        myNorms = norms(objectiveTerms,2,2);

        minimize sum(myNorms)
        subject to
            0 <= chi <= 1
            sum(chi(:)) == sum(image(:)) * frac(i)
            chi <= image
    cvx_end
    
    objectives(i) = cvx_optval;
    results{i} = chi;
end

%%
v = VideoWriter('video');
open(v)

for i=1:nTests
   imagesc(results{i});
   caxis([0,1])
   title(sprintf('%g',frac(i)))
   drawnow;
   pause(0.1);
   writeVideo(v,getframe);
end

close(v);