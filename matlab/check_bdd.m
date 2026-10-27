function check_bdd(inputDir, outputDir)
% Optional AC-WLS audit. Run install_matpower separately before this function.
assert(exist('dSbus_dV','file') == 2, 'MATPOWER dSbus_dV is not on the MATLAB path');
if ~isfolder(outputDir), mkdir(outputDir); end
network=load(fullfile(inputDir,'network.mat'));Ybus=network.G+1i*network.B;
files=dir(fullfile(inputDir,'*.mat'));sigma=.001;tau=3.841458820694126;
for k=1:numel(files)
    if strcmp(files(k).name,'network.mat'),continue;end
    d=load(fullfile(inputDir,files(k).name));N=size(d.z_pu,1);
    bdd_J=zeros(N,1);bdd_state=zeros(N,27);bdd_converged=false(N,1);
    for i=1:N
        [x,J,ok]=ac_wls(d.z_pu(i,:).',Ybus,sigma);
        bdd_J(i)=J;bdd_state(i,:)=x.';bdd_converged(i)=ok;
    end
    bdd_pass=bdd_converged & bdd_J<tau;
    save(fullfile(outputDir,files(k).name),'bdd_J','bdd_state','bdd_converged','bdd_pass','sigma','tau','-v7');
end
end
