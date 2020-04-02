%%% Knn Vote for the orginal sets, i.e., 411722
%%% Eval_can_out points in total£º 301*3*4096=3698688
can_path=fullfile('Eval_can_out/EVAL_can.txt');   
pc3_path=fullfile('EVAL_PC3.txt'); 
mkdir('myout1')
out_filepath=fullfile('myout1/EVAL_CLS.txt'); 
out=fopen(out_filepath,'w');
ca=dlmread(can_path,',');
pc=dlmread(pc3_path,' ');
pset.x=pc(:,1);
pset.y=pc(:,2); 
pset.z=pc(:,3);
[r,c]=size(pc);
% pc_xyz size: n*3
pc_xyz=zeros(r,3);
pc_xyz(:,1)=pset.x;
pc_xyz(:,2)=pset.y;
pc_xyz(:,3)=pset.z;
cset.x=ca(:,1);
cset.y=ca(:,2); 
cset.z=ca(:,3);
cset.c=ca(:,6); 
[rc,cc]=size(ca);
% pc_xyz size: n*3
c_xyz=zeros(rc,3);
c_xyz(:,1)=cset.x;
c_xyz(:,2)=cset.y;
c_xyz(:,3)=cset.z;
% finds the nearest neighbor in X for each point in Y
tic;[idx, dist] = knnsearch(c_xyz,pc_xyz,'dist','euclidean','k',81);toc
label_mat=[0,1,2,3,4,5,6,7,8];   % 15,21,81
tic;
for i = 1:r
    count=zeros(1,9);
    if ismember(0,cset.c(idx(i,:)))
        for j = 1:15
            num=idx(i,j);
            if dist(i,j)==0
                d=1;
            else
                d=1/dist(i,j);
            end
            if cset.c(num)==0
                count(1)=count(1)+1*d;
            elseif cset.c(num)==1
                count(2)=count(2)+1*d;
            elseif cset.c(num)==2
                count(3)=count(3)+1*d;
            elseif cset.c(num)==3
                count(4)=count(4)+1*d;
            elseif cset.c(num)==4
                count(5)=count(5)+1*d;
            elseif cset.c(num)==5
                count(6)=count(6)+1*d;
            elseif cset.c(num)==6
                count(7)=count(7)+1*d;
            elseif cset.c(num)==7
                count(8)=count(8)+1*d;
            else
                count(9)=count(9)+1*d;
            end
        end
    elseif ismember(3,cset.c(idx(i,:)))|| ismember(4,cset.c(idx(i,:))) || ismember(6,cset.c(idx(i,:)))||ismember(7,cset.c(idx(i,:)))
        for j = 1:21
            num=idx(i,j);
            if cset.c(num)==0
                count(1)=count(1)+1;
            elseif cset.c(num)==1
                count(2)=count(2)+1;
            elseif cset.c(num)==2
                count(3)=count(3)+1;
            elseif cset.c(num)==3
                count(4)=count(4)+1;
            elseif cset.c(num)==4
                count(5)=count(5)+1;
            elseif cset.c(num)==5
                count(6)=count(6)+1;
            elseif cset.c(num)==6
                count(7)=count(7)+1;
            elseif cset.c(num)==7
                count(8)=count(8)+1;
            else
                count(9)=count(9)+1;
            end
        end
    else
        for j = 1:81
            num=idx(i,j);
            if cset.c(num)==0
                count(1)=count(1)+1;
            elseif cset.c(num)==1
                count(2)=count(2)+1;
            elseif cset.c(num)==2
                count(3)=count(3)+1;
            elseif cset.c(num)==3
                count(4)=count(4)+1;
            elseif cset.c(num)==4
                count(5)=count(5)+1;
            elseif cset.c(num)==5
                count(6)=count(6)+1;
            elseif cset.c(num)==6
                count(7)=count(7)+1;
            elseif cset.c(num)==7
                count(8)=count(8)+1;
            else
                count(9)=count(9)+1;
            end
        end
    end
    [val,index]=max(count); 
    new_label=label_mat(index);
    fprintf(out,'%d\n',new_label);
end
toc;
fclose(out);
