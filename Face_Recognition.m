%%%%%%%%%%%%%%%%%%%%%%% Face Recognition using Eigenface Method %%%%%%%%%%%%%%%%%%%%%%%

%closes all the previous open windows
close all;

%clears command window
clc 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Training Process %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Providing path for Training and Testeing folder

% The training Images are renamed in training folder as 1.jpg, 2.jpg, ...
% 8.jpg for the convenience of the program. 

% The testing Images are renamed in testing folder as 1.jpg, 2.jpg, ...
% 18.jpg for the convenience of the program. (17 test faces and apple being the 18th image)

TrainDatabasePath = 'C:\Users\Hitarthi\Desktop\Project2\Face Dataset\training\';
TestDatabasePath = 'C:\Users\Hitarthi\Desktop\Project2\Face Dataset\testing\';

% Traning images(8) are saved in I1, I2,... I8
I1 = strcat(TrainDatabasePath,'1','.jpg');
I2 = strcat(TrainDatabasePath,'2','.jpg');
I3 = strcat(TrainDatabasePath,'3','.jpg');
I4 = strcat(TrainDatabasePath,'4','.jpg');
I5 = strcat(TrainDatabasePath,'5','.jpg');
I6 = strcat(TrainDatabasePath,'6','.jpg');
I7 = strcat(TrainDatabasePath,'7','.jpg');
I8 = strcat(TrainDatabasePath,'8','.jpg');


% 1. We are reading the image and converting it into 2-D array of type double. This will
% create matrix for the image on with which we can work further.
% 2. Saving images after resizing in A, B...H
% 3. Plotting the images in figure window all together



% Figure(Name,Value): modifies properties of the figure using one or more name-value pair arguments
% Subplot : Create axes in tiled positions
% pcolor(C) : draws a pseudocolor plot. The elements of C are linearly mapped to an index into the current colormap.
% flipud : Flip array up to down
% shading interp varies the color in each line segment and face by interpolating the colormap index or 
% true color value across the line or face.

% 'Xtick' and 'Ytick' : Set or query x-axis and y-axis tick values. Here
% they are set to null. 


figure('Name','Training Data')
A = imresize(double(imread(I1)),[195,231]);
subplot(3,3,1), pcolor(flipud(A)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('1.jpg');

B = imresize(double(imread(I2)),[195,231]);
subplot(3,3,2), pcolor(flipud(B)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('2.jpg');

C = imresize(double(imread(I3)),[195,231]);
subplot(3,3,3), pcolor(flipud(C)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('3.jpg');

D = imresize(double(imread(I4)),[195,231]);
subplot(3,3,4), pcolor(flipud(D)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('4.jpg');

E = imresize(double(imread(I5)),[195,231]);
subplot(3,3,5), pcolor(flipud(E)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('5.jpg');

F = imresize(double(imread(I6)),[195,231]);
subplot(3,3,6), pcolor(flipud(F)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('6.jpg');

G = imresize(double(imread(I7)),[195,231]);
subplot(3,3,7), pcolor(flipud(G)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('7.jpg');

H = imresize(double(imread(I8)),[195,231]);
subplot(3,3,8), pcolor(flipud(H)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('8.jpg');


% Number of Training images = n = 8, Each of dimension MxN (195x231)
% For each traning image, the rows are stacked together to form a column
% vector Ri of dimension MN x 1

% MN * 1 (Column Vector)
% reshape : Reshape array returned as a vector, matrix, multidimensional array, or cell array.
% The data type and number of elements in the reshaped array are the same as the data type and number of elements in original array.

% A,B .. H are reshaped in vector R1,R2, ... R8

R1= reshape(A,195*231,1);
R2= reshape(B,195*231,1);
R3= reshape(C,195*231,1);
R4= reshape(D,195*231,1);
R5= reshape(E,195*231,1);
R6= reshape(F,195*231,1);
R7= reshape(G,195*231,1);
R8= reshape(H,195*231,1);


% Generating mean face : By taking the average of n training face images
Avg_face = (R1+R2+R3+R4+R5+R6+R7+R8)/8;

% reshaping the Avg_face into original dimensions to plot it as figure on screen. 
Show_Avg_face = reshape(Avg_face,195,231);
figure('Name','Mean Image'),subplot(1,1,1), pcolor(flipud(Show_Avg_face)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);



% Subtract mean face from each traning images and save that value 
% reshape to plot the output after subtraction process.
% All training images after subtractiong the avg face is then plotted
% altogether. 

figure('Name','Original Image - Mean Image')
s1 = R1-Avg_face;
sr1 = reshape(s1,195,231);
subplot(3,3,1), pcolor(flipud(sr1)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);  title('1.jpg');

s2 = R2-Avg_face;
sr2 = reshape(s2,195,231);
subplot(3,3,2), pcolor(flipud(sr2)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);  title('2.jpg');

s3 = R3-Avg_face;
sr3 = reshape(s3,195,231);
subplot(3,3,3), pcolor(flipud(sr3)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);  title('3.jpg');

s4 = R4-Avg_face;
sr4 = reshape(s4,195,231);
subplot(3,3,4), pcolor(flipud(sr4)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);  title('4.jpg');

s5 = R5-Avg_face;
sr5 = reshape(s5,195,231);
subplot(3,3,5), pcolor(flipud(sr5)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);  title('5.jpg');

s6 = R6-Avg_face;
sr6 = reshape(s6,195,231);
subplot(3,3,6), pcolor(flipud(sr6)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);  title('6.jpg');

s7 = R7-Avg_face;
sr7 = reshape(s7,195,231);
subplot(3,3,7), pcolor(flipud(sr7)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);  title('7.jpg');

s8 = R8-Avg_face;
sr8 = reshape(s8,195,231);
subplot(3,3,8), pcolor(flipud(sr8)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);  title('8.jpg');


% Creating data matrix by putting all training faces into a single matrix 'Data'
% Data is a 8x45045 vector, where each row represent one training image. 
Data = [reshape(s1,1,195*231)
    reshape(s2,1,195*231)
    reshape(s3,1,195*231)
    reshape(s4,1,195*231)
    reshape(s5,1,195*231)
    reshape(s6,1,195*231)
    reshape(s7,1,195*231)
    reshape(s8,1,195*231)];


% Covariance matrix 
L = (Data)*(Data');

% L is of nxn dimension
size(L)



% eigs : returns subset of eigenvalues and eigenvectors of covariance matrix
% eigs of L will grab largest magnitude of 8 eigenvectors possible.



%%%%%%%% Di %%%%%%%%
% is diagonal matrix, the elements of this eigenvalues are arranged from
% largest to smallest, largest being the most important and smallest being
% the least.

%%%%%%%% V %%%%%%%%
% are the eigenvectors, the first eigenvector of this has dominant avegrage face,
% second eigenvector tells about the feature space.


[V,Di] = eigs(L,8,'largestabs');



% Eigenfaces/ Facespace/ Eigenspace can be found as follows:
% U is dimension of MN x n 

u = V*Data;
U = u';


% Each column of U represents an eigenface. we can output each eigenface as
% an MxN image 
% Eigen faces of Training images are combined and plotted in figure to
% display.


figure('Name','EigenFaces')
subplot(3,3,1), face1=reshape(U(:,1),195,231); pcolor(flipud(face1)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('1.jpg');
subplot(3,3,2), face1=reshape(U(:,2),195,231); pcolor(flipud(face1)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('2.jpg');
subplot(3,3,3), face1=reshape(U(:,3),195,231); pcolor(flipud(face1)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('3.jpg');
subplot(3,3,4), face1=reshape(U(:,4),195,231); pcolor(flipud(face1)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('4.jpg');
subplot(3,3,5), face1=reshape(U(:,5),195,231); pcolor(flipud(face1)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('5.jpg');
subplot(3,3,6), face1=reshape(U(:,6),195,231); pcolor(flipud(face1)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('6.jpg');
subplot(3,3,7), face1=reshape(U(:,7),195,231); pcolor(flipud(face1)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('7.jpg');
subplot(3,3,8), face1=reshape(U(:,8),195,231); pcolor(flipud(face1)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('8.jpg');



% Each traning face then can be projected onto the face space
% generating PCA coefficients of traning faces
% ?i = u*si for i = 1 to 8 as face1 to face8

face1 = u*s1;
face2 = u*s2;
face3 = u*s3;
face4 = u*s4;
face5 = u*s5;
face6 = u*s6;
face7 = u*s7;
face8 = u*s8;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Testing Process %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Extracting the PCA features from test image


% Enter the number of test image that needs to be recognised into Test 1. 
% Test image path is stored in Test1, and it is then being read and
% converted into matrix form.

Test1 = strcat(TestDatabasePath,'7','.jpg');
T1 = imresize( double( imread(Test1)), [195,231]);

% reshaping the test image to show at the end as the input of the program.
input1 = reshape(T1,195*231,1);

%subtract mean face from Input face
figure('Name','Test Image - Mean Image');
Diff = input1-Avg_face;

% reshaping the test image after subtracting the mean to display.
Difference = reshape(Diff,195,231);

% plotting it as figure.
subplot(1,1,1), pcolor(flipud(Difference)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); 




%Compute its projection onto face space
%Test image feature vector
%PCA coefficients of each test image are stored in variable ProjectedTestImage. 
ProjectedTestImage = u*Diff;

%Reconstruct input face image from eigenfaces.
figure('Name','Reconstructed Image');
Recon = U*ProjectedTestImage;

%plotting the reconstructed face. 
Reconstruction = reshape(Recon,195,231);
subplot(1,1,1), pcolor(flipud(Reconstruction)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);





% Computing distance between input face image and its reconstruction
% n = norm(v) returns the Euclidean norm of vector v.
distance = norm( Recon - input1 );




% concatenating all projected traning faces into single array. 
% C = horzcat(A1,...,AN) horizontally concatenates arrays A1,...,AN.
% All arrays in the argument list must have the same number of rows.
ProjectedImages = horzcat(face1,face2,face3,face4,face5,face6,face7,face8);


% Calculating Euclidean distances
% Euclidean distances between the projected test image and the projection
% of all centered training images are calculated. Test image is
% supposed to have minimum distance with its corresponding image in the
% training database.
% Euc_dist will have Euclidean distances between all training image and
% test image.

Euc_dist = [];
for i = 1 : 8
    q = ProjectedImages(:,i);
    temp = norm( ProjectedTestImage - q );
    Euc_dist = [Euc_dist temp];
end



% Minimum Euclidean distance is generated in Euc_dist_min
% Whichever distance is minimum between the projected test image and the projection
% of all centered training images, that training image's index is noted.
[Euc_dist_min , Recognized_index] = min(Euc_dist);


% The recognised output image name is saved and it is showed as Equivalent
% image compared to the test image. 
figure('Name','Test Image');
input1 = imresize(double(imread(Test1)),[231,195]);
subplot(1,1,1), pcolor(flipud(input1)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);
 
% the fetched recognized index is then saved in OutputName
% the training image at that index is fetched. 
OutputName = strcat(int2str(Recognized_index),'.jpg');
SelectedImage = strcat(TrainDatabasePath,OutputName);
figure('Name','Equivalent Image');
 
%output is then plotted into figure window.
output1 = imresize(double(imread(SelectedImage)),[231,195]);
subplot(1,1,1), pcolor(flipud(output1)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('Matched Training Image');
 


