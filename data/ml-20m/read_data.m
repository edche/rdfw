movieID = csvread('ratings.csv',1,1,[1 1 20000263 1]);
save('movieID.mat', 'movieID');
rating = csvread('ratings.csv',1,2,[1 2 20000263 2]);
save('rating.mat', 'rating');