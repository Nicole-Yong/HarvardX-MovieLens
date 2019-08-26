## Load the required libraries
library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(dplyr)
library(caret)
library(ggplot2)
library(tidyr)
library(hrbrthemes)
library(wordcloud)
library(RColorBrewer)

## Print session information
sessionInfo()


## Step 1: Data Gathering & Loading the Dataset
# Create train (edx) and test (validation) sets

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Ensure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)


## Step 2: Data Preprocessing

# Take a look at the train (edx) dataset
glimpse(edx)
summary(edx)
head(edx)

# Insights
# 1. No missing variables from summary table

# 2. Timestamp variable needs to be processed (converted to year)
edx <- mutate(edx, timestamp_year = year(as_datetime(timestamp)))

# 3. Genres variable needs to be processed (separate movies with multiple genres into individual rows per genre)
edx <- edx %>% separate_rows(genres, sep ="\\|")

# Take a look at the test (validation) dataset
glimpse(validation)
summary(validation)
head(validation)

# Insights
# 1. No missing variables from summary table

# 2. Timestamp variable needs to be processed (converted to year)
validation <- mutate(validation, timestamp_year = year(as_datetime(timestamp)))

# 3. Genres variable needs to be processed (separate movies with multiple genres into individual rows per genre)
validation <- validation %>% separate_rows(genres, sep ="\\|")


## Step 3: Data Exploration

# Explore number of unique movies and users in edx training set
n_distinct(edx$userId)
n_distinct(edx$movieId)

# Explore distribution of ratings in edx training set
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, fill="#69b3a2", color="#e9ecef") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 10000000, 1000000))) + 
  theme(plot.title = element_text(size=15), 
        axis.title = element_text(size=10),
        axis.text.x = element_text(size=8),
        axis.text.y = element_text(size=8)) +
  ggtitle("Distribution of Ratings (Training Set)") +
  xlab("Movie Ratings") + 
  ylab("Number of Ratings")

# Explore distribution of ratings in validation set
validation %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, fill="#69b3a2", color="#e9ecef") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 1000000, 100000))) +
  theme(plot.title = element_text(size=15), 
        axis.title = element_text(size=10),
        axis.text.x = element_text(size=8),
        axis.text.y = element_text(size=8)) +
  ggtitle("Distribution of Ratings (Validation Set)") +
  xlab("Movie Ratings") + 
  ylab("Number of Ratings")

# Insights
# Both train and test datasets have similar distribution of ratings
# More full ratings are given compared to half (0.5) ratings

# Explore number of ratings per unique movie
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, binwidth = 0.5, fill="#69b3a2", color="#e9ecef") +
  scale_x_log10() +
  theme(plot.title = element_text(size=15), 
        axis.title = element_text(size=10),
        axis.text.x = element_text(size=8),
        axis.text.y = element_text(size=8)) +
  ggtitle("Ratings per Movie") +
  xlab("Unique Movies") + 
  ylab("Number of Ratings")

# Explore number of ratings per unique user
edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, binwidth = 0.5, fill="#69b3a2", color="#e9ecef") +
  scale_x_log10() +
  theme(plot.title = element_text(size=15), 
        axis.title = element_text(size=10),
        axis.text.x = element_text(size=8),
        axis.text.y = element_text(size=8)) +
  ggtitle("Ratings per User")+
  xlab("Unique Users") + 
  ylab("Number of Ratings")

# Insights
# Certain movies have a higher number of ratings compared to others
# There are some movies with very few ratings
# Certain users rate more movies than other users

# Explore the genres of movies rated
count_genre <- edx %>% group_by(genres) %>% summarize(count_genre = n())
wordcloud(words = count_genre$genres, freq = count_genre$count_genre, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Spectral"))

# Insights
# Certain genres of movies garner more movie ratings
# However, note that a single movie can have multiple genres

## Step 4: Model Selection and Training

# Define RMSE (Root Mean Squared Error)
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Part 4.1 - Baseline RMSE
# To compute the baseline RMSE with the average rating computed from the training dataset

mu <- mean(edx$rating)
baseline_RMSE <- RMSE(edx$rating, mu)

# Test the baseline model against the test set
baseline_RMSE <- RMSE(validation$rating, mu)

# Add RMSE result to table
RMSE_results <- data_frame(Model = "Baseline RMSE", RMSE = baseline_RMSE)
RMSE_results %>% knitr::kable()

# Insights
# The goal of the subsequent models is to perform better than the baseline RMSE 
# To improve subsequent models using insights gained from data exploration

# Data Exploration Insights
# The top 3 ratings given by users are 4.0, 3.0 and 5.0 respectively 
# 0.5 (or half) ratings are less popular than whole ratings 
# Potential movie effect: Different movies have different ratings frequency and different ratings 
# Potential user effect: Certain users have rated more movies than other users

# Part 4.2 - Movie Effect
# Recap: Some movies have a higher (than mean) rating
# Account for estimated deviation of each movie's mean rating from total average mean

# Account for Movie Effect
movie_mu <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Test the adjusted model
# Add Movie Effect RMSE results to a table for later comparison
predicted_ratings <- mu +  validation %>%
  left_join(movie_mu, by='movieId') %>%
  pull(b_i)

adj_model_1_RMSE <- RMSE(validation$rating, predicted_ratings)

RMSE_results <- bind_rows(RMSE_results,
                          data_frame(Model="Movie Effect",  
                                     RMSE = adj_model_1_RMSE))
RMSE_results %>% knitr::kable()

# Part 4.3 - User Effect
# Recap: Different users rate movies differently 

# Account for movie effect and user effect in the predicted ratings 
user_mu <- edx %>%
  left_join(movie_mu, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Test adjusted model
# Add User Effect RMSE results to a table for later comparison
predicted_ratings <- validation%>%
  left_join(movie_mu, by='movieId') %>%
  left_join(user_mu, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

adj_model_2_RMSE <- RMSE(validation$rating, predicted_ratings)

RMSE_results <- bind_rows(RMSE_results,
                          data_frame(Model="Movie & User Effect",  
                                     RMSE = adj_model_2_RMSE))
RMSE_results %>% knitr::kable()

# Part 4.4 - Lambda (Regularization rate)
# The lambda (regularization parameter) reduces overfitting of the model
# It reduces the variance of the model's estimated regression parameters
# However, it adds bias to the estimate
# Therefore, try out various lambda values and select the lambda value that produces the smallest RMSE value

lambdas <- seq(0, 10, 0.25)

# For each lambda value, find b_i, b_u, predicted rating & test
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})

# Add results to a table for later comparison
RMSE_results <- bind_rows(RMSE_results,
                          data_frame(Model="Regularized Model",  
                                     RMSE = min(rmses)))

## Step 5: Evaluation
RMSE_results %>% knitr::kable()

# Conclusion
# The Regularized Model produces the lowest RMSE value