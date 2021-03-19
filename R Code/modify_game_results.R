library(tidyr)
library(readr)
library(dplyr)

t_games <- read_csv("game_results.csv")
t_games$Team <- as.factor(t_games$Team)
t_games$Team_1 <- as.factor(t_games$Team_1)
levels(t_games$Team)
levels(t_games$Team_1)

#Duplicating data frame
d_games <- t_games

#Flipping Team and Team_1 and changing name to Opponent instead of Team_1
d_games$Opponent <- d_games$Team
d_games$Opp_Score <- d_games$Score
d_games$Opp_Seed <- d_games$Seed

d_games$Team <- d_games$Team_1
d_games$Score <- d_games$Score_1
d_games$Seed <- d_games$Seed_1

d_games$Team_1 <- NULL
d_games$Score_1 <- NULL
d_games$Seed_1 <- NULL

#Changing name to Opponent instead of Team_1
t_games$Opponent <- t_games$Team_1
t_games$Opp_Score <- t_games$Score_1
t_games$Opp_Seed <- t_games$Seed_1

t_games$Team_1 <- NULL
t_games$Score_1 <- NULL
t_games$Seed_1 <- NULL

#Merging data frames so that rows are alternating
t_games$Game <- c(1:nrow(t_games))
d_games$Game <- c(1:nrow(d_games))

games <- rbind(t_games, d_games) %>% arrange(Game)
games$Game <- NULL

#Adding result of Win which is 1 if Team wins and 0 if Team loses
games <- games %>% mutate(Win = case_when(Score > Opp_Score ~ 1, Score < Opp_Score ~ 0))
games <- games %>% filter(Year >= 2001)
write.csv(games, file = "Tournament_Games.csv")
