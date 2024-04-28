# Load packages
pacman::p_load(tidyverse, ggplot2, scales, glmnet, pls, randomForest, rpart, DescTools, 
               readxl, corrplot, ggcorrplot, factoextra, ggfortify, clustMixType, wesanderson, 
               caret, rpart.plot, xgboost, reshape2, MLmetrics, SHAPforxgboost, ROSE, DiagrammeR)

# Set custom theme
MK_theme <-   theme(text = element_text(family = "Times New Roman"),
                    panel.background = element_blank(),
                    axis.text.x.bottom = element_text(size = 12),
                    axis.text.y.left = element_text(size = 12),
                    axis.title.x.bottom = element_text(size = 16),
                    axis.title.y.left = element_text(size = 16),
                    title = element_text(size = 16, face = "bold"),
                    panel.border = element_rect(color = "black", linewidth = 0.3, fill = "transparent"),
                    panel.grid = element_line(color = "black", linewidth = 0.1),
                    legend.position = "none")


# 1. Go to Blackboard and download the files. Starting with the CFPB complaint data, please load the data into R and clean the data set. Please note, each team has a different random sample.

## a. I am providing the code for the dependent variable “relief” from the company response variable. Please note, you will need to drop the company response variable before you do the modeling as this will perfectly predict the DV.

## b. Your group needs to make decisions on how to treat the rest of this variable. In your write up, defend your answer.

## i. How should “In progress” be coded?

## ii. Should the other conditions be coded into dummy variables?

## c. All of the CFPB data documentation is located at <https://cfpb.github.io/api/ccdb/fields.html> .

## i. While not required, it would be in your best interest to recode some of these variables to have shorter values in the name.


# Bring in data 
group1 <- read_csv("//apporto.com/dfs/GWU/Users/lzou30_gwu/Downloads/group1.csv")
colnames(group1)
group1 <- group1[,-1]

# Add leading zero
add_char <- function(x, pos, insert) {      
  gsub(paste0("^(.{", pos, "})(.*)$"),
       paste0("\\1", insert, "\\2"),
       x)
}

which(nchar(group1$ZIP.code) == 4) # no four-digit zip codes
group1.1 <- group1 %>%
  filter(State != "AE" & State != "AP") # Remove non-state observations

# Load in zip_fips
zip_fips <- read_csv("//apporto.com/dfs/GWU/Users/lzou30_gwu/Downloads/zip_fips.csv")
colnames(zip_fips)
zip_fips <- zip_fips[,-1]
str(zip_fips$ZIP)
table(zip_fips$STATE)

# Filter non-US states
zip_fips_states <- zip_fips %>%
  filter(STATE != "PR" & STATE != "GU" & STATE != "DC" & STATE != "VI")

# Add a leading zero if zip is only four characters long
zip_fips_states$ZIP <- ifelse(nchar(zip_fips_states$ZIP) == 4, add_char(x = zip_fips_states$ZIP,
                                                                        pos = 0, 
                                                                        insert = "0"), zip_fips_states$ZIP)

# Merge datasets to find odd zip codes
group1.2 <- merge(group1.1, zip_fips_states, 
                  by.x = c("ZIP.code", "State"),
                  by.y = c("ZIP", "STATE"), all.x = TRUE)

group1.2$missing_zip <- as.factor(is.na(group1.2$STCOUNTYFP))
group1.2 <- group1.2[,-15] # Drop company response to consumer b/c it is perfectly correlated with relief



# 2. Your first task is to clean the zip code variable. Many of the zip codes are missing, incomplete, or missing the leading zero. Your group must pick a missing data method to fix this data!

## a. For the dependent variable (relief) and 3 other variables of your choosing, do a simple statistical comparison (2 sample proportions or t-test, Chi-Square,…) to check for differences between the subgroups that are missing zip code and those that provided the information.

## i. Suggestion: You will want to create a dummy variable to indicate if the variable is missing for the rest of the analysis. Using the aggregate() function in r, this should be an easy task.


# Submitted.via
table(group1.2$Submitted.via, group1.2$missing_zip)
group1.3 <- group1.2 %>%
  mutate(Submitted.via = case_when(Submitted.via == "Web Referral" ~ "Referral",
                                   Submitted.via == "Web" ~ "Web", 
                                   Submitted.via == "Referral" ~ "Referral",
                                   Submitted.via == "Postal mail" ~ "Postal mail",
                                   Submitted.via == "Phone" ~ "Phone",
                                   Submitted.via == "Fax" ~ "Fax")) # Recode Web Referral to Referral
table(group1.3$Submitted.via, group1.3$missing_zip) # Meets assumptions now :) 
chisq.test(group1.3$Submitted.via, group1.3$missing_zip) # p-value significant

# Tags
table(group1.3$Tags, group1.3$missing_zip)
fisher.test(group1.3$Tags, group1.3$missing_zip) # Does not meet assumptions

# Relief
table(group1.3$relief, group1.3$missing_zip)
chisq.test(group1.3$relief, group1.3$missing_zip) # p-value = 1

# Timely.response
table(group1.3$Timely.response., group1.3$missing_zip)
chisq.test(group1.3$Timely.response., group1.3$missing_zip) # p-value *almost* significant



## b. Please note, you have the state in all cases. You must fix the fill this field with a zip code that is a plausible value (the corrected/imputed zip code must be from the State listed).

## i. You can drop any non-state in the analysis. (“DC”, “UNITED STATES MINOR OUTLYING ISLANDS”, “AA” (Armed Forces), …)

## c. Any imputation method is acceptable as long as you defend your choice. You can use mean/mode replacement, nearest neighbor, regression based models, …


# Use mode to impute zip code
group1.4 <- group1.3 %>%
  group_by(State) %>%
  mutate(imp_zip = ifelse(missing_zip == TRUE, Mode(ZIP.code),  ZIP.code))



## d. Please indicate any limitations or biases your imputation has on the analysis?

## i. If I replace all values with the state capital zip code, how would that impact my analysis?

# 3. Merge on the FIPS county code for each Zip code I have provided in the folder.

## a. All zip codes should have a FIPS county code. If it does not,

## i. Check to make sure you formatted the data right (i.e. 2019 will not merge with 02019 correctly if both are characters.)

## ii. If the zip code is nonexistent (someone made it up, like 99999), then you need to treat it as missing and replace it using the same method as in part 2.

## iii. \*\*\* Zip codes to change, and it is possible that the Zip to Fips code file I am using from last year misses a new observation. Just treat it as missing and move on.


group1.5 <- group1.4[,-c(19:20)] # remove COUNTYNAME & STCOUNTYFP
group1.6 <- merge(group1.5, zip_fips_states, by.x = c("imp_zip", "State"), by.y = c("ZIP", "STATE")) # merge


# 4. Merge on county level debt information from The Urban Institute for 2022 on by county FIPS code.

## a. Suggestion: Be careful how you load the data to make sure missing values do not change everything to text.

## b. There is other debt data at the site above if you want to add it also, but it is not required.

## c. Remember to do left joins/merges on the data. The data file also has state totals you want to drop from the analysis.


# Bring in data from the Urban Institute
county_debt <- read_excel("//apporto.com/dfs/GWU/Users/lzou30_gwu/Downloads/dia_lbls_all_overall_county_2022_02_14Sep2023.xlsx")

# Change column names
names(county_debt)[names(county_debt) == "County FIPS"] <- "fips"

# Add a leading zero to the group1.6 FIPS code
group1.6$fips <- ifelse(nchar(group1.6$STCOUNTYFP) == 4, add_char(x = group1.6$STCOUNTYFP,
                                                                  pos = 0, 
                                                                  insert = "0"), group1.6$STCOUNTYFP)

# Merge county debt data with group1.6
group1.7 <- merge(group1.6, county_debt, by = "fips", all.x = TRUE)



# 5. Load the Census demographic file into R. Before you merge it into the debt collection data, you need to combine the State and County FIPs to merge the data and then develop features. Each group need to add measures that would indicate bias in debt collection against:

## a. Gender

## b. Age

## i. Bias against young (0-24 years) and old (65+) are of particular concern

### ii. You need to create a dummy variable for county level older Americans.

## c. Race

## d. Ethnicity

### i. Hispanic vs. non-Hispanic at minimum.

## e. Also, from the original CFPB dataset create dummy variables for Service Members and Older American based on the Tags column.


# Read in Census data
census <- read_csv("//apporto.com/dfs/GWU/Users/lzou30_gwu/Downloads/cc-est2022-all.csv")
colnames(census)

# Add leading zeros to county and state FIPS
census$STATE2 <- sprintf("%02s", census$STATE)
census$COUNTY2 <- sprintf("%03s", census$COUNTY)

# Append state to county FIPS
census$STCOUNTY <- paste0(census$STATE2, census$COUNTY2)

# Move STCOUNTY to the first column of the dataset
census <- census %>%
  relocate(STCOUNTY, .before = SUMLEV)

# Subsetting for the total column (where AGEGRP = 0)
census1 <- subset(census, AGEGRP == 0)

# Find the mean grouped by STCOUNTY, AGEGRP because there are four years we need to collapse
census2 <- census %>%
  group_by(STCOUNTY, AGEGRP) %>%
  summarize(across(TOT_POP:HNAC_FEMALE, ~mean(., na.rm = TRUE)))

# Rename columns
colnames(census2) <- paste0(colnames(census2), "_MEAN")
names(census2)[names(census2) == "STCOUNTY_MEAN"] <- "STCOUNTY"
names(census2)[names(census2) == "AGEGRP_MEAN"] <- "AGEGRP"

# Subset to total 
census_total <- census2 %>%
  filter(AGEGRP == 0)

# Subsetting for youth (0-24)
censusyoung <- census2 %>%
  filter(AGEGRP > 0 & AGEGRP <= 5)

# Subsetting for elderly (65+)
censusold <- census2 %>%
  filter(AGEGRP >= 14 & AGEGRP <= 18)

# Find the sum of young people by county
censusyoungsum <- censusyoung %>%
  group_by(STCOUNTY) %>%
  summarize(
    YOUTH_MALE = sum(TOT_MALE_MEAN),
    YOUTH_FEMALE = sum(TOT_FEMALE_MEAN),
  )

# Find the sum of elderly people by county
censusoldsum <- censusold %>%
  group_by(STCOUNTY) %>%
  summarize(
    OLD_MALE = sum(TOT_MALE_MEAN),
    OLD_FEMALE = sum(TOT_FEMALE_MEAN),
  )

# Merging the young/old variables with the rest of the census data
census3 <- merge(census_total, censusyoungsum, by = "STCOUNTY")
census4 <- merge(census3, censusoldsum, by = "STCOUNTY")

# Creating variables to indicate bias
census4 <- census4 %>%
  mutate(pct_old = (OLD_FEMALE + OLD_MALE)/TOT_POP_MEAN, 
         pct_young = (YOUTH_FEMALE + YOUTH_MALE)/TOT_POP_MEAN,
         pct_blk = (BA_FEMALE_MEAN + BA_MALE_MEAN)/TOT_POP_MEAN,
         pct_asian = (AA_FEMALE_MEAN + AA_MALE_MEAN)/TOT_POP_MEAN,
         pct_native = (IA_FEMALE_MEAN + IA_MALE_MEAN)/TOT_POP_MEAN,
         pct_hisp = (H_FEMALE_MEAN + H_MALE_MEAN)/TOT_POP_MEAN,
         pct_women = TOT_FEMALE_MEAN/TOT_POP_MEAN, 
         county_old = ifelse(pct_old > 0.168, 1, 0)) # Share of elderly population in America in 2020 was 16.8%, if a county has over this threshold, it is considered to be an older county 




# 6. Once you have taken all the Census data and created a data frame of demographic measures with one row for each county, merge it onto your debt collection data frame.


# Merge debt data with Census data
group1.8 <- merge(group1.7, census4, by.x = "fips", by.y = "STCOUNTY")

# Create a flag for servicemembers and older Americans
group1.8$service_mem <- ifelse(grepl("Servicemember", group1.8$Tags), 1, 0)
group1.8$old <- ifelse(grepl("Older American", group1.8$Tags), 1, 0)
group1.8$old_service_mem <- ifelse(grepl("Older American, Servicemember", group1.8$Tags), 1, 0)
group1.8$none <- ifelse(grepl("None", group1.8$Tags), 1, 0)



# 7. The debt collection variables are highly correlated. Take all or a subset of at least 5 of them and try to create a principal component out of them. Merge the resulting principal component back onto the main data set.

## a. You may choose to keep only a subset in the final PCA analysis, but you must start with at least 5 variables.

### i. Often we start with a larger number but drop the ones that don’t fit.

## b. You are welcome to keep more than one PC.

### i. Maybe 3 variables load on one and 4 on another PC. Then just load them as two new variables.

## c. You may also try PCA on other variables in the data set to see if you get a better model as part of your feature selection.

## d. Suggestion: Examine the percent missing for each variable as you are selecting variables.


# Random forest for missing data in the Urban Institute data
debt_subset_rf <- county_debt[,-(2:3)]
debt_subset_rf2 <- debt_subset_rf %>%
  sapply(as.numeric) %>%
  as.data.frame()
debt_subset_rf2$is_complete <- as.factor(complete.cases(debt_subset_rf2)) # variable for complete cases

set.seed(19745) # set random seed
debt_subset_rf3 <- rfImpute(fips~., debt_subset_rf2[,-27]) # impute using RF
colnames(debt_subset_rf3) <- abbreviate(colnames(debt_subset_rf3), minlength = 4) # change column names

# Get correlation by removing fips code
debt_matrix_rf <- cor(debt_subset_rf3[,-1])
ggcorrplot(debt_matrix_rf)

# Subset variables that will be included in PCA (based on the correlation matrix)
pca_vars <- debt_subset_rf3 %>%
  select(c(SwadicWc, SwadicA, SwmdicA, SwmdicCoc, SwmdicWc, AhiA, AhiWc, AhiCoc)) 

# Perform PCA on selected variables and save the scores/loadings
debt_pca_rf <- prcomp(pca_vars, scale = TRUE)
summary(debt_pca_rf)
fviz_eig(debt_pca_rf, addlabels = TRUE)
loadings <- debt_pca_rf$rotation
scores <- debt_pca_rf$x

# Remove variables used in PCA from the dataset and add the principal components
county_debt1 <- debt_subset_rf3 %>%
  select(-c(SwadicWc, SwadicA, SwmdicA, SwmdicCoc, SwmdicWc, AhiA, AhiWc, AhiCoc))
county_debt1 <- cbind(county_debt1, scores[, 1:2])

# Add leading zero to fips before merging back onto rest of data
county_debt1$fips2<- ifelse(nchar(county_debt1$fips) == 4, add_char(x = county_debt1$fips,
                                                                    pos = 0, 
                                                                    insert = "0"), county_debt1$fips)


# Merge back with the rest of the data
group1.9 <- group1.8[,-c(24:50)]
group1.9 <- merge(group1.9, county_debt1[,-1], by.x = "fips", by.y = "fips2", all.x = TRUE)



# 8. Do a mixed-type Cluster analysis using the R package clustMixType. Create a clustering based on medical debt.

## a. Make a binary variable out of Sub Product == Medical Debt.

## b. Pick 5 other variables that you believe may be related to medical debt. There must be at least 2 factors and 2 continuous.

## c. Save the clusters as output and add them back to the main data set.


# Make binary variable out of Sub product == Medical Debt
group1.9$medical_debt <- ifelse((group1.9$Sub.product == "Medical debt" | group1.9$Sub.product == "Medical"), 1, 0)

z <- group1.9[, c("medical_debt", "old_service_mem",
                  "pct_old", "pct_young",
                  "SoslhwsldidA", "Consumer.disputed.")]

z[,1]<- as.factor(z[,1])
z[,2]<- as.factor(z[,2])
z[,3]<- scale(as.numeric(z[,3]))
z[,4] <- scale(as.numeric(z[,4]))
z[,5]<- scale(as.numeric(z[,5]))
z[,6] <- as.factor(z[,6])

set.seed(19745)

# 2 
kpres2 <-kproto(x = z,k=2)
kpres2

# 4 
kpres4 <-kproto(x = z,k=4)
kpres4

# 5 
kpres5 <- kproto(x = z, k = 5)
kpres5

# Scree plot
n.scree <- ncol(z)
Es <- numeric(n.scree)
for(i in 1:n.scree){ 
  kpres <- kproto(z, k = i, nstart = 5, verbose = FALSE) 
  Es[i] <- kpres$tot.withinss
} 

plot(1:n.scree, Es[1:6], type = "b", ylab = "Objective Function",
     xlab = "# Clusters", main = "ScreePlot")

# Add with four clusters (based on scree plot)
group1.9$cluster4 <- kpres4$cluster



# 9. You are about to start doing supervised machine learning to predict debt relief. Before you proceed any further, sit with your group and make sure you are content with the measurement of all the variables.

## a. Check missing values on merged data. Impute/delete as necessary and document accordingly.

## b. Identify which variables should be in the model (i.e. Should date of complaint be included?) You do not need to keep all of the variables, you can choose to drop the ones you think add less value. You should at least 2 keep variables from each data set.

## c. Check the measurement of each variable (i.e. Should date be measure, by day or year?) Should something be logged?

## d. Check all the factors to make sure they have enough observations at each level.

## e. Optional: You may want to make a design matrix that that includes only the independent variables you want in the model and drops the id variables such as zip code and Complaint ID.

## f. You may add Clusters and principal components from other variables if you think it will help.

## g. You are absolutely allowed to add additional variables from other public sources to your dataset. This is in no way required.


# Turn some variables into factors
lapply(group1.9, is.factor)
group1.9$Sub.product <- as.factor(group1.9$Sub.product)
group1.9$Issue <- as.factor(group1.9$Issue)
group1.9$Company.public.response <- as.factor(group1.9$Company.public.response)
group1.9$Submitted.via <- as.factor(group1.9$Submitted.via)
group1.9$Timely.response. <- as.factor(group1.9$Timely.response.)
group1.9$cluster4 <- as.factor(group1.9$cluster4)

# Check factor levels before removing variables
nlevels(as.factor(group1.9$Company)) # over 2000 levels with very few obs. per level

# Drop medical debt, old_service_mem, SoslhwsldidA, consumer disputed, pct_old out of the data because they are represented in the cluster. Drop state, zip code (keep the imputed one), county, product (all debt collection), customer complaint narrative (unique), complaint ID (unique), date sent (because it is nearly identical to date received), company (over 2000 levels)
group2.0 <- group1.9 %>%
  select(-c(State, ZIP.code, Sub.issue, Company.public.response, Consumer.disputed., pct_young, SoslhwsldidA, old_service_mem, medical_debt, Consumer.complaint.narrative, Complaint.ID, Product, COUNTYNAME, STCOUNTYFP, AGEGRP, Company, Date.sent.to.company, Tags))

# Extract year from date received
group2.0$year <- format(as.Date(group2.0$Date.received, format="%m/%d/%y"),"%y")
group2.0$year <- paste0("20", group2.0$year)

# Drop most of the Census variables, date received (because we have the year now)
group2.1 <- group2.0[, -c(1:3, 11:87)]

# Recode levels of subproduct to collapse alike levels
group2.1 <- group2.1 %>%
  mutate(Sub.product2 = case_when(Sub.product == "Auto" ~ "Auto debt",
                                  Sub.product == "Credit card" ~ "Credit card debt",
                                  Sub.product == "Federal student loan" ~ "Student loan debt",
                                  Sub.product == "Non-federal student loan" ~ "Student loan debt",
                                  Sub.product == "Federal student loan debt" ~ "Student loan debt",
                                  Sub.product == "Private student loan debt" ~ "Student loan debt",
                                  Sub.product == "Mortgage" ~ "Mortgage debt",
                                  Sub.product == "Other (i.e. phone, health club, etc.)" ~ "Other debt",
                                  Sub.product == "Payday loan" ~ "Payday loan debt",
                                  Sub.product == "Medical" ~ "Medical debt",
                                  .default = Sub.product))
table(group2.1$Sub.product2)

# Remove the original Sub.product variable
group2.1 <- group2.1 %>%
  select(-c(Sub.product))



# 10. Using k-fold cross validation, determine the best machine learning model to predict debt relief. At minimum you must run:

## a. A linear probability model (can be OLS, ridge, lasso, …)

## b. A logistic model

## c. A CART / regression tree

## d. Random Forest

## e. Gradient Boosting/ XGBoost

# Choosing variables; removing race variables because we have Sopoc (share of people of color by county), removing White community specific variables because we have the share of white people through the share of people of color by county, and removing community of color variables because they were ~80% missing and we imputed that data, so it's likely not accurate even if it comes back as a significant predictor in our models
group2.2 <- group2.1 %>%
  select(-c(pct_hisp, pct_asian, pct_blk, pct_native, MdicWc, CcddrWc, AldrWc, MccddWc, SoslhwsldidWc, SwadicCoc, MdicCoc, AldrCoc, CcddrCoc, MccddCoc, SoslhwsldidCoc)) %>%
  mutate(relief = as.factor(relief))

# Oversample the data 
group2.3 <- ovun.sample(relief ~ ., data = group2.2, method = "over", seed = 182654)
group2.3 <- as.data.frame(group2.3$data) # coerce to dataframe

### LOGIT WITH RIDGE REGULARIZATION
# 5-fold CV
set.seed(182654)
train_control <- trainControl(method = "cv", 
                              number = 5)
default_idx = createDataPartition(group2.3$relief, p = 0.75, list = FALSE) # use 75% of data in training
default_trn = group2.3[default_idx, ] # partition into train
default_tst = group2.3[-default_idx, ] # partition into test

grid <- expand.grid(alpha = 0, lambda = seq(0.0001, 1, length = 100))
logit.ridge <- train(relief ~., data = default_trn, trControl = train_control, 
                     method = "glmnet", family = "binomial", tuneGrid = grid,
                     preProcess = c("center", "scale"))

# Prediction
ridge.predict <- predict(logit.ridge, default_tst)
table(Predicted = ridge.predict, Actual = default_tst$relief) 
ridge.accuracy <- (5947 + 6272)/(5947 + 6272 + 4072 + 3702) # 0.6111639
ridge.F1 <- F1_Score(ridge.predict, default_tst$relief) # 0.6047387

# Get predicted probabilities and plot the QQ plot
predicted_probs <- predict(logit.ridge, newdata = default_tst, type = "prob")
predicted_quantiles <- quantile(predicted_probs[,2], probs = seq(0, 1, by = 0.01)) # assuming 2nd column is the probability of positive class
theoretical_quantiles <- qnorm(seq(0, 1, by = 0.01))
plot(theoretical_quantiles, predicted_quantiles, 
     xlab = "Theoretical Quantiles", ylab = "Predicted Quantiles",
     main = "QQ Plot for Logistic Regression w/ Ridge Model")
abline(0, 1, col = "red")

### OLS & STEPWISE REGRESSION
# OLS
set.seed(182654)
default_trn$relief <- as.numeric(as.character(default_trn$relief))
default_tst$relief <- as.numeric(as.character(default_tst$relief))

lm <- train(relief~., data = default_trn, trControl = train_control,
            method = "lm")

lm.predict <- predict(lm, default_tst)
lm.predict <- round(lm.predict, 0)
table(Predicted = lm.predict, Actual = default_tst$relief)
lm.accuracy <- (5969 + 6265)/(5969 + 6265 + 4050 + 3709) # 0.6119142
lm.F1 <- F1_Score(lm.predict, default_tst$relief) # 0.6060821
lm.residuals <- residuals(lm$finalModel) # get the residuals
qqnorm(lm.residuals) # qqplot of the residuals; clearly not Normal, but I wouldn't expect it to be


# STEPWISE
lm.step <- train(relief~., data = default_trn, trControl = train_control,
                 method = "leapSeq")

lm.step.predict <- predict(lm.step, default_tst)
lm.step.predict <- round(lm.step.predict, 0)
table(Predicted = lm.step.predict, Actual = default_tst$relief)
lm.step.accuracy <- (7281 + 4128)/(7281 + 4128 + 2738 + 5846) # 0.5706497
lm.step.F1 <- F1_Score(lm.step.predict, default_tst$relief) # 0.6291368

### CART/REGRESSION TREE
set.seed(182654)
default_trn$relief <- as.factor(default_trn$relief)
default_tst$relief <- as.factor(default_tst$relief)
cp_grid <- data.frame(cp = seq(0.001, .1, .01)) # grid for complexity parameter
relief.tree <- train(relief ~., data = default_trn, 
                     method = "rpart", 
                     tuneGrid = cp_grid,
                     trControl = train_control) # chose cp = 0.001
print(relief.tree) # results
prp(relief.tree$finalModel) # plot tree
relief.tree$finalModel$variable.importance # plot variable importance
tree.predict <- predict(relief.tree, default_tst) 
table(Predicted = tree.predict, Actual = default_tst$relief)
tree.accuracy <- (7165 + 5055)/(7165 + 5055 + 2854 + 4919) # 0.6112139
tree.F1 <- F1_Score(tree.predict, default_tst$relief) # 0.6483283

### RANDOM FOREST
set.seed(182654) # set random seed
default_trn$relief <- as.factor(default_trn$relief) # make relief a factor again 
default_tst$relief <- as.factor(default_tst$relief)
relief.rf <- train(relief ~., data = default_trn, 
                   method = "rf",
                   trControl = train_control, ntree = 100)
relief.rf
rf.predict <- predict(relief.rf, default_tst, method = "class")
table(Predicted = rf.predict, Actual = default_tst$relief)
rf.accuracy <- (8371 + 9686)/(8371 + 9686 + 1648 + 288) # 0.9031661
rf.F1 <- F1_Score(rf.predict, default_tst$relief) # 0.8963486

### XGBOOST
# Run a XGBoost model with the default parameters.
set.seed(182654)
xtrain <- sparse.model.matrix(relief ~ .-1, data = default_trn)
ytrain <- as.array(default_trn$relief)
xtest <- sparse.model.matrix(relief ~ .-1, data = default_tst)
ytest <- as.array(default_tst$relief)

xgb1 <- xgboost(data = xtrain, label = as.numeric(ytrain) - 1, nrounds = 100, objective = "binary:logistic")

# Predictions
boost.predict <- predict(xgb1, xtest)
table(Predicted = ifelse(boost.predict > 0.5, 1, 0), Actual = ytest)
boost.accuracy <- (6939 + 7470)/(6939 + 7470 + 3080 + 2504) # 0.7207022
boost.F1 <- F1_Score(ifelse(boost.predict > 0.5, 1, 0), ytest) # 0.7130819

### Perform a grid search for the proper parameters ###
# = eta candidates = # 
eta = c(0.05, 0.1, 0.2, 0.5, 1) 

# = colsample_bylevel candidates = # 
cs = c(1/3, 2/3, 1) 

# = max_depth candidates = # 
md = c(2, 4, 6, 10) 

# = sub_sample candidates = # 
ss = c(0.25, 0.5, 0.75, 1) 

# = standard model is the second value of each vector above = # 
standard = c(2, 2, 3, 2) 

# = min_child_weights candidates = # 
mcw = c(1, 10, 100, 400) 

# = gamma candidates = # 
gamma = c(0.1, 1, 10, 100) 

### Eta search
set.seed(13856) 
conv_eta = matrix(NA,500,length(eta)) 
pred_eta = matrix(NA,nrow(default_tst), length(eta))
colnames(conv_eta) = colnames(pred_eta) = eta 
for(i in 1:length(eta)){ 
  params=list(eta = eta[i], colsample_bylevel=cs[standard[2]], 
              subsample = ss[standard[4]], max_depth = md[standard[3]], 
              min_child_weigth = 1) 
  xgb=xgboost(xtrain, label = ytrain, nrounds = 500, params = params) 
  conv_eta[,i] = xgb$evaluation_log$train_rmse 
  pred_eta[,i] = predict(xgb, xtest) } 
conv_eta = data.frame(iter=1:500, conv_eta) 
conv_eta = melt(conv_eta, id.vars = "iter") 
ggplot(data = conv_eta) + geom_line(aes(x = iter, y = value, color = variable))

# Find RMSE
(RMSE_eta = sqrt(colMeans((as.numeric(ytest)-pred_eta)^2))) # 0.5

########################### 
# colsample_bylevel 
set.seed(1284654) 
conv_cs = matrix(NA,500,length(cs)) 
pred_cs = matrix(NA,nrow(default_tst), length(cs)) 
colnames(conv_cs) = colnames(pred_cs) = cs 
for(i in 1:length(cs)){ 
  params = list(eta = eta[standard[1]], colsample_bylevel = cs[i], 
                subsample = ss[standard[4]], max_depth = md[standard[3]],
                min_child_weigth = 1) 
  xgb=xgboost(xtrain, label = ytrain,nrounds = 500, params = params) 
  conv_cs[,i] = xgb$evaluation_log$train_rmse 
  pred_cs[,i] = predict(xgb, xtest)}

conv_cs = data.frame(iter=1:500, conv_cs) 
conv_cs = melt(conv_cs, id.vars = "iter") 
ggplot(data = conv_cs) + geom_line(aes(x = iter, y = value, color = variable)) 

# Find RMSE
(RMSE_cs = sqrt(colMeans((as.numeric(ytest)-pred_cs)^2))) # 1

##########################
## Max Depth 
set.seed(1284654) 
conv_md=matrix(NA,500,length(md)) 
pred_md=matrix(NA,nrow(default_tst),length(md)) 
colnames(conv_md)=colnames(pred_md)=md 
for(i in 1:length(md)){ 
  params=list(eta=eta[standard[1]],
              colsample_bylevel=cs[standard[2]], subsample=ss[standard[4]],
              max_depth=md[i], min_child_weigth=1) 
  xgb=xgboost(xtrain, label = ytrain,nrounds = 500,params=params) 
  conv_md[,i] = xgb$evaluation_log$train_rmse 
  pred_md[,i] = predict(xgb, xtest) } 
conv_md=data.frame(iter=1:500,conv_md) 
conv_md=melt(conv_md,id.vars = "iter")
ggplot(data=conv_md)+geom_line(aes(x=iter,y=value,color=variable))

# Find RMSE
(RMSE_md=sqrt(colMeans((as.numeric(ytest)-pred_md)^2))) # 10

##########################
## sub_sample 
set.seed(1284654) 
conv_ss=matrix(NA,500,length(ss)) 
pred_ss=matrix(NA,nrow(default_tst),length(ss)) 
colnames(conv_ss)=colnames(pred_ss)=ss 
for(i in 1:length(ss)){ 
  params=list(eta=eta[standard[1]], 
              colsample_bylevel=cs[standard[2]], subsample=ss[i],
              max_depth=md[standard[3]], min_child_weigth=1) 
  xgb=xgboost(xtrain, label = ytrain,nrounds = 500,params=params) 
  conv_ss[,i] = xgb$evaluation_log$train_rmse 
  pred_ss[,i] = predict(xgb, xtest) } 
conv_ss=data.frame(iter=1:500,conv_ss) 
conv_ss=melt(conv_ss,id.vars = "iter")
ggplot(data=conv_ss)+geom_line(aes(x=iter,y=value,color=variable))

# Find RMSE
(RMSE_ss=sqrt(colMeans((as.numeric(ytest)-pred_ss)^2))) # 0.75


############################
## min_child_weight 
set.seed(12754) 
conv_mcw = matrix(NA,500,length(mcw)) 
pred_mcw = matrix(NA,nrow(default_tst), length(mcw)) 
colnames(conv_mcw) = colnames(pred_mcw) = mcw 
for(i in 1:length(mcw)){ 
  params = list(eta = 0.5, colsample_bylevel=1, 
                subsample = 0.75, max_depth = 10, min_child_weight = mcw[i], gamma = 0.1) 
  xgb = xgboost(xtrain, label = ytrain, nrounds = 500, params = params) 
  conv_mcw[,i] = xgb$evaluation_log$train_rmse 
  pred_mcw[,i] = predict(xgb, xtest) } 
conv_mcw = data.frame(iter=1:500, conv_mcw) 
conv_mcw = melt(conv_mcw, id.vars = "iter")
ggplot(data = conv_mcw) + geom_line(aes(x = iter, y = value, color = variable)) 

# Find RMSE
(RMSE_mcw = sqrt(colMeans((as.numeric(ytest)-pred_mcw)^2))) # 1


######################## 
### Gamma
set.seed(12897564) 
conv_gamma = matrix(NA,500,length(gamma)) 
pred_gamma = matrix(NA,nrow(default_tst), length(gamma)) 
colnames(conv_gamma) = colnames(pred_gamma) = gamma 
for(i in 1:length(gamma)){ 
  params = list(eta = 0.5, colsample_bylevel=1, 
                subsample = 0.75, max_depth = 10, min_child_weight = 1, 
                gamma = gamma[i]) 
  xgb = xgboost(xtrain, label = ytrain, nrounds = 500, params = params) 
  conv_gamma[,i] = xgb$evaluation_log$train_rmse 
  pred_gamma[,i] = predict(xgb, xtest) } 
conv_gamma = data.frame(iter=1:500, conv_gamma) 
conv_gamma = melt(conv_gamma, id.vars = "iter") 
ggplot(data = conv_gamma) + geom_line(aes(x = iter, y = value, color = variable))

# Find RMSE
(RMSE_gamma = sqrt(colMeans((as.numeric(ytest)-pred_gamma)^2))) # 0.1

### Running the model with the correct parameters
# Setting hyperparameter values
set.seed(194564)
params = list(eta = 0.5, colsample_bylevel = 1, subsample = 0.75 , max_depth = 10, min_child_weight = 1, gamma = 0.1)
xgb2 <- xgboost(data = xtrain, label = as.numeric(ytrain) - 1, params = params,
                nrounds = 500, objective = "binary:logistic")

# Predictions
boost.predict2 <- predict(xgb2, xtest) 
table(Predicted = ifelse(boost.predict2 > 0.5, 1, 0), Actual = ytest)
boost.accuracy2 <-  (8285 + 9577)/(8285 + 9577 + 1734 + 397) # 0.8934127 (eta = 0.5, colsample_bylevel = 1, subsample = 0.75 , max_depth = 10, min_child_weight = 1, gamma = 0.1)
boost.F1.2 <- F1_Score(ifelse(boost.predict2 > 0.5, 1, 0), ytest) # 0.8860489

# Get importance matrix
importance_matrix <- xgb.importance(model = xgb2)
xgb.plot.importance(importance_matrix, xlab = "Feature Importance")

# Plot a few trees to see interactions
# xgb.plot.tree(model = xgb2, trees = 20)



# RUN ON TEST DATA (HAS NOT BEEN UPDATED 4/27)

test_all <- read_csv("//apporto.com/dfs/GWU/Users/lzou30_gwu/Downloads/test_all.csv")
test_all1 <- test_all[, c(3:21, 23)]

table(test_all1$State)
test_all1.1 <- test_all1 %>%
  filter(State != "AE" & State != "AP") # Remove non-state observations

# Filter non-US states
zip_fips_states <- zip_fips %>%
  filter(STATE != "PR" & STATE != "GU" & STATE != "DC" & STATE != "VI")

# Merge datasets to find odd zip codes
test_all1.2 <- merge(test_all1.1, zip_fips_states,
                     by.x = c("ZIP.code", "State"),
                     by.y = c("ZIP", "STATE"), all.x = TRUE)

test_all1.2$missing_zip <- as.factor(is.na(test_all1.2$STCOUNTYFP))
test_all1.2 <- test_all1.2[,-15] # Drop company response to consumer b/c it is perfectly correlated with relief

# Submitted.via
table(test_all1.2$Submitted.via, test_all1.2$missing_zip)
test_all1.3 <- test_all1.2 %>%
  mutate(Submitted.via = case_when(Submitted.via == "Web Referral" ~ "Referral",
                                   Submitted.via == "Web" ~ "Web",
                                   Submitted.via == "Referral" ~ "Referral",
                                   Submitted.via == "Postal mail" ~ "Postal mail",
                                   Submitted.via == "Phone" ~ "Phone",
                                   Submitted.via == "Fax" ~ "Fax")) # Recode Web Referral to Referral

# Use mode to impute zip code
test_all1.4 <- test_all1.3 %>%
  group_by(State) %>%
  mutate(imp_zip = ifelse(missing_zip == TRUE, Mode(ZIP.code),  ZIP.code))

test_all1.5 <- test_all1.4[,-c(20:21)] # remove COUNTYNAME & STCOUNTYFP
test_all1.6 <- merge(test_all1.5, zip_fips_states, by.x = c("imp_zip", "State"), by.y = c("ZIP", "STATE")) # merge

# Add a leading zero to the test_all1.6 FIPS code
test_all1.6$fips <- ifelse(nchar(test_all1.6$STCOUNTYFP) == 4, add_char(x = test_all1.6$STCOUNTYFP,
                                                                        pos = 0,
                                                                        insert = "0"), test_all1.6$STCOUNTYFP)

# Merge county debt data with test_all1.6
test_all1.7 <- merge(test_all1.6, county_debt, by = "fips", all.x = TRUE)

# Merge debt data with Census data
test_all1.8 <- merge(test_all1.7, census4, by.x = "fips", by.y = "STCOUNTY")

# Create a flag for servicemembers and older Americans
test_all1.8$service_mem <- ifelse(grepl("Servicemember", test_all1.8$Tags), 1, 0)
test_all1.8$old <- ifelse(grepl("Older American", test_all1.8$Tags), 1, 0)
test_all1.8$old_service_mem <- ifelse(grepl("Older American, Servicemember", test_all1.8$Tags), 1, 0)
test_all1.8$none <- ifelse(grepl("None", test_all1.8$Tags), 1, 0)

# Create a subset of debt collection
debt_subset <- county_debt[, -c(2,3)]

# Abbreviate the names of the variables
colnames(debt_subset) <- abbreviate(colnames(debt_subset), minlength = 4)

# Make all variables numeric
debt_subset2 <- debt_subset %>%
  sapply(as.numeric) %>%
  as.data.frame()
debt_subset2$is_complete <- as.factor(complete.cases(debt_subset2)) # variable for complete cases

set.seed(19745) # set random seed
debt_subset3 <- rfImpute(fips~., debt_subset2[,-27]) # impute using RF

# Get correlation by removing fips code
debt_matrix_rf2 <- cor(debt_subset3[,-1])
ggcorrplot(debt_matrix_rf2)

# Subset variables that will be included in PCA (based on the correlation matrix)
pca_vars2 <- debt_subset3 %>%
  select(c(SwadicWc, SwadicA, SwmdicA, SwmdicCoc, SwmdicWc, AhiA, AhiWc, AhiCoc))

# Perform PCA on selected variables and save the scores/loadings
debt_pca_rf2 <- prcomp(pca_vars2, scale = TRUE)
summary(debt_pca_rf2)
fviz_eig(debt_pca_rf2, addlabels = TRUE)
loadings2 <- debt_pca_rf2$rotation
scores2 <- debt_pca_rf2$x

# Remove variables used in PCA from the dataset and add the principal components
county_debt1.2 <- debt_subset3 %>%
  select(-c(SwadicWc, SwadicA, SwmdicA, SwmdicCoc, SwmdicWc, AhiA, AhiWc, AhiCoc))
county_debt1.2 <- cbind(county_debt1.2, scores2[, 1:2])

# Add leading zero to fips before merging back onto rest of data
county_debt1.2$fips2<- ifelse(nchar(county_debt1.2$fips) == 4, add_char(x = county_debt1.2$fips,
                                                                        pos = 0,
                                                                        insert = "0"), county_debt1.2$fips)

# Merge back with the rest of the data
test_all1.9 <- test_all1.8[,-c(25:51)]
test_all1.9 <- merge(test_all1.9, county_debt1.2[,-1], by.x = "fips", by.y = "fips2", all.x = TRUE)

# Make binary variable out of Sub product == Medical Debt
test_all1.9$medical_debt <- ifelse((test_all1.9$Sub.product == "Medical debt" | test_all1.9$Sub.product == "Medical"), 1, 0)

z <- test_all1.9[, c("medical_debt", "old_service_mem",
                     "pct_old", "pct_young",
                     "SoslhwsldidA", "Consumer.disputed.")]

z[,1]<- as.factor(z[,1])
z[,2]<- as.factor(z[,2])
z[,3]<- scale(as.numeric(z[,3]))
z[,4] <- scale(as.numeric(z[,4]))
z[,5]<- scale(as.numeric(z[,5]))
z[,6] <- as.factor(z[,6])

set.seed(19745)

# 4
kpres4.2 <-kproto(x = z,k=4)
kpres4.2

# Scree plot
n.scree <- ncol(z)
Es <- numeric(n.scree)
for(i in 1:n.scree){
  kpres <- kproto(z, k = i, nstart = 5, verbose = FALSE)
  Es[i] <- kpres$tot.withinss
}

plot(1:n.scree, Es[1:6], type = "b", ylab = "Objective Function",
     xlab = "# Clusters", main = "ScreePlot")

# Add with four clusters (based on scree plot)
test_all1.9$cluster4 <- kpres4.2$cluster

# Turn some variables into factors
test_all1.9$Sub.product <- as.factor(test_all1.9$Sub.product)
test_all1.9$Issue <- as.factor(test_all1.9$Issue)
test_all1.9$Company.public.response <- as.factor(test_all1.9$Company.public.response)
test_all1.9$Submitted.via <- as.factor(test_all1.9$Submitted.via)
test_all1.9$Timely.response. <- as.factor(test_all1.9$Timely.response.)
test_all1.9$cluster4 <- as.factor(test_all1.9$cluster4)

# Drop medical debt, old_service_mem, SoslhwsldidA, consumer disputed, pct_old out of the data because they are represented in the cluster. Drop state, zip code (keep the imputed one), county, product (all debt collection), customer complaint narrative (unique), complaint ID (unique), date sent (because it is nearly identical to date received), company (over 2000 levels)
test_all2.0 <- test_all1.9 %>%
  select(-c(State, zip2, ZIP.code, Sub.issue, Company.public.response, Consumer.disputed., pct_young, SoslhwsldidA, old_service_mem, medical_debt, Consumer.complaint.narrative, Complaint.ID, STCOUNTYFP, Product, COUNTYNAME, AGEGRP, Company, Date.sent.to.company, Tags))

# Extract year from date received
test_all2.0$year <- format(as.Date(test_all2.0$Date.received, format="%m/%d/%Y"),"%y")
test_all2.0$year <- paste0("20", test_all2.0$year)

# Drop most of the Census variables, date received (because we have the year now)
test_all2.1 <- test_all2.0[, -c(1:3, 11:87)]

# Recode levels of subproduct to collapse alike levels
test_all2.1 <- test_all2.1 %>%
  mutate(Sub.product2 = case_when(Sub.product == "Auto" ~ "Auto debt",
                                  Sub.product == "Credit card" ~ "Credit card debt",
                                  Sub.product == "Federal student loan" ~ "Student loan debt",
                                  Sub.product == "Non-federal student loan" ~ "Student loan debt",
                                  Sub.product == "Federal student loan debt" ~ "Student loan debt",
                                  Sub.product == "Private student loan debt" ~ "Student loan debt",
                                  Sub.product == "Mortgage" ~ "Mortgage debt",
                                  Sub.product == "Other (i.e. phone, health club, etc.)" ~ "Other debt",
                                  Sub.product == "Payday loan" ~ "Payday loan debt",
                                  Sub.product == "Medical" ~ "Medical debt",
                                  .default = Sub.product))

# Remove the original Sub.product variable
test_all2.1 <- test_all2.1 %>%
  select(-c(Sub.product))

# Choosing variables; removing race variables because we have Sopoc (share of people of color by county), removing White community specific variables because we have the share of white people through the share of people of color by county, and removing community of color variables because they were ~80% missing and we imputed that data, so it's likely not accurate even if it comes back as a significant predictor in our models
test_all2.2 <- test_all2.1 %>%
  select(-c(pct_hisp, pct_asian, pct_blk, pct_native, MdicWc, CcddrWc, AldrWc, MccddWc, SoslhwsldidWc, SwadicCoc, MdicCoc, AldrCoc, CcddrCoc, MccddCoc, SoslhwsldidCoc)) %>%
  mutate(relief = as.factor(relief))

### RANDOM FOREST
set.seed(182654) # set random seed
rf.predict2 <- predict(relief.rf, na.omit(test_all2.2), method = "class")
table(Predicted = rf.predict2, Actual = na.omit(test_all2.2)$relief)
rf.accuracy2 <- (31759 + 1498)/(31759 + 1498 + 6168 + 4475) # 0.7575626
rf.F1_2 <- F1_Score(rf.predict2, na.omit(test_all2.2)$relief) # 0.8564879


# 11. Compare and contrast the models you decide to run. Choose the model you think is optimal and explain why?

## a. Do they meet the assumptions of the model?

## b. Can they be interpreted?

## c. Which has the best fit?

# 12. After picking the best model, drop one variable. The dummy variable for Older American you made from the original data set tags and rerun. Does the aggregated older American percentages from the Census data provide the same level of details and predictive accuracy as the micro data provided by the original dataset? Do you think these results are representative of other demographic characteristics.
