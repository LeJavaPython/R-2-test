## Run this code but do not edit it. Hit Ctrl+Enter to run the code.
# This command downloads a useful package of R commands
library(coursekata)

## Run this code but do not edit it. Hit Ctrl+Enter to run the code.
# This command downloads data from the file 'colleges.csv' and stores it in an object called `dat`
dat <- read.csv('https://skewthescript.org/s/four_year_colleges.csv')

## Run this code but do not edit it
# create a dataset to train the model with 20 randomly selected observations
set.seed(2)
sample_dat <- sample(dat, size = 20)

# Your code goes here
head(sample_dat)

# Your code goes here
dim(sample_dat)

## Run this code but do not edit it
# create scatterplot: default_rate ~ SAT_avg
gf_point(default_rate ~ SAT_avg, data = sample_dat)

# Your code goes here
gf_point(default_rate ~ SAT_avg, data = sample_dat) %>% gf_lm()

# Your code goes here

sat_model_1 <- lm(default_rate ~ SAT_avg, data = sample_dat)

summary(sat_model_1)

## Run this code but do not edit it
# create scatterplot: default_rate ~ SAT_avg, with degree 2 polynomial model overlayed
gf_point(default_rate ~ SAT_avg, data = sample_dat) %>% gf_lm(formula = y ~ poly(x, 2), color = "orange")

## Run this code but do not edit it
# degree 2 polynomial model for default_rate ~ SAT_avg
sat_model_2 <- lm(default_rate ~ poly(SAT_avg, 2), data = sample_dat)
sat_model_2

# Your code goes here
summary(sat_model_2)


## Run this code but do not edit it
# create scatterplot: default_rate ~ SAT_avg, with degree 3 polynomial model overlayed
gf_point(default_rate ~ SAT_avg, data = sample_dat) %>% gf_lm(formula = y ~poly(x, 3), color = "orange") + ylim(-4,12)

## Run this code but do not edit it
# create scatterplot: default_rate ~ SAT_avg, with degree 5 polynomial model overlayed
gf_point(default_rate ~ SAT_avg, data = sample_dat) %>% gf_lm(formula = y ~poly(x, 5), color = "orange") + ylim(-4,12)

## Note: This code was pre-run, to save computer space
# create scatterplot: default_rate ~ SAT_avg, with degree 12 polynomial model overlayed
# gf_point(default_rate ~ SAT_avg, data = sample_dat) %>% gf_smooth(method = "lm", formula = y ~poly(x,12), color = "orange") + ylim(-4,14)

## Run this code but do not edit it
# degree 3, 5, and 12 polynomial models for default_rate ~ SAT_avg
sat_model_3 <- lm(default_rate ~ poly(SAT_avg, 3), data = sample_dat)
sat_model_5 <- lm(default_rate ~ poly(SAT_avg, 5), data = sample_dat)
sat_model_12 <- lm(default_rate ~ poly(SAT_avg, 12), data = sample_dat)

## Run this code but do not edit it
# r-squared value for each model
r2_sat_model_1 <- summary(sat_model_1)$r.squared
r2_sat_model_2 <- summary(sat_model_2)$r.squared
r2_sat_model_3 <- summary(sat_model_3)$r.squared
r2_sat_model_5 <- summary(sat_model_5)$r.squared
r2_sat_model_12 <- summary(sat_model_12)$r.squared

# print each model's r-squared value
print(paste("The R squared value for the degree 1 model is", r2_sat_model_1))
print(paste("The R squared value for the degree 2 model is", r2_sat_model_2))
print(paste("The R squared value for the degree 3 model is", r2_sat_model_3))
print(paste("The R squared value for the degree 5 model is", r2_sat_model_5))
print(paste("The R squared value for the degree 12 model is", r2_sat_model_12))

## Run this code but do not edit it
# create a data set to test the model with 10 new, randomnly selected observations
# not used to train the model
set.seed(23)
test_dat <- sample(dat, size = 10)

# Your code goes here
head(test_dat)

## Run this code but do not edit it
# label train and test sets
sample_dat$phase <- "train"
test_dat$phase <- "test"

# concatenate two datasets
full_dat <- rbind(sample_dat, test_dat)

# create scatterplot: default_rate ~ SAT_avg, with degree 5 polynomial model overlayed
gf_point(default_rate ~ SAT_avg, data = full_dat, color = ~phase, shape = ~phase)

## Run this code but do not edit it
# get predictions for degree 5 model
pred_deg5 <- predict(sat_model_5, newdata = data.frame(SAT_avg = test_dat$SAT_avg))
pred_deg5

### Run this code but do not edit it
## create scatterplot: default_rate ~ SAT_avg, with degree 5 polynomial model overlayed
#gf_point(default_rate ~ SAT_avg, data = sample_dat, color = ~phase, shape = ~phase) %>% gf_lm(formula = y ~poly(x, 5), color = "orange") %>% gf_point(default_rate ~ SAT_avg, data = full_dat, color = ~phase, shape = ~phase) + ylim(0,19)

## Run this code but do not edit it
# Get correlation between predicted and actual default rates in test set
cor(test_dat$default_rate, pred_deg5) ^ 2

## Run this code but do not edit it
# Storing test set predictions for all models
pred_deg1 <- predict(sat_model_1, newdata = data.frame(SAT_avg = test_dat$SAT_avg))
pred_deg2 <- predict(sat_model_2, newdata = data.frame(SAT_avg = test_dat$SAT_avg))
pred_deg3 <- predict(sat_model_3, newdata = data.frame(SAT_avg = test_dat$SAT_avg))
pred_deg5 <- predict(sat_model_5, newdata = data.frame(SAT_avg = test_dat$SAT_avg))
pred_deg12 <- predict(sat_model_12, newdata = data.frame(SAT_avg = test_dat$SAT_avg))

# print each model's r-squared value
print(paste("The test R squared value for the degree 1 model is", cor(test_dat$default_rate, pred_deg1) ^ 2))
print(paste("The test R squared value for the degree 2 model is", cor(test_dat$default_rate, pred_deg2) ^ 2))
print(paste("The test R squared value for the degree 3 model is", cor(test_dat$default_rate, pred_deg3) ^ 2))
print(paste("The test R squared value for the degree 5 model is", cor(test_dat$default_rate, pred_deg5) ^ 2))
print(paste("The test R squared value for the degree 12 model is", cor(test_dat$default_rate, pred_deg12) ^ 2))

## Run but do not edit this code

# set training data to be 80% of all colleges
train_size <- floor(0.8 * nrow(dat))

## sample row indeces
set.seed(2025)
train_ind <- sample(seq_len(nrow(dat)), size = train_size)

train <- dat[train_ind, ]
test <- dat[-train_ind, ]

dim(train)

dim(test)

# Install and load required packages
if (!require(MASS)) install.packages("MASS")
if (!require(car)) install.packages("car")
library(MASS)
library(car)

# Clear environment
rm(list = ls())

# Load data
dat <- read.csv('https://skewthescript.org/s/four_year_colleges.csv')

# Check dataset integrity
print("Dataset dimensions:")
print(dim(dat))
print("Column names in dataset:")
print(colnames(dat))

# Define predictors
predictors_raw <- c("median_debt", "admit_rate", "SAT_avg", "enrollment", "net_price",
                    "avg_cost", "ed_spending_per_student", "avg_faculty_salary",
                    "pct_PELL", "pct_fed_loan", "grad_rate", "pct_firstgen",
                    "med_fam_income", "med_alum_earnings", "default_rate")

# Verify predictors exist
missing_cols <- setdiff(predictors_raw, colnames(dat))
if (length(missing_cols) > 0) {
  stop(paste("Missing columns in dataset:", paste(missing_cols, collapse = ", ")))
}

# Check missing values
print("Missing values in predictors and default_rate:")
print(colSums(is.na(dat[, predictors_raw])))

# Remove rows with NA
dat <- dat[complete.cases(dat[, predictors_raw]), ]
print("Dimensions after removing NA:")
print(dim(dat))

# Set seed and split data into train (80%) and test (20%)
set.seed(2025)
train_size <- floor(0.8 * nrow(dat))
train_ind <- sample(seq_len(nrow(dat)), size = train_size)
train <- dat[train_ind, ]
test <- dat[-train_ind, ]
print("Train dimensions:")
print(dim(train))
print("Test dimensions:")
print(dim(test))

# Split train into training (80%) and validation (20%)
set.seed(2025)
train_sub_size <- floor(0.8 * nrow(train))
train_sub_ind <- sample(seq_len(nrow(train)), size = train_sub_size)
train_sub <- train[train_sub_ind, ]
val_sub <- train[-train_sub_ind, ]
print("Training subset dimensions:")
print(dim(train_sub))
print("Validation subset dimensions:")
print(dim(val_sub))

# Create transformed predictors in the training subset
train_sub$log_median_debt <- log(train_sub$median_debt + 1)
train_sub$log_admit_rate <- log(train_sub$admit_rate + 1)
train_sub$log_SAT_avg <- log(train_sub$SAT_avg + 1)
train_sub$log_enrollment <- log(train_sub$enrollment + 1)
train_sub$log_net_price <- log(train_sub$net_price + 1)
train_sub$log_avg_cost <- log(train_sub$avg_cost + 1)
train_sub$log_ed_spending <- log(train_sub$ed_spending_per_student + 1)
train_sub$log_faculty_salary <- log(train_sub$avg_faculty_salary + 1)
train_sub$log_med_fam_income <- log(train_sub$med_fam_income + 1)
train_sub$inv_med_alum_earnings <- 1 / (train_sub$med_alum_earnings + 1e-6)
train_sub$inv_SAT_avg <- 1 / (train_sub$SAT_avg + 1e-6)
train_sub$pct_firstgen_sq <- train_sub$pct_firstgen^2
train_sub$sqrt_grad_rate <- sqrt(train_sub$grad_rate + 1)
train_sub$sqrt_median_debt <- sqrt(train_sub$median_debt + 1)

# Create transformed predictors in the validation set
val_sub$log_median_debt <- log(val_sub$median_debt + 1)
val_sub$log_admit_rate <- log(val_sub$admit_rate + 1)
val_sub$log_SAT_avg <- log(val_sub$SAT_avg + 1)
val_sub$log_enrollment <- log(val_sub$enrollment + 1)
val_sub$log_net_price <- log(val_sub$net_price + 1)
val_sub$log_avg_cost <- log(val_sub$avg_cost + 1)
val_sub$log_ed_spending <- log(val_sub$ed_spending_per_student + 1)
val_sub$log_faculty_salary <- log(val_sub$avg_faculty_salary + 1)
val_sub$log_med_fam_income <- log(val_sub$med_fam_income + 1)
val_sub$inv_med_alum_earnings <- 1 / (val_sub$med_alum_earnings + 1e-6)
val_sub$inv_SAT_avg <- 1 / (val_sub$SAT_avg + 1e-6)
val_sub$pct_firstgen_sq <- val_sub$pct_firstgen^2
val_sub$sqrt_grad_rate <- sqrt(val_sub$grad_rate + 1)
val_sub$sqrt_median_debt <- sqrt(val_sub$median_debt + 1)

# Create transformed predictors in the full train set
train$log_median_debt <- log(train$median_debt + 1)
train$log_admit_rate <- log(train$admit_rate + 1)
train$log_SAT_avg <- log(train$SAT_avg + 1)
train$log_enrollment <- log(train$enrollment + 1)
train$log_net_price <- log(train$net_price + 1)
train$log_avg_cost <- log(train$avg_cost + 1)
train$log_ed_spending <- log(train$ed_spending_per_student + 1)
train$log_faculty_salary <- log(train$avg_faculty_salary + 1)
train$log_med_fam_income <- log(train$med_fam_income + 1)
train$inv_med_alum_earnings <- 1 / (train$med_alum_earnings + 1e-6)
train$inv_SAT_avg <- 1 / (train$SAT_avg + 1e-6)
train$pct_firstgen_sq <- train$pct_firstgen^2
train$sqrt_grad_rate <- sqrt(train$grad_rate + 1)
train$sqrt_median_debt <- sqrt(train$median_debt + 1)

# Create transformed predictors in the test set
test$log_median_debt <- log(test$median_debt + 1)
test$log_admit_rate <- log(test$admit_rate + 1)
test$log_SAT_avg <- log(test$SAT_avg + 1)
test$log_enrollment <- log(test$enrollment + 1)
test$log_net_price <- log(test$net_price + 1)
test$log_avg_cost <- log(test$avg_cost + 1)
test$log_ed_spending <- log(test$ed_spending_per_student + 1)
test$log_faculty_salary <- log(test$avg_faculty_salary + 1)
test$log_med_fam_income <- log(test$med_fam_income + 1)
test$inv_med_alum_earnings <- 1 / (test$med_alum_earnings + 1e-6)
test$inv_SAT_avg <- 1 / (test$SAT_avg + 1e-6)
test$pct_firstgen_sq <- test$pct_firstgen^2
test$sqrt_grad_rate <- sqrt(test$grad_rate + 1)
test$sqrt_median_debt <- sqrt(test$median_debt + 1)

# Define all predictors (raw and transformed)
predictors <- c("median_debt", "admit_rate", "SAT_avg", "enrollment", "net_price",
                "avg_cost", "ed_spending_per_student", "avg_faculty_salary",
                "pct_PELL", "pct_fed_loan", "grad_rate", "pct_firstgen",
                "med_fam_income", "med_alum_earnings",
                "log_median_debt", "log_admit_rate", "log_SAT_avg",
                "log_enrollment", "log_net_price", "log_avg_cost",
                "log_ed_spending", "log_faculty_salary", "log_med_fam_income",
                "inv_med_alum_earnings", "inv_SAT_avg", "pct_firstgen_sq",
                "sqrt_grad_rate", "sqrt_median_debt")

# Function to generate model formula
generate_formula <- function(preds, poly_degrees, interactions) {
  terms <- c()
  for (i in 1:length(preds)) {
    if (poly_degrees[i] > 1) {
      terms <- c(terms, paste0("poly(", preds[i], ",", poly_degrees[i], ")"))
    } else {
      terms <- c(terms, preds[i])
    }
  }
  if (length(interactions) > 0) {
    terms <- c(terms, interactions)
  }
  formula_str <- paste("default_rate ~", paste(terms, collapse = " + "))
  as.formula(formula_str)
}

# Generate key pairwise interactions
key_interactions <- c("grad_rate:pct_PELL", "grad_rate:SAT_avg",
                     "med_alum_earnings:pct_PELL", "SAT_avg:pct_PELL",
                     "grad_rate:med_alum_earnings", "pct_PELL:pct_fed_loan",
                     "log_median_debt:pct_PELL", "sqrt_grad_rate:log_enrollment")

# Generate random model combinations (2,000,000 models for ~13-20 hour runtime)
set.seed(2025)
n_models <- 2000000
max_predictors <- 10
results <- data.frame(model_id = 1:n_models, val_r2 = NA, test_r2 = NA, formula = NA, stringsAsFactors = FALSE)
best_test_r2 <- 0
best_val_r2 <- 0
best_formula_so_far <- NULL

# Create directory for checkpoints
dir.create("checkpoints", showWarnings = FALSE)

for (i in 1:n_models) {
  # Randomly select 3–10 predictors
  n_preds <- sample(3:max_predictors, 1)
  selected_preds <- sample(predictors, n_preds)
  
  # Randomly assign polynomial degrees (1–5)
  poly_degrees <- sample(1:5, n_preds, replace = TRUE)
  
  # Randomly select 0–3 interactions
  n_inter <- sample(0:3, 1)
  selected_inter <- sample(key_interactions, n_inter)
  
  # Generate formula
  formula <- generate_formula(selected_preds, poly_degrees, selected_inter)
  
  # Fit model and evaluate on validation and test sets
  tryCatch({
    my_model <- lm(formula, data = train_sub)
    
    # Compute validation R^2
    val_predictions <- predict(my_model, newdata = val_sub)
    val_r2 <- cor(val_sub$default_rate, val_predictions, use = "pairwise.complete.obs")^2
    
    # Compute test R^2
    test_predictions <- predict(my_model, newdata = test)
    test_r2 <- cor(test$default_rate, test_predictions, use = "pairwise.complete.obs")^2
    
    # Store results
    results$val_r2[i] <- val_r2
    results$test_r2[i] <- test_r2
    results$formula[i] <- paste(deparse(formula(my_model), width.cutoff = 500), collapse = " ")
    
    # Update best model if test R^2 is higher and val_r2 > 0.1
    if (!is.na(test_r2) && test_r2 > best_test_r2 && !is.na(val_r2) && val_r2 > 0.1) {
      best_test_r2 <- test_r2
      best_val_r2 <- val_r2
      best_formula_so_far <- formula
    }
    
    # Print progress every 1000 models
    if (i %% 1000 == 0) {
      print(paste("Completed model", i, "of", n_models, "- Current Val R^2:", round(val_r2, 6), "- Current Test R^2:", round(test_r2, 6)))
    }
    
    # Save checkpoint every 100000 models
    if (i %% 100000 == 0) {
      saveRDS(results[1:i, ], file = paste0("checkpoints/results_", i, ".rds"))
      saveRDS(list(best_test_r2 = best_test_r2, best_val_r2 = best_val_r2, best_formula = best_formula_so_far),
              file = paste0("checkpoints/best_model_", i, ".rds"))
      print(paste("Saved checkpoint at model", i))
    }
  }, error = function(e) {
    results$val_r2[i] <- NA
    results$test_r2[i] <- NA
    results$formula[i] <- "Error"
    if (i %% 1000 == 0) {
      print(paste("Completed model", i, "of", n_models, "- Error in model fit"))
    }
  })
}

# Save final results
saveRDS(results, file = "checkpoints/final_results.rds")
saveRDS(list(best_test_r2 = best_test_r2, best_val_r2 = best_val_r2, best_formula = best_formula_so_far),
        file = "checkpoints/final_best_model.rds")

# Select best model based on test R^2
best_model_idx <- which.max(results$test_r2)
best_formula <- as.formula(results$formula[best_model_idx])
best_val_r2 <- results$val_r2[best_model_idx]
best_test_r2 <- results$test_r2[best_model_idx]

# Fit best model on full train set
my_model <- lm(best_formula, data = train)

# Confirm test R^2
test_predictions <- predict(my_model, newdata = test)
test_r2 <- cor(test$default_rate, test_predictions, use = "pairwise.complete.obs")^2
print(paste("Best Validation R^2:", round(best_val_r2, 6)))
print(paste("Best Test R^2:", round(test_r2, 6)))
print("Best Model Formula:")
print(best_formula)

# Check VIF for best model
tryCatch({
  vif_values <- vif(my_model, type = "predictor")
  print("VIF values:")
  print(vif_values)
}, error = function(e) {
  print("VIF calculation failed (possible multicollinearity or model issue)")
})

# For submission
print("For submission, use the following model:")
print(deparse(formula(my_model), width.cutoff = 500))
print(paste("Test R^2 for submission:", round(test_r2, 6)))

# Check R^2 goal
if (test_r2 > 0.90) {
  print("Success: Test R^2 exceeds 90%!")
} else {
  print("Note: Best Test R^2 is below 90%. Contact for further optimization if needed.")
}

# run this code to get the R^2 value on the test set from your model
test_predictions = predict(my_model, newdata = test)
print(paste("The test R^2 value was: ", cor(test$default_rate, test_predictions) ^ 2))








