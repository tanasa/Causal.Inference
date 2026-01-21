cat("PROPENSITY SCORE")

cat("

The propensity score is the probability that a unit (person, customer, patient) receives treatment, given their observed characteristics.

Mathematical Definition

e(X) = P(W = 1 | X)

Where:

e(X) = propensity score
W = treatment indicator (1 = treated, 0 = control)
X = observed covariates/features

Why Do We Need It?

The Problem: Selection Bias

In observational studies (non-randomized), treatment assignment isn't random. 
People who get treated are often systematically different from those who don't.

Example: Studying the effect of a new drug on recovery time

Sicker patients → more likely to get the drug
If we just compare treated vs. untreated, we're comparing apples to oranges

The Solution: Propensity Scores
The propensity score helps us:

Balance treatment groups - make them comparable
Control for confounding - adjust for systematic differences
Mimic randomization - create pseudo-random assignment

")

cat("

Key Theorem: The Balancing Property : Rosenbaum & Rubin (1983)

Translation: 

Within groups with the same propensity score, treatment is independent of covariates. 

The treated and control groups are balanced.

")

cat(" Propensity Score = Probability of receiving treatment given observed covariates
    
W = Treatment (1 = treated, 0 = control)
X = Covariates (age, BMI, smoking, etc.)

Why use it  ? Confounding !
")

cat("

Real-World Example: Exercise → Blood Pressure

Person Age BMI Smoker Exercise (W) BP (Y)
A  65  30  Yes 1  130 
B  30  22  No  0  118
")



cat(" Simple Example : Scenario: Job Training Program ")

# Simulate data
set.seed(123)
n <- 1000

data <- data.frame(
  age = rnorm(n, 35, 10),
  education = rnorm(n, 14, 2),
  prior_wage = rnorm(n, 30000, 10000)
)

# Treatment assignment is NOT random

# Younger, more educated people are more likely to get training
data$prob_treatment <- plogis(-2 + 0.05*data$age + 0.3*data$education + 0.00001*data$prior_wage)
data$treated <- rbinom(n, 1, data$prob_treatment)

# Outcome: post-training wage
true_effect <- 5000  # True causal effect
data$post_wage <- 20000 + 
  500*data$age + 
  2000*data$education + 
  0.5*data$prior_wage +
  data$treated * true_effect +
  rnorm(n, 0, 3000)

head(data, 5)
tail(data, 5)

# Check imbalance
library(dplyr)
data %>%
  group_by(treated) %>%
  summarise(
    n = n(),
    avg_age = mean(age),
    avg_education = mean(education),
    avg_prior_wage = mean(prior_wage)
  )

# Result: Treated group is younger and more educated → biased comparison!

cat("Estimating Propensity Scores")

cat('

You have observational data, not a randomized experiment.

That means:

Some people were more likely to be treated (e.g., training program, exercise, coupon, drug)

Others more likely to be controls

This creates confounding.

To fix that, we need to estimate each person’s probability of receiving treatment, given their observed characteristics.

That probability is called the PROPENSITY SCORE.

🧠 Why do we fit a logistic regression?

Because:

✔️ Propensity score = P(treated=1∣X)
✔️ Logistic regression directly models probability of treatment.

Your model:

ps_model <- glm(treated ~ age + education + prior_wage,
                data = data,
                family = binomial(link = "logit"))

Is estimating:

P(treated=1∣age, education, prior_wage)

i.e.:

“Given your age, education, and prior wage, what is your probability of being put in the treated group?”

This is exactly the definition of the PROPENSITY SCORE.

🎯 Why does logistic regression work?

Because:

Logistic regression outputs values between 0 and 1

It gives a probability of treatment

It uses the covariates that may cause selection bias

📌 Why do we use the predicted probabilities as propensity scores?

This line:

data$propensity_score <- predict(ps_model, type = "response")

The type = "response" part tells R:

“Give me the predicted probability, not the log-odds.”

| Person | Age | Education   | Prior wage | Propensity score |
| ------ | --- | ----------- | ---------- | ---------------- |
| 1      | 22  | High School | 18k        | 0.18             |
| 2      | 45  | B.S.        | 55k        | 0.72             |
| 3      | 30  | M.S.        | 90k        | 0.85             |

Meaning:

Person 1 has 18% chance of being treated

Person 2 has 72% chance

Person 3 has 85% chance

')

# Fit logistic regression
ps_model <- glm(treated ~ age + education + prior_wage,
                data = data,
                family = binomial(link = "logit"))
ps_model

# Get propensity scores
data$propensity_score <- predict(ps_model, type = "response")

head(data, 5)
tail(data, 5)

# Visualize
library(ggplot2)
ggplot(data, aes(x = propensity_score, fill = factor(treated))) +
  geom_histogram(alpha = 0.5, position = "identity", bins = 30) +
  labs(
    title = "Propensity Score Distribution",
    x = "Propensity Score (Probability of Treatment)",
    y = "Count",
    fill = "Treated"
  ) +
  scale_fill_manual(values = c("0" = "blue", "1" = "red"),
                    labels = c("No", "Yes")) +
  theme_minimal()

cat("

The plots shows :

Red = treated group

Blue = control group

What you look for:

Do treated and control overlap?

If yes → matching is possible

If no → extreme confounding (bad)

Which covariates drive treatment assignment?

If matching or weighting will work well

Good overlap = good causal identification

Poor overlap = unstable estimates

")

cat("

📝 Summary in one sentence

We fit a logistic regression to estimate each person’s probability of receiving treatment (the propensity score), 
and we visualize these scores to understand how much treated and control groups overlap — 
and whether matching / weighting will produce valid causal estimates.
    
")

cat("Methods Using Propensity Scores")

cat("

1. Matching

Idea: Match each treated unit with a control unit that has a similar propensity score.

")

# ?matchit

# matchit() is the main function of MatchIt and performs pairing, subset selection, and subclassification 
# with the aim of creating treatment and control groups balanced on included covariates. 

# MatchIt implements the suggestions of Ho, Imai, King, and Stuart (2007) for improving parametric statistical models 
# by preprocessing data with nonparametric matching methods.

# matchit(
#  formula,
#  data = NULL,
#  method = "nearest",
#  distance = "glm",
#  link = "logit"



library(MatchIt)

# Perform propensity score matching
matched <- matchit(treated ~ age + education + prior_wage,
                   data = data,
                   method = "nearest",  # nearest neighbor matching
                   ratio = 1,           # 1:1 matching
                   caliper = 0.1)       # maximum distance

matched

# Get matched data
matched_data <- match.data(matched)
head(matched_data, 5)

# Check balance after matching
matched_data %>%
  group_by(treated) %>%
  summarise(
    n = n(),
    avg_age = mean(age),
    avg_education = mean(education),
    avg_prior_wage = mean(prior_wage)
  )

head(matched_data[c("treated", "distance", "weights", "subclass")])

cat('

distance = the propensity score (P(treated=1|X))

weights = matching weights

subclass = matched pair/stratum id

')

# Extract PS directly

ps <- matched$distance
head(ps)

logit_ps <- predict(matched$model, type = "link")
head(logit_ps)

summary(matched)                 # balance summary
plot(matched, type = "jitter")   # PS overlap before/after
plot(matched, type = "hist")     # PS histograms




# Estimate treatment effect on matched data

ate_matched <- lm(post_wage ~ treated, data = matched_data)
summary(ate_matched)

cat("\nEstimated Treatment Effect (Matching):", 
    round(coef(ate_matched)["treated"], 2), "\n")
cat("True Effect:", true_effect, "\n")

# “After we’ve balanced treated and control groups by matching on similar covariates, what is the average difference in outcome?”
# That difference ≈ the causal treatment effect, assuming:
# Covariate balance achieved (no confounding left),

cat("NOTE")

cat('

matched <- matchit(treated ~ age + education + prior_wage,
                   data = data,
                   method = "nearest",
                   distance = "mahalanobis")

Here:

You are not using the propensity score.

Instead, treated and control units are matched using Mahalanobis distance on the covariates (age, education, prior_wage).

This means each treated unit is paired with the control unit that has the smallest multivariate distance in those covariates.

')

cat('

The Mahalanobis distance measures the distance between two vectors, 

taking into account the correlations among the variables.

Alternative

distance = "logit" → match on propensity scores (default).

')



cat(" 

2. Inverse Probability Weighting (IPW)

Idea: Weight observations by the inverse of their probability of receiving the treatment they actually received.

")

cat("

Idea behind IPTW

Inverse Probability of Treatment Weighting creates a pseudo-population in which treatment assignment is independent of the covariates.

Treated units with low propensity (unlikely to be treated) get large weight

Controls with high propensity (unlikely to be untreated) get large weight

This makes the reweighted sample look as if treatment was randomized.

")

# Compute IPW weights

data$ipw_weight <- ifelse(data$treated == 1,
                          1 / data$propensity_score,          # For treated
                          1 / (1 - data$propensity_score))    # For control

# Trim extreme weights (optional but recommended)
data$ipw_weight_trimmed <- pmin(data$ipw_weight, quantile(data$ipw_weight, 0.99))

# Estimate ATE using IPW

library(survey)

design_ipw <- svydesign(ids = ~1, weights = ~ipw_weight_trimmed, data = data)

ate_ipw <- svyglm(post_wage ~ treated, design = design_ipw)

cat("\nEstimated Treatment Effect (IPW):", 
    round(coef(ate_ipw)["treated"], 2), "\n")

cat("True Effect:", true_effect, "\n")

# How IPW works:

# Treated units with low propensity (unlikely to be treated) get high weight → upweight rare cases
# Control units with high propensity (likely to be treated but weren't) get high weight → upweight rare cases
# This creates a "pseudo-population" where treatment is independent of X



cat(" Stratification (Subclassification) ")

cat(" Divide data into strata based on propensity score, estimate effect within each stratum, then average. ")

# data$propensity_score

# 1) Fit the propensity score model (logistic regression)
ps_model <- glm(treated ~ age + education + prior_wage,
                data = data, family = binomial())

# 2) Get the propensity scores p = P(treated = 1 | X)
data$propensity_score <- predict(ps_model, type = "response")  # numeric in (0,1)

# (optional but recommended) clip extremes to avoid 0/1
eps <- 1e-4
data$propensity_score <- pmin(pmax(data$propensity_score, eps), 1 - eps)

# Quick sanity checks
summary(data$propensity_score)
stopifnot(is.numeric(data$propensity_score), all(is.finite(data$propensity_score)))

# Create propensity score quintiles
data$ps_stratum <- cut(data$propensity_score,
                       breaks = quantile(data$propensity_score, probs = seq(0, 1, 0.2)),
                       include.lowest = TRUE,
                       labels = 1:5)

# Estimate effect within each stratum
stratum_effects <- data %>%
  group_by(ps_stratum) %>%
  summarise(
    n = n(),
    effect = mean(post_wage[treated == 1]) - mean(post_wage[treated == 0]),
    .groups = 'drop'
  )

print(stratum_effects)

# Overall ATE (weighted average)
ate_stratified <- weighted.mean(stratum_effects$effect, 
                                stratum_effects$n)

cat("\nEstimated Treatment Effect (Stratification):", 
    round(ate_stratified, 2), "\n")
cat("True Effect:", true_effect, "\n")

cat(" Assessing Balance ")

cat("after using propensity scores, we need to check if balance improved")

# Function to compute standardized mean difference
compute_smd <- function(data, var, treat_var = "treated") {
  treated_mean <- mean(data[[var]][data[[treat_var]] == 1])
  control_mean <- mean(data[[var]][data[[treat_var]] == 0])
  pooled_sd <- sqrt((var(data[[var]][data[[treat_var]] == 1]) + 
                     var(data[[var]][data[[treat_var]] == 0])) / 2)
  smd <- (treated_mean - control_mean) / pooled_sd
  return(smd)
}

# Before matching
smd_before <- data.frame(
  variable = c("age", "education", "prior_wage"),
  smd = c(
    compute_smd(data, "age"),
    compute_smd(data, "education"),
    compute_smd(data, "prior_wage")
  )
)

# After matching
smd_after <- data.frame(
  variable = c("age", "education", "prior_wage"),
  smd = c(
    compute_smd(matched_data, "age"),
    compute_smd(matched_data, "education"),
    compute_smd(matched_data, "prior_wage")
  )
)

# Plot balance
library(ggplot2)
balance_data <- rbind(
  data.frame(smd_before, timing = "Before"),
  data.frame(smd_after, timing = "After")
)

ggplot(balance_data, aes(x = variable, y = abs(smd), fill = timing)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_hline(yintercept = 0.1, linetype = "dashed", color = "red") +
  labs(
    title = "Covariate Balance: Before vs After Matching",
    subtitle = "Standardized Mean Difference (SMD < 0.1 indicates good balance)",
    x = "Variable",
    y = "Absolute SMD",
    fill = ""
  ) +
  theme_minimal() +
  coord_flip()



cat("Another example : Grok")


# ===============================================
# PROPENSITY SCORE: Exercise → Blood Pressure
# ===============================================

library(dplyr)
library(MatchIt)
library(ggplot2)

# --- 1. SIMULATE DATA ---
set.seed(123)
n <- 1000

data <- data.frame(
  age = runif(n, 20, 70),
  bmi = rnorm(n, 27, 5),
  smoking = rbinom(n, 1, 0.3),
  W = 0  # Will be assigned
)

# People more likely to exercise: younger, lower BMI, non-smokers
logit_p <- -2 + 0.05*data$age - 0.1*data$bmi - 1*data$smoking
data$propensity <- plogis(logit_p)
data$W <- rbinom(n, 1, data$propensity)

# Outcome: exercise lowers BP, but confounders
data$Y <- 120 + 0.5*data$age + 1*data$bmi + 5*data$smoking - 8*data$W + rnorm(n)

cat("First 5 rows:\n")
print(head(data))

# --- 2. ESTIMATE PROPENSITY SCORE ---
ps_model <- glm(W ~ age + bmi + smoking, data = data, family = binomial)
data$ps <- predict(ps_model, type = "response")
print(head(data))

# --- 3. CHECK BALANCE BEFORE/AFTER ---
balance_plot <- function(data, var) {
  ggplot(data, aes(x = !!sym(var), fill = factor(W))) +
    geom_density(alpha = 0.5) +
    labs(title = paste("Balance on", var), fill = "Exercise")
}

print(balance_plot(data, "age"))
print(balance_plot(data, "bmi"))

# --- 4. PROPENSITY SCORE MATCHING ---
match_out <- matchit(W ~ age + bmi + smoking, data = data, method = "nearest", ratio = 1)
matched_data <- match.data(match_out)

# --- 5. CHECK BALANCE AFTER MATCHING ---
print(balance_plot(matched_data, "age"))

# --- 6. ESTIMATE TREATMENT EFFECT ---
ate_matched <- lm(Y ~ W, data = matched_data)$coefficients["W"]
cat("\nATE (Matched) =", round(ate_matched, 2), "mmHg\n")

# --- 7. IPTW (Inverse Probability of Treatment Weighting) ---
data$weight <- ifelse(data$W == 1, 1/data$ps, 1/(1-data$ps))
ate_iptw <- weighted.mean(data$Y[data$W==1], data$weight[data$W==1]) - 
            weighted.mean(data$Y[data$W==0], data$weight[data$W==0])
cat("ATE (IPTW) =", round(ate_iptw, 2), "mmHg\n")





