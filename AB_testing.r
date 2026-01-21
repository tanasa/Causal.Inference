cat('

A/B testing (also called split testing) is a statistical method used to compare two or more versions of something 
— such as a website, advertisement, email, or algorithm — 
to determine which one performs better based on a specific metric.

🧩 How It Works

Form a hypothesis:

Example: “Changing the button color from blue to green will increase click-through rate (CTR).”

Create two versions:

A (control): the current version

B (variant): the modified version

Split your audience randomly:

50% of users see version A

50% see version B

Collect data:
Measure how users behave (e.g., how many clicks, purchases, signups).

Analyze results:
Use statistical tests (often a t-test or chi-squared test) to determine whether the difference in outcomes is statistically significant.

📊 Example

Let’s say you send two versions of an email:

Email A: current subject line

Email B: new subject line

After sending to 10,000 users:

A → 300 people click → 3% CTR

B → 420 people click → 4.2% CTR

If the difference (1.2%) is statistically significant, you adopt Email B as the new version.

⚙️ Applications

Marketing: Test ad headlines or images

Product design: Compare user interface layouts

Machine learning: Compare model versions (e.g., recommendation algorithms)

Healthcare: Compare two treatment strategies (called randomized controlled trials in that field)

📈 Key Concepts

Control group: The baseline condition (version A)

Treatment group: The modified condition (version B)

Statistical significance: The probability that the observed difference is not due to random chance (often p < 0.05)

Sample size: Enough users are needed to detect a meaningful difference confidently

')

cat('

📊 How It’s Done

Form a hypothesis: For example, “Changing the CTA button color will increase sign-ups.”

Create two versions: A (original) and B (modified).

Randomly assign users: Each user sees only one version.

Measure outcomes: Track performance metrics like engagement, conversions, or revenue.

Analyze results: Use statistical tests to determine if differences are significant.

🔍 Beyond A/B: Multivariate Testing
If you want to test more than one variable at a time (e.g., button color and headline), multivariate testing is the next step. It’s more complex but can reveal interactions between elements.

')

cat("Simple proportion test")

# Example 1: Simple proportion test
set.seed(123)
n_A <- 5000; n_B <- 5000
conversions_A <- 320  # 6.4%
conversions_B <- 380  # 7.6%

# group: Labels each row as belonging to group A or B.
# converted: Binary outcome (1 = converted, 0 = not).

# Create data
group <- c(rep("A", n_A), rep("B", n_B))
converted <- c(rep(1, conversions_A), rep(0, n_A - conversions_A),
               rep(1, conversions_B), rep(0, n_B - conversions_B))
df <- data.frame(group, converted)
head(df, 2)
tail(df, 2)

# table(df$group, df$converted) creates a 2×2 contingency table.
# prop.test() performs a chi-squared test for proportions.
# It tests the null hypothesis: conversion rates are equal between groups.

table(df$group, df$converted)

# Proportion test
prop_test <- prop.test(table(df$group, df$converted))
print(prop_test)


# rate_A and rate_B are the observed conversion rates.
# Lift is the difference in conversion rates:
# Lift
# = rate𝐵−rate𝐴
# = 7.6% − 6.4 %
# = 1.2 %
# This quantifies the improvement from A to B.

# Effect size: Absolute lift
rate_A <- mean(df$converted[df$group == "A"])
rate_B <- mean(df$converted[df$group == "B"])
cat(sprintf("Conversion A: %.2f%% | B: %.2f%% | Lift: %.2f%%\n", 
            rate_A*100, rate_B*100, (rate_B - rate_A)*100))

print(rate_A) 
print(rate_B)

mean(df$converted[df$group == "A"])

?prop.test

cat("Chi-square test")

# Example 2: Chi-square test on no-show rates
set.seed(456)
n_A <- 1200; n_B <- 1200

# Simulated no-show rates: A=22%, B=18%
no_show_A <- rbinom(1, n_A, 0.22)
no_show_B <- rbinom(1, n_B, 0.18)
show_A <- n_A - no_show_A
show_B <- n_B - no_show_B

# Contingency table
contingency <- matrix(c(show_A, no_show_A, show_B, no_show_B), nrow = 2,
                      dimnames = list(Group = c("A_48h", "B_24h"),
                                      Outcome = c("Showed", "No-Show")))
print(contingency)

# Chi-square test
chi_test <- chisq.test(contingency)
print(chi_test)

# Relative risk reduction
rr_reduction <- (no_show_A/n_A) / (no_show_B/n_B) - 1
cat(sprintf("No-show A: %.1f%% | B: %.1f%% | RRR: %.1f%%\n",
            100*no_show_A/n_A, 100*no_show_B/n_B, 100*rr_reduction))

cat('

Performs a Chi-squared test of independence.

Checks whether two categorical variables (e.g., Group A vs B and Outcome: Show vs No-show) are independent or associated.

Interpretation:

If p-value < 0.05 → The difference between A and B is statistically significant (group and outcome are not independent).

If p-value ≥ 0.05 → No significant difference.

Relative Risk Reduction (RRR)

rr_reduction <- (no_show_A/n_A) / (no_show_B/n_B) - 1
cat(sprintf("No-show A: %.1f%% | B: %.1f%% | RRR: %.1f%%\n",
             100*no_show_A/n_A,      100*no_show_B/n_B, 100*rr_reduction))

What it means

It compares the risk (probability) of a bad outcome between two groups — usually used in medicine or performance metrics.

no_show_A / n_A = risk in group A

no_show_B / n_B = risk in group B

Relative Risk Reduction (RRR) = (Risk_A / Risk_B) - 1

If negative → A has lower risk (better).
If positive → A has higher risk (worse).

')

cat('

Term	Purpose	Output interpretation

chisq.test()	Tests if two categorical variables are independent	p-value < 0.05 → significant difference

Relative Risk Reduction (RRR)	Quantifies how much smaller the “risk” is in A vs B	Negative = A is better (lower risk)

')

cat("Fisher and ChiSq tests")

cat("
Both Fisher’s exact test and the Chi-square (χ²) test are used to assess whether two categorical variables are independent, 
but they differ in how they do it and when each is appropriate.

Fisher’s exact test

Best for 2×2 contingency tables.

Used when any expected cell count < 5 (violating chi-square assumptions).

Often used in genetics, small clinical studies, or rare-event data.

Chi-square test of independence

Can handle larger tables (e.g., 3×4).

Requires sufficiently large sample sizes (expected counts ≥ 5 in each cell).

Common in surveys and large datasets.
")

cat("

Fisher’s exact test will calculate the exact probability of observing this or a more extreme distribution, 
given these marginal totals.

Chi-square test would compute a χ² statistic = Σ((Observed – Expected)² / Expected), 
then use the χ² distribution (df = 1) to approximate the p-value.

")

# ?fisher.test  # Fisher's exact test

# ?chisq.test  # Chi-square test

# Example 3: T-test on continuous outcome (time in minutes)

# Goal: Increase time spent on educational content (minutes).
# A: Static PDF
# B: Interactive video module

set.seed(789)
n_A <- 800; n_B <- 800

# Simulated data: A ~ N(4.2, 2.1), B ~ N(6.8, 2.5)
time_A <- rnorm(n_A, mean = 4.2, sd = 2.1)
time_B <- rnorm(n_B, mean = 6.8, sd = 2.5)
time_A[time_A < 0] <- 0
time_B[time_B < 0] <- 0

df <- data.frame(
  group = rep(c("A_PDF", "B_Video"), c(n_A, n_B)),
  time_minutes = c(time_A, time_B)
)

head(df)
tail(df)

# Visualize
boxplot(time_minutes ~ group, data = df, main = "Time on Education",
        ylab = "Minutes", col = c("lightcoral", "lightgreen"))

# T-test
t_result <- t.test(time_minutes ~ group, data = df, var.equal = FALSE)
print(t_result)

# Cohen's d (effect size)
library(effsize)
cohen_d <- cohen.d(time_minutes ~ group, data = df)
print(cohen_d)

cat("🔹 What Cohen’s d Measures

Cohen’s d quantifies how large the difference between two means is, in units of standard deviations.

It tells you how much two groups differ — not just whether that difference is statistically significant 
(as the t-test does).

")

cat("

So:

If d=0.0: the groups have nearly identical means.

If d=1.0: their means differ by roughly one standard deviation.

🔹 Interpretation Guidelines (Cohen, 1988)

Cohen’s d	Interpretation
0.2	Small effect
0.5	Medium effect
0.8	Large effect

")

cat('

| **Test**        | **Purpose**                                                               |
| --------------- | ------------------------------------------------------------------------- |
| **t-test**      | Determines if the mean difference is statistically significant (p-value). |
| **Cohen’s *d*** | Quantifies the *magnitude* of that difference (effect size).              |

A t-test might show p < 0.05 (significant), but Cohen’s d tells you whether that difference is meaningful in practice.

')



print("CUPED")

cat("

CUPED stands for Controlled-experiment Using Pre-Experiment Data, 
and it’s a technique used to reduce variance and improve statistical power in A/B testing 
by leveraging baseline (pre-treatment) data.

")

cat("

You’re testing whether Group B improves medication adherence compared to Group A. You have:

Pre-experiment data: % of days each user was adherent before the intervention.

Post-experiment data: % of days each user was adherent after the intervention.

True effect: Group B improves adherence by 3 percentage points.

")

# 1. Simulate Users and Baseline Adherence

set.seed(101)
n <- 3000
users <- data.frame(
  user_id = 1:n,
  baseline_adherence = rbeta(n, 2, 5) * 100,
  group = rep(c("A", "B"), each = n/2)
)

head(users, 2)
tail(users, 2)

# 2. Simulate Post-Experiment Adherence

users$post_adherence <- with(users,
  baseline_adherence * 0.8 + 
  rnorm(n, mean = 0, sd = 10) +
  ifelse(group == "B", 3, 0)
)

head(users, 2)
tail(users, 2)

# This compares average post-adherence between groups without using baseline data.
# It may detect the effect, but with more noise.

t_standard <- t.test(post_adherence ~ group, data = users)
print(t_standard)

# Apply CUPED Adjustment

cv <- cov(users$baseline_adherence, users$post_adherence)
theta <- cv / var(users$baseline_adherence)
users$cuped <- users$post_adherence - theta * users$baseline_adherence

head(users, 2)
tail(users, 2)

# CUPED uses baseline adherence to reduce noise in the post-adherence measurement.
# theta is the adjustment factor based on covariance.
# cuped is the adjusted outcome.

t_cuped <- t.test(cuped ~ group, data = users)
print(t_cuped)

# Measure Variance Reduction
# This calculates how much variance was reduced by using CUPED.
# Lower variance = more statistical power.

var_standard <- var(users$post_adherence[users$group == "A"]) + var(users$post_adherence[users$group == "B"])
var_cuped <- var(users$cuped[users$group == "A"]) + var(users$cuped[users$group == "B"])
cat(sprintf("\nVariance reduction: %.1f%%\n", 100 * (1 - var_cuped / var_standard)))

cat('Lower variance = more statistical power.')

cat('

Standard t-test: compares raw post-treatment outcomes.

CUPED t-test: adjusts for baseline behavior to reduce noise.

Benefit: CUPED improves sensitivity and reduces required sample size.

')

?cov

cat("

AB Test with Patient Stratification - Medication Adherence App

Scenario: A hospital tests two versions of a medication reminder app, 
stratifying by age group to see if effects differ. 

")   

library(tidyverse)

# Simulate data with age stratification
set.seed(456)
n <- 800

data <- tibble(
  patient_id = 1:n,
  age_group = sample(c("18-40", "41-65", "65+"), n, replace = TRUE, 
                     prob = c(0.3, 0.4, 0.3)),
  app_version = sample(c("Standard", "Gamified"), n, replace = TRUE),
  adherence_score = case_when(
    app_version == "Standard" & age_group == "18-40" ~ rnorm(n, 72, 15),
    app_version == "Standard" & age_group == "41-65" ~ rnorm(n, 78, 12),
    app_version == "Standard" & age_group == "65+" ~ rnorm(n, 75, 14),
    app_version == "Gamified" & age_group == "18-40" ~ rnorm(n, 82, 13),
    app_version == "Gamified" & age_group == "41-65" ~ rnorm(n, 80, 12),
    app_version == "Gamified" & age_group == "65+" ~ rnorm(n, 71, 15)
  )
) %>%
  mutate(adherence_score = pmin(pmax(adherence_score, 0), 100))

head(data, 2)
tail(data, 2)
dim(data)
unique(data$app_version)
unique(data$age_group)

# Overall analysis
overall_results <- data %>%
  group_by(app_version) %>%
  summarise(
    n = n(),
    mean_adherence = mean(adherence_score),
    sd_adherence = sd(adherence_score)
  )

print(overall_results)

# Stratified analysis
stratified_results <- data %>%
  group_by(age_group, app_version) %>%
  summarise(
    n = n(),
    mean_adherence = mean(adherence_score),
    sd_adherence = sd(adherence_score),
    .groups = "drop"
  )

print(stratified_results)

cat("\nT test\n")
# T-test overall
t.test(adherence_score ~ app_version, data = data)

cat("\nANOVA\n")
# ANOVA to test interaction between age and version
 aov(adherence_score ~ app_version * age_group, data = data)

# Visualization
ggplot(data, aes(x = app_version, y = adherence_score, fill = app_version)) +
  geom_boxplot() +
  facet_wrap(~ age_group) +
  labs(title = "Medication Adherence by App Version and Age Group",
       y = "Adherence Score (%)",
       x = "App Version") +
  theme_minimal() +
  theme(legend.position = "none")

# T TEST

# The Gamified app group had a mean adherence score of 77.4, while the Standard app averaged 75.6 
# — about 1.75 points higher for Gamified users.

# The p-value (0.061) is slightly above 0.05, meaning:
# There is no statistically significant difference at the conventional 5% level.
# But it’s marginal — we might say it trends toward significance (suggestive but not strong evidence).
# The 95% CI (–0.08 to 3.58) crosses zero → cons

# Two-way ANOVA (App × Age Group)

# | Effect                      | Meaning                    | p-value      | Interpretation                                                          |
# | :-------------------------- | :------------------------- | :----------- | :---------------------------------------------------------------------- |
# | **app_version**             | Main effect of app version | 0.0516       | Marginal — similar to t-test; overall difference between apps is small  |
# | **age_group**               | Main effect of age group   | **6.24e-08** | Very significant — adherence differs strongly by age group overall      |
# | **app_version × age_group** | Interaction                | **8.56e-08** | Very significant — the effect of the app version *depends on* age group |


# | Question                                                    | Result          | Interpretation                  |
# | :---------------------------------------------------------- | :-------------- | :------------------------------ |
# | Is there a global difference between Gamified and Standard? | *p ≈ 0.06*      | Slight, non-significant         |
# | Does adherence vary by age?                                 | ***p < 0.001*** | Strong differences by age group |
# | Does the app’s effect depend on age?                        | ***p < 0.001*** | Yes — clear interaction         |


# NOTES :

# Both t-tests and ANOVA (Analysis of Variance) compare means between groups.

# | Situation                | Typical test |
# | :----------------------- | :----------- |
# | Compare 2 groups         | **t-test**   |
# | Compare 3 or more groups | **ANOVA**    |

# A two-sample t-test and a one-way ANOVA with two groups give the same result (identical p-value).

# summary(aov(adherence_score ~ app_version , data = data))
# summary(aov(adherence_score ~ app_version + age_group, data = data))
# summary(aov(adherence_score ~ app_version * age_group, data = data))

t.test(adherence_score ~ app_version, data = data)
summary(aov(adherence_score ~ app_version , data = data))

#🔹 Why ANOVA is needed

# When you have more than two groups, you could do multiple t-tests —
# but that inflates your Type I error rate (the chance of false positives).

# ANOVA solves that by testing all group means simultaneously with one overall F-test.
# You have three age groups (18–40, 41–65, 65+).

# You want to test if at least one has a different mean adherence.
# → Use ANOVA, not three pairwise t-tests.

# The statistical idea
# Both t-tests and ANOVA compare variance between groups to variance within groups.
# ANOVA generalizes this to many groups → ratio forms the F statistic.

# Step 1: Why post-hoc tests are needed
# When ANOVA gives a significant result, it only tells you:
# “At least one group mean is different from the others.”
# … but it doesn’t tell which groups differ.

#🔹 Step 2: How to test pairwise differences
# You now run pairwise t-tests between all groups — but corrected for multiple comparisons.
# 🔹 Step 3: The correction — controlling false positives
# Doing multiple t-tests increases the chance of false positives (Type I error).
# That’s why we use post-hoc tests with correction.

# Common options:

# Method	What it does
# | Method                                          | What it does                                                           |
# | :---------------------------------------------- | :--------------------------------------------------------------------- |
# | **Tukey’s HSD** (Honest Significant Difference) | The *classic* post-hoc for ANOVA; adjusts for all pairwise comparisons |
# | **Bonferroni**                                  | Very conservative, divides α by #comparisons                           |
# | **Holm**                                        | Slightly less conservative than Bonferroni                             |
# | **Scheffé**                                     | For any linear combination of means (flexible)                         |

anova_model <- aov(adherence_score ~ age_group, data = data)
summary(anova_model)
TukeyHSD(anova_model)

# Each row = one pairwise comparison.
# diff = mean difference.
# lwr, upr = 95% confidence interval.
# p adj = adjusted p-value (corrected for multiple testing)


cat("

From 1 variable → n variables: the general idea

A t-test compares 2 groups (1 variable).

A one-way ANOVA compares 3+ groups (1 variable).

A two-way ANOVA analyzes 2 variables (and their interaction).

A multi-way ANOVA (sometimes called n-way ANOVA) analyzes n categorical independent variables — and all their possible interactions.

expands automatically into:

p_version+age_group+gender+(app_version:age_group)+(app_version:gender)+(age_group:gender)+(app_version:age_

Once you include many factors, there are many possible comparisons.

You can still use Tukey’s HSD, but now specify which factor or interaction you’re testing.

")



TukeyHSD(anova_model, "age_group")
TukeyHSD(anova_model, "app_version:age_group")

# data$app_version <- factor(data$app_version)
# data$age_group <- factor(data$age_group)

# anova_model <- aov(adherence_score ~ app_version * age_group, data = data)

library(emmeans) 

# (Estimated Marginal Means) package
# This gives you pairwise comparisons within each age group and automatically adjusts p-values for multiple tests.
# emmeans(anova_model, pairwise ~ app_version | age_group)


cat("

When “n variables” becomes too many

If you have many predictors, not all categorical, or continuous ones too:

You move beyond ANOVA to a multiple linear regression model:

lm(adherence_score ~ app_version + age_group + gender + bmi + time_since_diagnosis, data = data)

summary()

")

print("Fit a 3-way ANOVA (app × age × gender) ")

library(tidyverse)
library(emmeans)

set.seed(123)
n <- 900

data3 <- tibble(
  patient_id  = 1:n,
  age_group   = sample(c("18-40", "41-65", "65+"), n, replace = TRUE,
                       prob = c(0.3, 0.4, 0.3)),
  app_version = sample(c("Standard", "Gamified"), n, replace = TRUE),
  gender      = sample(c("Male", "Female"), n, replace = TRUE)
) %>%
  mutate(
    # Base mean by age
    base_mean = case_when(
      age_group == "18-40" ~ 75,
      age_group == "41-65" ~ 78,
      age_group == "65+"   ~ 73
    ),
    
    # App effect depends on age (interaction app × age)
    app_effect = case_when(
      app_version == "Gamified" & age_group == "18-40" ~ +7,
      app_version == "Gamified" & age_group == "41-65" ~ +2,
      app_version == "Gamified" & age_group == "65+"   ~ -3,
      TRUE ~ 0
    ),
    
    # Small gender effect (e.g. females slightly higher adherence)
    gender_effect = case_when(
      gender == "Female" ~ +2,
      TRUE ~ 0
    ),
    
    # Generate adherence with some noise
    adherence_score = base_mean + app_effect + gender_effect + rnorm(n, 0, 10),
    adherence_score = pmin(pmax(adherence_score, 0), 100)
  )

head(data3)

ref_grid(anova_model)


library(tidyverse)
library(emmeans)

# ... data3 simulated ...

anova3 <- aov(adherence_score ~ app_version * age_group * gender,
              data = data3)
summary(anova3)


# Get estimated marginal means for app × age × gender
emm_full <- emmeans(anova3, ~ app_version * age_group * gender)
emm_df   <- as.data.frame(emm_full)

head(emm_df)




print("Adjustments for Multiple Testing")

cat("

🎯 1. Why we need adjustment
# When you test many hypotheses at once, you increase your chance of finding false positives (Type I errors).
# Example:
# You run 20 independent tests with α = 0.05.
# Even if none of them are truly significant, on average
# 20 × 0.05 = 1 test will appear significant just by chance.
# So the more tests you do, the higher the risk of incorrectly calling something “significant.

# 🧩 2. The logic behind “20 × 0.05 = 1”
# When we say:
# “If you run 20 tests at α = 0.05, about 1 will be significant by chance,”
# we mean on average, if all null hypotheses are actually true.

# Here’s the reasoning:

# For each hypothesis test, when the null hypothesis is true,
# the probability of a false positive (incorrectly rejecting H₀) is α = 0.05.

# The number of false positives among m independent tests follows a Binomial distribution:
# X ∼ Binomial(n=m,p=α)

# The expected (average) number of false positives is: E[X] = m × α
# So if  α = 0.05: E[X] = 20 × 0.05 = 1
# That means: on average, you’ll get 1 false positive per 20 tests — purely due to random chance, 
# even if nothing real is happening.

")

cat("
| Concept        | What it means                                                                     |
| -------------- | --------------------------------------------------------------------------------- |
| **FWER**       | Probability that **at least one** false positive occurs                           |
| **FDR**        | Expected **proportion** of false positives among the significant ones             |
| **Bonferroni** | Protects strongly against *any* false positives (too strict in large-scale data)  |
| **BH / FDR**   | Allows some false positives but controls their *rate* — better for large datasets |
")


cat("

FWER – Family-Wise Error Rate

Probability of making at least one false positive among all tests.

𝑚
m = number of hypotheses/tests

𝛼
α = significance threshold (e.g. 0.05)

")

alpha <- 0.05
m <- 20
FWER <- 1 - (1 - alpha)^m
FWER

cat("

Bonferroni correction

The Bonferroni method controls the FWER by making each test stricter:

Only p-values ≤ α′ are considered significant.

")

alpha <- 0.05
m <- 20
alpha_prime <- alpha / m
alpha_prime



cat("

FDR – False Discovery Rate

Expected proportion of false positives among the rejected (significant) hypotheses.

FDR=E[V/R]
𝑉
V = number of false positives

𝑅
R = total number of rejected hypotheses
(if R=0, FDR = 0 by definition)

FDR doesn’t forbid false positives — it limits their proportion.
")

cat("

Benjamini–Hochberg (BH) procedure

This controls the FDR, not FWER — it’s less strict and better for large-scale data (e.g., gene expression).

Algorithm:

Sort p-values ascending: 
Compute critical values:
Find the largest i such that ≤p(i)crit
	​
All p-values up to that i are significant.
")

pvals <- c(0.001, 0.02, 0.03, 0.07, 0.5)
p.adjust(pvals, method = "BH")


cat("

| Goal                  | What it’s protecting against                          |
| --------------------- | ----------------------------------------------------- |
| **Bonferroni / FWER** | “I don’t want *any* false positives.”                 |
| **BH / FDR**          | “I accept that maybe 5% of my discoveries are false.” |

")



cat("Non-parametric tests\n")

cat('
Scenario, Parametric Test, Non-Parametric Alternative, Why Non-Parametric?

Conversion rate (binary), Z-test / Chi-square,Fisher’s Exact Test, "Small sample, rare events"

"Time on page, revenue (continuous)", T-test, Mann-Whitney U (Wilcoxon rank-sum), "Skewed, heavy-tailed data"

Ordinal ratings (1–5 stars),T-test / ANOVA,Wilcoxon / Kruskal-Wallis, Not interval-scale

Multiple groups,ANOVA,Kruskal-Wallis,Non-normal residuals
')

# Test,Use Case,Non-Parametric Test,R Code
# 1,Click-through rate (small n),Fisher’s Exact,fisher.test(table)
# 2,Time to book appointment (skewed),Mann-Whitney U,wilcox.test()
# 3,Patient satisfaction (1–5),Wilcoxon rank-sum,wilcox.test()
# 4,Cost per patient (outliers),Mann-Whitney U,wilcox.test()

# ?wilcox.test

cat("

The Mann–Whitney U test and the Wilcoxon rank-sum test are essentially the same test, 
but they come from different historical origins and notations.

Both tests compare two independent samples to determine whether their distributions differ, 
typically interpreted as testing whether one tends to have larger values than the other.

They are both non-parametric alternatives to the two-sample t-test.

In R: wilcox.test() = Mann-Whitney U = Wilcoxon rank-sum

")


cat("

| Aspect                 | **Mann–Whitney U test**                                                                                   | **Wilcoxon rank-sum test**                  |
| ---------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **Origin**             | Developed by Henry Mann & Donald Whitney (1947)                                                           | Developed earlier by Frank Wilcoxon (1945)  |
| **Statistic computed** | ( U ) statistic (based on counts of rank comparisons)                                                     | ( W ) statistic (sum of ranks in one group) |
| **Relationship**       | ( U = W - \frac{n_1(n_1 + 1)}{2} )                                                                        | ( W = U + \frac{n_1(n_1 + 1)}{2} )          |
| **Data type**          | Two independent samples                                                                                   | Two independent samples                     |
| **Interpretation**     | Tests whether the probability that a randomly chosen observation from A is greater than from B equals 0.5 | Same idea, expressed as rank-sum difference |

")

set.seed(123)
groupA <- rnorm(30, mean = 100, sd = 10)
groupB <- rnorm(30, mean = 105, sd = 10)

# Wilcoxon rank-sum test (same as Mann–Whitney)
res <- wilcox.test(groupA, groupB)
res

n1 <- length(groupA)
U <- res$statistic - n1 * (n1 + 1) / 2
U

cat("There’s Also: Wilcoxon Signed-Rank Test")

cat("

| Test Name                                           | Data Type                      | Purpose                                                            | Equivalent Parametric Test |
| --------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------ | -------------------------- |
| **Wilcoxon Rank-Sum Test** (aka **Mann–Whitney U**) | 2 **independent** samples      | Compare whether one group tends to have larger values than another | 2-sample **t-test**        |
| **Wilcoxon Signed-Rank Test**                       | 2 **paired/dependent** samples | Compare before/after (paired) measurements                         | **Paired t-test**          |

")

set.seed(123)

# Simulated data: same 20 patients, before & after treatment
before <- rnorm(20, mean = 140, sd = 10)
after  <- before - rnorm(20, mean = 5, sd = 5)  # treatment lowers BP by ~5

# Paired Wilcoxon signed-rank test
res <- wilcox.test(before, after, paired = TRUE, alternative = "greater")

res

# A step-by-step explanation of how the sum of ranks works in the Mann-Whitney U / Wilcoxon rank-sum test 
# — with visuals, math, and R code.

cat('

Value,Group,Rank
2.1,A,1
3.4,A,2
4.0,B,3
5.2,A,4
6.5,B,5
7.1,B,6

Group,Ranks,Sum of Ranks
A,"1, 2, 4",7
B,"3, 5, 6",14

Statistic,Formula,
Wilcoxon W
Mann-Whitney U

Compare to expected value and compute the p-value
')

# Data
group <- factor(rep(c("A", "B", "C"), each = 3))
values <- c(3, 5, 2, 8, 7, 6, 4, 5, 3)

# Step 1: Rank all observations together
ranks <- rank(values, ties.method = "average")
ranks 

# Combine into a data frame
df <- data.frame(group, values, ranks)
df

# Compute sum of ranks per group

aggregate(ranks ~ group, data = df, FUN = sum)

# Compute the Wilcoxon W statistic

# The Wilcoxon rank-sum test statistic (W) for a group is simply the sum of its 
# WA = 7
# 𝑊𝐵 = 14

# Compute the Mann–Whitney U statistic

# Expected value and variance under H₀

# Compute Z-score and p-value (normal approximation)



cat("Kruskal-Wallis")

cat("

The Kruskal-Wallis test is a non-parametric (distribution-free) statistical test used to compare three or 
more independent groups when the assumptions of one-way ANOVA (e.g., normality and equal variances) are not met. 
It is the non-parametric alternative to one-way ANOVA.

When to Use Kruskal-Wallis

Use it when:

You have 3 or more independent groups (e.g., treatment A, B, C).
The dependent variable is ordinal or continuous but not normally distributed.
You want to test whether the distributions (typically medians) of the groups are the same.

It does not test means directly — it tests whether the ranks of the data differ across groups.

")

cat("

Key Idea: Rank-Based

Combine all data from all groups.

Combine all data from all groups.
Rank all observations from lowest to highest (1, 2, 3, …).
Sum the ranks within each group.
If groups are similar, their sums of ranks should be close.
If one group tends to have higher (or lower) values, its rank sum will differ.

The Kruskal–Wallis statistic 𝐻 : H measures how different those rank sums are.


Steps to Perform Kruskal-Wallis


Rank all data across groups (1 = smallest, handle ties by averaging).
Sum ranks for each group ($R_i$).
Compute H using the formula.
Compare H to critical value from chi-square table (df = k−1) or get p-value.
Reject H₀ if p-value < α (usually 0.05).


Post-Hoc Tests (if H is significant)


Since Kruskal-Wallis is omnibus, follow up with:

Dunn’s test (with Bonferroni correction)
Pairwise Wilcoxon rank-sum (Mann-Whitney) tests

")

set.seed(123)
pain_A <- rpois(20, 5)
pain_B <- rpois(20, 7)
pain_C <- rpois(20, 6)

pain <- c(pain_A, pain_B, pain_C)
group <- factor(rep(c("A_standard", "B_new_drug", "C_physiotherapy"), each = 20))

head(pain)
head(group)

tail(pain)
tail(group)

kruskal.test(pain ~ group)

# p = 0.009 → reject H₀ → at least one treatment group has a different median pain reduction.

cat("Post-hoc pairwise comparisons")

library(FSA)
dunnTest(pain ~ group, method = "bonferroni")

# This means at least one group differs significantly from the others in median pain reduction.
# Post-hoc Dunn pairwise comparisons
# | Comparison                       | Z     | p.unadj | p.adj (Bonferroni) | Interpretation                                |
# | -------------------------------- | ----- | ------- | ------------------ | --------------------------------------------- |
# | **A_standard – B_new_drug**      | −2.30 | 0.021   | **0.064**          | borderline (not significant after Bonferroni) |
# | **A_standard – C_physiotherapy** | +0.63 | 0.528   | 1.000              | not significant                               |
# | **B_new_drug – C_physiotherapy** | +2.94 | 0.0033  | **0.010**          | significant difference                        |

# Summary:
# Kruskal–Wallis p < 0.01 → reject H₀ (some group differs)
# Dunn post-hoc:
# B vs C → significant
# A vs B → borderline
# A vs C → no differenc





# ===================================================================
# SURVIVAL A/B TEST: Standard vs Intensive Therapy
# Simulation + Kaplan-Meier + Log-Rank + Cox (Frequentist)
# ===================================================================

# --------------------------------------------------
# 1. SETUP & PACKAGES
# --------------------------------------------------
set.seed(2025)
library(survival)
library(survminer)     # for nice KM plots
library(dplyr)

# --------------------------------------------------
# 2. SIMULATION PARAMETERS
# --------------------------------------------------
n_per_arm <- 300
arm <- rep(c("A_standard", "B_intensive"), each = n_per_arm)

# True hazard rates: B has 30% lower hazard (better survival)
lambda_A <- 0.10   # 10% hazard per unit time
lambda_B <- 0.07   # 7% hazard per unit time

# --------------------------------------------------
# 3. GENERATE EVENT TIMES (Exponential distribution)
# --------------------------------------------------
rate_vec <- ifelse(arm == "A_standard", lambda_A, lambda_B)
true_event_time <- rexp(n = length(arm), rate = rate_vec)

# --------------------------------------------------
# 4. GENERATE CENSORING TIMES (Independent)
# --------------------------------------------------
censor_time <- rexp(n = length(arm), rate = 0.05)  # ~5% censoring per unit time

# --------------------------------------------------
# 5. OBSERVED DATA
# --------------------------------------------------
time_obs <- pmin(true_event_time, censor_time)
status   <- as.integer(true_event_time <= censor_time)  # 1 = event, 0 = censored

# --------------------------------------------------
# 6. DATA FRAME
# --------------------------------------------------
dat_surv <- data.frame(
  time   = time_obs,
  status = status,
  arm    = factor(arm, levels = c("A_standard", "B_intensive"))
)

cat("Data simulated. First 6 rows:\n")
print(head(dat_surv))

# --------------------------------------------------
# 7. KAPLAN-MEIER CURVES
# --------------------------------------------------
km_fit <- survfit(Surv(time, status) ~ arm, data = dat_surv)

# Print summary
print(km_fit)

# Plot with survminer
ggkm <- ggsurvplot(
  km_fit,
  data = dat_surv,
  risk.table = TRUE,
  pval = TRUE,
  pval.coord = c(5, 0.9),
  conf.int = TRUE,
  xlab = "Time",
  ylab = "Survival Probability",
  title = "Kaplan-Meier: Standard vs Intensive Therapy",
  legend.title = "Treatment Arm",
  legend.labs = c("A: Standard", "B: Intensive"),
  palette = c("#E74C3C", "#3498DB"),
  font.main = c(14, "bold"),
  font.x = c(12),
  font.y = c(12),
  font.tickslab = c(11)
)

print(ggkm)

# --------------------------------------------------
# 8. LOG-RANK TEST
# --------------------------------------------------
logrank <- survdiff(Surv(time, status) ~ arm, data = dat_surv)
logrank_p <- 1 - pchisq(logrank$chisq, df = 1)

cat(sprintf("\nLog-Rank Test:\n"))
cat(sprintf("  Chi-square = %.2f, df = 1, p-value = %.4f\n",
            logrank$chisq, logrank_p))

# --------------------------------------------------
# 9. COX PROPORTIONAL HAZARDS MODEL (Frequentist)
# --------------------------------------------------
cox_fit <- coxph(Surv(time, status) ~ arm, data = dat_surv)
cox_summary <- summary(cox_fit)

cat(sprintf("\nCox Model Results:\n"))
print(cox_summary)

# Hazard Ratio (B vs A)
HR <- exp(coef(cox_fit))
CI <- exp(confint(cox_fit))

HR_table <- data.frame(
  HR        = round(HR[1], 3),
  Lower_95  = round(CI[1], 3),
  Upper_95  = round(CI[2], 3),
  p_value   = round(coef(summary(cox_fit))[,"Pr(>|z|)"][1], 4)
)
rownames(HR_table) <- "B_intensive vs A_standard"

cat("\nHazard Ratio (B vs A):\n")
print(HR_table)

# --------------------------------------------------
# 10. FINAL SUMMARY TABLE (Log-Rank + Cox)
# --------------------------------------------------
summary_table <- data.frame(
  Method     = c("Log-Rank", "Cox (Frequentist)"),
  HR_B_vs_A  = c(NA, round(HR[1], 3)),
  Lower_95   = c(NA, round(CI[1], 3)),
  Upper_95   = c(NA, round(CI[2], 3)),
  p_value    = c(round(logrank_p, 4),
                 round(coef(summary(cox_fit))[,"Pr(>|z|)"][1], 4))
)

cat("\n=== FINAL RESULTS SUMMARY ===\n")
print(summary_table, row.names = FALSE)

# --------------------------------------------------
# 11. OPTIONAL: Save plot
# --------------------------------------------------
# ggsave("km_plot.png", ggkm$plot, width = 8, hei_





cat("Homoscedasticity")

cat("

1. Homoscedasticity (Good!)
Definition: The variance of the residuals (errors) is constant across all levels of the independent variable(s).

In simple terms: The spread of the data points around the regression line is the same whether the predicted value is low, 
medium, or high.

→ Variance of the error = constant.

Why It Matters:

Required for valid inference in linear regression (e.g., p-values, confidence intervals).

")

cat("Heteroscedasticity")

cat("

2. Heteroscedasticity (Problem !)

Definition: The variance of the residuals is NOT constant — it changes with the level of the independent variable(s).

The spread of errors increases or decreases as X increases.

→ Variance of the error changes.

Issue,Consequence

Biased standard errors,p-values and confidence intervals are wrong
Inefficient estimates,OLS is no longer BLUE (Best Linear Unbiased Estimator)
Invalid hypothesis tests,Can't trust t-tests or F-tests

")


# ================================================================
# Example: Homoscedastic vs. Heteroscedastic residuals
# ================================================================

set.seed(123)

# Generate predictor x
x <- seq(1, 100, by = 1)

# ------------------------------
# Homoscedastic case
# ------------------------------
# Errors have constant variance (sd = 5)
y_homo <- 3 + 0.5 * x + rnorm(length(x), mean = 0, sd = 5)
df_homo <- data.frame(x = x, y = y_homo)

# Fit model
model_homo <- lm(y ~ x, data = df_homo)

# Plot residuals vs fitted values
plot(model_homo, which = 1, main = "Homoscedastic residuals (constant variance)")

# ------------------------------
# Heteroscedastic case
# ------------------------------
# Errors have variance that grows with x (sd = 0.1 * x)
y_hetero <- 3 + 0.5 * x + rnorm(length(x), mean = 0, sd = 0.1 * x)
df_hetero <- data.frame(x = x, y = y_hetero)

# Fit model
model_hetero <- lm(y ~ x, data = df_hetero)

# Plot residuals vs fitted values
plot(model_hetero, which = 1, main = "Heteroscedastic residuals (fan shape)")


cat('

Formal statistical tests:

library(lmtest)
bptest(model)   # Breusch–Pagan test

H₀: homoscedasticity
p < 0.05: reject H₀ → heteroscedasticity present

')

library(lmtest)

bptest(model_homo)

bptest(model_hetero)

cat("

| Concept                | Description             | Implication                                 |
| ---------------------- | ----------------------- | ------------------------------------------- |
| **Homoscedasticity**   | Constant error variance | OLS assumptions valid                       |
| **Heteroscedasticity** | Error variance changes  | Standard errors invalid, use robust methods |

")

cat("

OLS = Ordinary Least Squares

Ordinary Least Squares (OLS) is the most common method used to estimate the parameters of a linear regression model.

In plain terms:

OLS finds the “best-fitting” straight line through your data — the one that minimizes the sum of squared differences 
between the actual and predicted values.

# Simple OLS regression

model <- lm(y ~ x, data = df)
summary(model)

")

cat("What to do about heteroscedasticity ?
    
Typical transforms :

If variance grows with the mean → log transform

If counts → sqrt or log

If proportions near 0 or 1 → logit or arcsin sqrt

")

set.seed(123)
x <- 1:100

# Errors with increasing variance (heteroscedastic)
y_hetero <- 3 + 0.5 * x + rnorm(length(x), mean = 0, sd = 0.1 * x)
df_hetero <- data.frame(x = x, y = y_hetero)

# OLS model
model_hetero <- lm(y ~ x, data = df_hetero)

# Plot residuals vs fitted
plot(model_hetero, which = 1, main = "Heteroscedastic residuals (fan shape)")

library(lmtest)
bptest(model_hetero)

# Transform response
df_hetero$y_log <- log(df_hetero$y)

# Refit model
model_log <- lm(y_log ~ x, data = df_hetero)

# Plot residuals again
plot(model_log, which = 1, main = "After log-transform of y")

# Test again
bptest(model_log)


cat("

Generalized Linear Models (GLMs) can indeed help handle heteroscedasticity, but not in the same way as “fixing” it in a linear model.

Let’s unpack this properly — both conceptually and with R examples 👇

What is a GLM :

A Generalized Linear Model (GLM) extends ordinary linear regression (lm) by allowing:

Non-normal outcome distributions (e.g. Binomial, Poisson, Gamma, etc.)

Non-constant variance (heteroscedasticity) that depends on the mean

A link function connecting the mean to predictors

GLMs don’t “fix” heteroscedasticity — they model it

by using the correct likelihood function for the data’s mean–variance relationship.

So:

If your residual variance grows with the mean → try a Gamma GLM

If variance ≈ mean → Poisson GLM

If binary → Binomial GLM

For arbitrary variance → Weighted LS or GLS

")

cat('

# Generalized linear model (variance modeled via Gamma)
glm(y ~ x, family = Gamma(link = "log"), data = df)

')



cat(' 

What is LOGIT ?

The logit function transforms a probability (between 0 and 1) into a real number between 
−∞ and +∞, and is defined as the logarithm of the odds:

logit(𝑝) = ln(p / (1-p))

| Probability (p) | Logit(p) Formula | Result      |
| --------------- | ---------------- | ----------- |
| (p = 0.5)       | (ln(1))         | (0)         |
| (p = 0.8)       | (ln(4))         | ≈ **1.39**  |
| (p = 0.2)       | (ln(0.25))      | ≈ **−1.39** |

To get back the probability, you apply the inverse logit (also called the sigmoid function).

| Step                  | Formula                 | Interpretation                                       |
| --------------------- | ----------------------- | ---------------------------------------------------- |
| **Forward (logit)**   | ( log(frac{p}{1-p}) ) | Converts a probability → linear predictor (log-odds) |
| **Inverse (sigmoid)** | ( frac{1}{1+e^{-x}} )  | Converts a linear predictor → probability            |

The inverse logit produces an S-shaped curve (“sigmoid”) that maps any real number → [0, 1].

That’s why it’s so popular in:
Logistic regression (binary classification)
Neural networks (sigmoid activation)
Probability calibration

')



cat(' 

What is Welch T-test ?

| Situation                                          | Recommended test   |
| -------------------------------------------------- | ------------------ |
| Two independent samples with **equal variances**   | Student’s t-test   |
| Two independent samples with **unequal variances** | **Welch t-test** ✅ |
| Two related (paired) samples                       | Paired t-test      |

So if your groups have different spreads (heteroscedasticity), Welch’s t-test is the safer and more robust choice.
')

set.seed(123)
groupA <- rnorm(30, mean = 10, sd = 3)
groupB <- rnorm(40, mean = 12, sd = 6)

# Standard t-test with equal variance assumption
t.test(groupA, groupB, var.equal = TRUE)

# Welch’s t-test (default in R)
t.test(groupA, groupB) # or
t.test(groupA, groupB, var.equal = FALSE)


cat('

Quick diagnostic before choosing

You can check equality of variances with Levene’s test:

')

set.seed(123)

# Two groups with different spreads
groupA <- rnorm(30, mean = 10, sd = 2)
groupB <- rnorm(30, mean = 10, sd = 5)

# Combine into a data frame
df <- data.frame(
  values = c(groupA, groupB),
  group  = factor(rep(c("A", "B"), each = 30))
)

head(df)
tail(df)

# Install and load the car package if not installed
# install.packages("car")
library(car)

# Levene’s test
leveneTest(values ~ group, data = df)

cat('

| Statistic  | Value                                             | Interpretation                |
| ---------- | ------------------------------------------------- | ----------------------------- |
| F(1, 58)   | 10.46                                             | Significant                   |
| p-value    | 0.0020                                            | Reject H₀ → Unequal variances |
| Conclusion | Groups have different variances (heteroscedastic) |                               |
'
)

cat('

If you understand how Welch’s t-test works for two groups (handling unequal variances), 
then Welch’s ANOVA is simply the generalization of that idea to three or more groups.

Welch’s ANOVA is a version of the standard one-way ANOVA that does not assume equal variances (homoscedasticity) across groups.
It’s often called:

“One-way ANOVA with unequal variances”

It is more robust when:

Groups have different variances, and/or

Groups have unequal sample sizes

| Feature                          | Classic ANOVA (`aov`) | Welch’s ANOVA                         |
| -------------------------------- | --------------------- | ------------------------------------- |
| Variances assumed equal?         | ✅ Yes                 | ❌ No                                  |
| Uses pooled variance?            | ✅ Yes                 | ❌ No                                  |
| Type of F-statistic              | Standard F            | Adjusted F (Welch correction)         |
| Robust to unequal n & variances? | ❌                     | ✅ Yes                                 |
| Function in R                    | `aov()`               | `oneway.test(..., var.equal = FALSE)` |

')

set.seed(123)

# Three groups with unequal variances
groupA <- rnorm(30, mean = 10, sd = 2)
groupB <- rnorm(30, mean = 12, sd = 4)
groupC <- rnorm(30, mean = 15, sd = 6)

df <- data.frame(
  values = c(groupA, groupB, groupC),
  group  = factor(rep(c("A", "B", "C"), each = 30))
)

# Standard ANOVA (assumes equal variances)
cat("aov")
anova_eq <- aov(values ~ group, data = df)
summary(anova_eq)

# Welch’s ANOVA (robust to unequal variances)
cat("oneway.test")
welch_anova <- oneway.test(values ~ group, data = df, var.equal = FALSE)
welch_anova

cat('

Welch’s ANOVA is a robust version of one-way ANOVA that allows for unequal variances and unequal sample sizes.
It uses an adjusted F-statistic and fractional degrees of freedom.

')

cat('

If the Welch ANOVA is significant → you can run Games-Howell post-hoc tests, 
which are the analog of Tukey’s test but for unequal variances.

')

library(rstatix)
games_howell_test(df, values ~ group)


# | Comparison | Mean difference (`estimate`) | 95% CI (`conf.low`–`conf.high`) | Adjusted p-value (`p.adj`) | Significance (`p.adj.signif`) | Interpretation                  |
# | ---------- | ---------------------------- | ------------------------------- | -------------------------- | ----------------------------- | ------------------------------- |
# | **A vs B** | 2.81                         | [1.10, 4.52]                    | 0.000708                   | ***                           | Significant difference          |
# | **A vs C** | 5.24                         | [2.76, 7.73]                    | 0.0000261                  | ****                          | Strongly significant difference |
# | **B vs C** | 2.43                         | [–0.30, 5.17]                   | 0.09                       | ns                            | Not significant                 |



cat(' Confidence Intervals : 

A confidence interval is a range of values that is likely to contain the true population parameter (e.g., mean, proportion, difference) with a certain level of confidence.

It answers: "If I repeated my study many times, in what range would the true value usually fall?"

A confidence interval tells you:

CI = Estimate ± t ∗×SE

Where:

Estimate = sample mean (or coefficient)

SE = standard error (how variable your estimate is)

t* = critical value from the t-distribution for the desired confidence level (e.g., 1.96 for 95%)

Where:

Estimate = sample mean (or coefficient)

SE = standard error (how variable your estimate is)

t* = critical value from the t-distribution for the desired confidence level (e.g., 1.96 for 95%).

It reflects both precision (width of the interval) and certainty (confidence level).

| Level | Wider or Narrower | Confidence     |
| ----- | ----------------- | -------------- |
| 90%   | Narrower          | Less confident |
| 95%   | Medium            | Standard       |
| 99%   | Wider             | More confident |

True population mean (μ) = 50

Sample 1: 95% CI = [48, 54] → contains μ
Sample 2: 95% CI = [46, 52] → contains μ
Sample 3: 95% CI = [51, 57] → contains μ
...
Sample 96: 95% CI = [44, 48] → misses μ

→ 95 out of 100 intervals capture μ

')


cat('

Formula (for a mean with known standard error)

CI = 𝑥 ± 𝑧 ∗ SE

Where:

𝑥 = sample mean

𝑧 = z-score for the desired confidence level (e.g., 1.96 for 95%)

SE = standard error of the mean

')

# Simulate sample data
set.seed(123)
sample <- rnorm(100, mean = 50, sd = 10)

# Calculate 95% confidence interval
mean_val <- mean(sample)
se <- sd(sample) / sqrt(length(sample))
ci_lower <- mean_val - 1.96 * se
ci_upper <- mean_val + 1.96 * se

cat(sprintf("95%% CI: [%.2f, %.2f]\n", ci_lower, ci_upper))

# It does NOT mean: There's a 95% probability the true value is in this specific interval 
# (the true value either is or isn't in there - we just don't know which).

cat('

The Core Concept : 95% Confidence Interval means:

If we repeated this study 100 times with different samples
About 95 of those intervals would contain the true population parameter
About 5 would miss it

')

cat('

Standard Deviation (SD) : Spread of individual data points

Standard Error (SE) : Spread of sample means

Standard Error : 

Standard Error ExplainedStandard Error (SE) measures how much sample statistics (like the mean) vary from sample to sample.
It tells you: "If I took many samples, how spread out would the sample means be?"

SE = SD / √n

where:
- SD = standard deviation
- n = sample size

')

library(tidyverse)

set.seed(789)
n <- 100

# Generate cholesterol data
cholesterol <- rnorm(n, mean = 200, sd = 40)

# Calculate statistics
mean_chol <- mean(cholesterol)
sd_chol <- sd(cholesterol)
se_chol <- sd_chol / sqrt(n)

cat("=== CHOLESTEROL LEVELS ===\n")
cat("Sample size:", n, "\n")
cat("Mean:", round(mean_chol, 2), "mg/dL\n")
cat("SD:", round(sd_chol, 2), "mg/dL ← variability between patients\n")
cat("SE:", round(se_chol, 2), "mg/dL ← uncertainty about the mean\n\n")

# 95% CI using SE
ci_lower <- mean_chol - 1.96 * se_chol
ci_upper <- mean_chol + 1.96 * se_chol

cat("95% CI:", round(ci_lower, 2), "to", round(ci_upper, 2), "\n")
cat("This CI is built using the SE!\n\n")

# ===== VISUALIZE =====

chol_data <- tibble(cholesterol = cholesterol)

ggplot(chol_data, aes(x = cholesterol)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, 
                 fill = "lightblue", alpha = 0.7) +
  
  # Add SD markers
  geom_vline(xintercept = mean_chol, color = "red", linewidth = 1) +
  geom_vline(xintercept = c(mean_chol - sd_chol, mean_chol + sd_chol), 
             color = "blue", linetype = "dashed", linewidth = 1) +
  
  annotate("text", x = mean_chol, y = 0.012, label = "Mean", 
           color = "red", vjust = -0.5) +
  annotate("text", x = mean_chol - sd_chol, y = 0.012, label = "Mean - SD", 
           color = "blue", vjust = -0.5, angle = 90) +
  
  # Add SE visualization (not on same scale, shown in subtitle)
  labs(title = "Cholesterol Distribution",
       subtitle = paste0("Mean = ", round(mean_chol, 1), " mg/dL, ",
                        "SD = ", round(sd_chol, 1), " (blue lines), ",
                        "SE = ", round(se_chol, 1)),
       x = "Cholesterol (mg/dL)",
       y = "Density") +
  theme_minimal()

cat(' Central limit Theorem')

cat("

Central Limit Theorem (CLT) – Complete Explanation

The Central Limit Theorem (CLT) is one of the most powerful and fundamental ideas in statistics.

In simple terms:

No matter what the original population looks like (even if it's skewed, bimodal, or weird),
the distribution of sample means will be approximately normal — as long as the sample size is large enough.

Central Limit Theorem = Magic of Averages

Take enough samples → averages behave normally → This is why confidence intervals and t-tests work in real life

")






