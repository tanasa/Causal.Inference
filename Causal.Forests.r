cat("CAUSAL FORESTS")

cat("

🌲 1️⃣ What is a Causal Forest?

A Causal Forest is a machine-learning method (built on random forests) that estimates causal effects, not just predictions.

Instead of predicting an outcome Y, it estimates how much the treatment changes Y for each observation.

🧠 2️⃣ Intuition

Regular random forests:

Fit many decision trees that predict Y from X.

Average the predictions.

Causal forests:

Fit many trees that learn treatment effect heterogeneity — how the effect of treatment differs across subgroups.

Average those treatment effects.

They combine:

Random forest flexibility (nonlinear, nonparametric, no need for model form)

Causal inference structure (treatment vs control, confounding adjustment)

🔬 3️⃣ The logic step-by-step

Treatment variable 

W∈{0,1} : (e.g., treated vs control)
Outcome variable 𝑌
Covariates X (possible confounders)

A CAUSAL FOREST SPLITS THE DATA INTO TREES TO FIND SUBGROUPS WHERE:

THE TREATMENT EFFECT DIFFERS SIGNIFICANTLY,

BUT COVARIATES ARE STILL BALANCED (CONFOUNDING IS CONTROLLED LOCALLY)

💡 4️⃣ Example
Example: Effect of an exercise program on blood pressure

| Variable | Meaning                                    |
| -------- | ------------------------------------------ |
| **Y**    | Blood pressure after 3 months              |
| **W**    | 1 = received exercise program; 0 = control |
| **X**    | Age, BMI, baseline BP, smoking, diet score |

You suspect the program might help some people more than others (e.g., younger or overweight individuals).

With a causal forest: You can estimate:

Overall treatment effect (ATE): average reduction in BP due to the program.

Conditional effects: how much BP drops for each person or subgroup, given their 𝑋

An example : 

| Subgroup        | Estimated effect (Δ BP) |
| --------------- | ----------------------- |
| Young, high BMI | −10 mmHg                |
| Older, low BMI  | −3 mmHg                 |
| Smokers         | −1 mmHg                 |


")

cat(" 

🧩 1️⃣ The three pillars of causal notation :

| Symbol | Meaning                                      | Typical Example                                    |
| ------ | -------------------------------------------- | -------------------------------------------------- |
| **X**  | Covariates (or features, confounders)        | Age, gender, BMI, baseline income, smoking status  |
| **W**  | Treatment or exposure (binary or continuous) | 1 = treated / 0 = control, or dosage amount        |
| **Y**  | Outcome variable                             | Blood pressure, income, disease status, exam score |

⚙️ 2️⃣ What each represents conceptually

=========================== X: COVARIATES

Observed characteristics of the units (individuals, patients, users, etc.).

Used to control for confounding — variables that affect both treatment and outcome.

Not manipulated — they’re “background” context.

Example: X = [Age, BMI, Smoking]

=========================== W: TREATMENT (EXPOSURE) : The variable whose CAUSAL EFFECT you want to ESTIMATE. ✅

The VARIABLE whose CAUSAL EFFECT you want to ESTIMATE. ✅

Can be:

BINARY : treated (1) vs control (0)

CONTINUOUS : dosage, intensity, exposure time, etc.

Example: W = ExerciseProgram (1 = received, 0 = not)

=========================== Y: OUTCOME

The result or endpoint potentially affected by the treatment.

What you care about changing or explaining.

Example: Y = Blood Pressure After 3 Months

")

cat("

IN A CAUSAL FOREST:

X: INPUTS (AGE, BMI, ETC.)

W: TREATMENT VARIABLE

Y: OBSERVED OUTCOME

THE MODEL OUTPUTS AN ESTIMATED TREATMENT EFFECT FOR EACH INDIVIDUAL. ✅

")

cat(' An example :

| Person | Age | BMI | Smoking | Exercise (W) | BP After 3 Mo (Y) |
| ------ | --- | --- | ------- | ------------ | ----------------- |
| A      | 25  | 28  | No      | 1            | 110               |
| B      | 60  | 25  | Yes     | 0            | 135               |
| C      | 45  | 30  | Yes     | 1            | 125               |

Here:

X={Age,BMI,Smoking}

W=Exercise

Y=Blood Pressure

The causal question: 

“What is the effect of exercise (W) on blood pressure (Y), after accounting for Age, BMI, and Smoking (X)?”

WHAT IS THE EFFECT OF EXERCISE (W) ON BLOOD PRESSURE (Y), AFTER ACCOUNTING FOR AGE, BMI, AND SMOKING (X)? ✅

✅ 5️⃣ Summary

| Symbol | Name                             | Role                                                     |
| ------ | -------------------------------- | -------------------------------------------------------- |
| **X**  | Covariates (context/confounders) | Control variables used to adjust and model heterogeneity |
| **W**  | Treatment / Intervention         | What we manipulate or compare                            |
| **Y**  | Outcome                          | The effect we measure                                    |


')



library(grf)

# Simulate data
set.seed(73)
n <- 2000

X <- data.frame(age = runif(n, 20, 70),
                bmi = rnorm(n, 25, 4))
W <- rbinom(n, 1, 0.5)

# Heterogeneous treatment effect: stronger effect for high BMI
tau <- 5 + 0.5 * (X$bmi - 25)
Y <- 120 - 0.8*X$age + tau*W + rnorm(n, 0, 5)

cat("X")
head(X)
tail(X)

cat("W")
head(W)
tail(W)

cat("Y")
head(Y)
tail(Y)

# Fit causal forest
cf <- causal_forest(X, Y, W)
cat("causal forest\n")
print(cf)

# Estimate average and individual treatment effects
ate <- average_treatment_effect(cf)
cat("ate\n")
cat(ate) 

head(predict(cf))
tail(predict(cf))
# head(predict(cf)$predictions)
# tail(predict(cf)$predictions)

cat("\nAverage treatment effect:", round(ate[1], 2), "\n")
cat("On average, the treatment (W) increases the outcome (Y) by 5.03 units, after adjusting for all covariates (X).")

hist(predict(cf)$predictions, main = "Estimated Individual Treatment Effects")

cat("\n\n")
cat("\naverage_treatment_effect\n")
ate <- average_treatment_effect(cf, target.sample = "all")
ate


names(cf)

head(predict(cf))
tail(predict(cf))

library("ggplot2")

# 'X.orig' 'Y.orig' 'W.orig' 'Y.hat' 'W.hat' 'clusters'

# head(cf$X.orig, 2)
# head(cf$Y.orig, 2)
# head(cf$W.orig, 2)

head(cf$clusters)

vi <- variable_importance(cf)
print(vi)
data.frame(variable = colnames(cf$X.orig), importance = vi) |>
           dplyr::arrange(dplyr::desc(importance))

# Assuming you already have: cf <- causal_forest(X, Y, W)

pred <- predict(cf, estimate.variance = TRUE)
cate <- as.numeric(pred$predictions)

# Replace "bmi" with an actual column in your X data frame/matrix
bmi <- cf$X.orig[, "bmi"]
ggplot(data.frame(bmi, cate), aes(bmi, cate)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "loess", se = TRUE) +
  labs(title = "CATE vs BMI", x = "BMI", y = "Estimated CATE")

# Replace "bmi" with an actual column in your X data frame/matrix
bmi <- cf$X.orig[, "age"]
ggplot(data.frame(bmi, cate), aes(bmi, cate)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "loess", se = TRUE) +
  labs(title = "CATE vs AGE", x = "AGE", y = "Estimated CATE")



cat("

🩺 1️⃣ CATE vs BMI

The blue LOESS line shows a strong positive trend:

As BMI increases, the estimated treatment effect (CATE) also increases.

Around BMI ≈ 18–20, the effect is small (~2–3 units).

By BMI ≈ 30, the effect grows to ~7.5 units.

✅ Interpretation:

The treatment (e.g., exercise, intervention, or therapy) is much more effective for individuals with higher BMI.
In other words, people with higher BMI benefit more strongly from the treatment on the outcome (Y).

This indicates treatment effect heterogeneity — the effect size depends strongly on BMI.

👩‍🦳 2️⃣ CATE vs AGE

The CATE line for AGE is relatively flat, with only mild curvature:

Slight dip around age 30–40, minor increase near 50–60.

Overall range of effect: ~4.5 to 5.5 units.

✅ Interpretation:

The treatment effect does not vary substantially with age — it remains roughly constant across the age spectrum.

So while BMI drives meaningful heterogeneity, age is not a strong moderator of treatment effect in this dataset.

")



cat("

📊 2️⃣ Significance for the Average Treatment Effect (ATE)

")

ate <- average_treatment_effect(cf, target.sample = "all", method = "AIPW")
ate

cat("

✅ Interpretation:

The average treatment effect is statistically significant, because the 95% confidence interval does not include 0.

This is analogous to a significant regression coefficient — you can think of it as “the treatment overall has a real effect.”

")

cat("

🧩 3️⃣ Significance for individual (CATE) estimates

Each predicted treatment effect comes with a variance estimate when you run:

")

pred <- predict(cf, estimate.variance = TRUE)
cate <- as.numeric(pred$predictions)
se   <- sqrt(as.numeric(pred$variance))

ci_l <- cate - 1.96 * se
ci_u <- cate + 1.96 * se

cat("

You can then check which individuals have confidence intervals that exclude 0:

")

sig_pos <- mean(ci_l > 0)  # share with significantly positive effect
sig_neg <- mean(ci_u < 0)  # share with significantly negative effect

c(sig_positive = sig_pos, sig_negative = sig_neg)

cat(" \n \n ✅ Interpretation:

sig_positive → proportion of individuals with statistically positive effect

sig_negative → proportion with significantly negative effect

Usually, most individuals won’t have individually significant CATEs (wide CIs), 
but patterns across groups (e.g., by BMI) can be meaningful.

")

cat("

⚙️ 4️⃣ Testing heterogeneity significance

You can test whether heterogeneity itself is statistically significant, using best linear projection (BLP):

")

test_calibration(cf)

cat(" \n

This performs a formal test of calibration:

Null hypothesis: the forest’s heterogeneity is random (no systematic variation)

If the p-value is small → there is significant treatment effect heterogeneity.

")

cat("SUMMARY")

cat("

🧩 1️⃣ Recall what each variable represents

| Symbol | Meaning                                               | Role in the causal model                                |
| ------ | ----------------------------------------------------- | ------------------------------------------------------- |
| **W**  | Treatment (e.g. *exercise* = 1, no exercise = 0)      | The causal intervention whose effect you are estimating |
| **Y**  | Outcome (e.g. *blood pressure*, *heart health score*) | The result potentially affected by exercise             |
| **X**  | Covariates (e.g. *BMI*, *Age*, Smoking, etc.)         | Context or confounders that may affect both W and Y     |

🧠 2️⃣ What the causal forest does : 

How much does the treatment (exercise) change the outcome (Y) for someone with characteristics X (like BMI or Age)?

So:

Exercise (W) is the causal variable.

BMI and Age (X) are moderators or effect modifiers — they tell us for whom the exercise has a bigger or smaller effect.

📊 3️⃣ What your plots show

You plotted Estimated CATE vs BMI and CATE vs AGE:

CATE vs BMI:
Strong upward trend — higher BMI → stronger treatment effect.
➜ Exercise has a larger causal effect on heart health (or BP reduction) for people with higher BMI.

CATE vs AGE:
Flat relationship — effect of exercise is roughly the same across ages.
➜ Age does not significantly modify the causal effect of exercise.

💡 4️⃣ Interpretation in plain English

The causal variable is exercise — that’s what changes the outcome.

BMI and Age are not causal themselves in this model, but they help explain heterogeneity — i.e., who benefits most from the causal effect of exercise.

So:

🔍 6️⃣ In summary

✅ Exercise → directly causes improvement in Y.

⚙️ BMI → modifies (moderates) how strong that causal effect is.

⚙️ Age → does not substantially modify it.

| Variable         | Type                          | Interpretation                                                               |
| ---------------- | ----------------------------- | ---------------------------------------------------------------------------- |
| **Exercise (W)** | **Causal treatment**          | Directly changes the outcome. ATE ≈ +5 units → significant causal effect.    |
| **BMI (X₁)**     | **Effect modifier**           | The effect of exercise increases with BMI — stronger benefit for higher BMI. |
| **Age (X₂)**     | **Covariate (weak modifier)** | Age doesn’t meaningfully change the effect of exercise.                      |


")



cat("The example provided by COPILOT")

library(grf)

# === Simulate Data for Causal Inference ===

n_samples <- 1000  # Number of individuals

# Covariates (e.g., age, education, income, etc.)
covariates <- matrix(rnorm(n_samples * 5), n_samples, 5)

# Treatment assignment (binary: 1 = treated, 0 = control)
treatment <- rbinom(n_samples, 1, 0.5)

# Outcome: depends on treatment, one covariate, and random noise
# True treatment effect is 3; covariate 1 also affects outcome
outcome <- 3 * treatment + covariates[, 1] + rnorm(n_samples)

# === Fit a Causal Forest Model ===

causal_forest_model <- causal_forest(covariates, outcome, treatment)
causal_forest_model 

# Estimate individual-level treatment effects (CATEs)
individual_effects <- predict(causal_forest_model)$predictions
# individual_effects

# Estimate the average treatment effect (ATE)
ate = average_treatment_effect(causal_forest_model)
print(ate)



# install.packages("grf")
library(grf)

# Simulate data
set.seed(123)
n <- 2000
X <- data.frame(
  age = runif(n, 20, 80),
  bmi = rnorm(n, 25, 5),
  smoker = rbinom(n, 1, 0.3)
)
W <- rbinom(n, 1, 0.5)  # Treatment: 1=drug, 0=placebo
tau <- with(X, 10 + 5*(age > 60) - 3*bmi/10 + 4*smoker)  # True CATE
Y <- 100 + tau * W + rnorm(n, 0, 5)  # Outcome

# Fit Causal Forest
cf <- causal_forest(
  X = X,
  Y = Y,
  W = W,
  num.trees = 2000,
  honesty = TRUE
)

# Predict CATE for new patients
new_patients <- data.frame(
  age = c(70, 40),
  bmi = c(30, 25),
  smoker = c(1, 0)
)
cate_pred <- predict(cf, new_patients)

print(cate_pred)



# Load library
library(grf)

# Simulated data: Job training program
set.seed(123)
n <- 2000

# Covariates (features)
age <- rnorm(n, mean = 35, sd = 10)
education <- rnorm(n, mean = 14, sd = 2)
experience <- rnorm(n, mean = 10, sd = 5)

X <- cbind(age, education, experience)
head(X) 
# Treatment assignment (not perfectly random - there's confounding)
propensity <- plogis(-1 + 0.02*age + 0.1*education)
W <- rbinom(n, 1, propensity)
head(W)

# Outcome: Wages with heterogeneous treatment effects
# Key: treatment effect varies by age!
treatment_effect <- 3000 + 200*(age - 35) - 100*(age - 35)^2/10
Y <- 30000 + 500*education + 100*experience + W*treatment_effect + rnorm(n, sd = 5000)
head(Y) 

# Fit causal forest
cf <- causal_forest(X, Y, W)

# Predict individual treatment effects
tau_hat <- predict(cf)$predictions

cat("🧭 1️⃣ Inspect and summarize")

summary(tau_hat)
hist(tau_hat, main = "Distribution of Estimated Treatment Effects (CATE)",
     xlab = "CATE (tau_hat)", col = "lightblue", border = "white")


# This tells you the range and shape of your individual treatment effects.
# You’ll usually see a bell-shaped or skewed spread: some people benefit more, some less.

cat("🧩 2️⃣ Merge with your features to study heterogeneity")

df <- data.frame(age = X[,1], 
                 education = X[,2], 
                 experience = X[,3],
                 W, Y, tau_hat)
head(df, 4)

cat("📊 3️⃣ Visualize how the effect varies across key covariates")

ggplot(df, aes(age, tau_hat)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "loess", se = TRUE, color = "blue") +
  labs(title = "Estimated Treatment Effect vs AGE",
       x = "Age", y = "Estimated CATE (tau_hat)")


ggplot(df, aes(experience, tau_hat)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "loess", se = TRUE, color = "blue") +
  labs(title = "Estimated Treatment Effect vs EXPERIENCE",
       x = "Age", y = "Estimated CATE (tau_hat)")

cat("📈 4️⃣ Quantify subgroup effects

Example: average treatment effects by age group")
    
library(dplyr)

df %>%
  mutate(age_group = cut(age, breaks = quantile(age, probs = seq(0,1,0.2)), include.lowest = TRUE)) %>%
  group_by(age_group) %>%
  summarise(mean_tau = mean(tau_hat), sd_tau = sd(tau_hat), n = n())


average_treatment_effect(cf, method = "AIPW")


test_calibration(cf)


colnames(X)

variable_importance(cf)

vi <- variable_importance(cf)
df_vi = data.frame(variable = colnames(X), importance = vi)
df_vi
cat("

These numbers measure how much each variable contributes to explaining heterogeneity in the treatment effect — 
that is, differences in how the treatment (e.g., job training) impacts wages across individuals.

✅ Interpretation:

AGE (0.67):
This is by far the most important variable.
→ The effect of the job training program (the causal variable) varies strongly by age.
In other words, the model found that age is the main moderator of how much benefit people get from the training.

EDUCATION (0.16):
Some contribution — education slightly influences treatment effect heterogeneity.
→ Perhaps people with higher education see somewhat different gains, but not nearly as much as variation by age.

EXPERIENCE (0.17):
Similar, small contribution — moderate heterogeneity, possibly younger vs older workers with similar experience respond differently.

🔍 In plain English

The causal forest discovered that age is the key factor determining who benefits most from the job training program.

While education and experience have minor roles, they don’t strongly change the size of the causal effect.

")

ggplot(df_vi, aes(x = reorder(variable, importance), y = importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Variable Importance for Treatment Effect Heterogeneity",
       x = "Variable", y = "Importance")



