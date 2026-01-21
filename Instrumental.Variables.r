cat("INSTRUMENTAL VARIABLES")

cat("

🎯 What Are Instrumental Variables?

Instrumental Variables (IV) are a causal inference method used when the treatment is not randomly assigned and 
there is unmeasured confounding you cannot control for.

🔥 The Core Idea

You find a variable Z (the instrument) that:

1. Affects the treatment (Z → W)

The instrument must change whether someone receives the treatment.

2. Does NOT directly affect the outcome (Z ↛ Y)

It should affect the outcome only through the treatment.

3. Is NOT related to confounders

Z must be as good as random with respect to unmeasured confounders.

📦 Why Use IV?

When we cannot rely on:

matching

regression

causal forests

uplift modeling

propensity scores

because confounders are unobserved.

→ IV is a way to “recover randomization.”

🧠 Intuition (the elevator pitch)

The Instrument works like a natural experiment.

If something (Z) pushes people into treatment in a random-like way,

but does not affect the outcome directly,

we can use it to identify the causal effect.

")

cat("

✔️ Classic Example (Interview Ready)

🎵 Example: Effect of attending music lessons (W) on cognitive performance (Y)

Confounder:

Kids from wealthy families are more likely to take lessons AND have higher test scores.

You cannot measure family motivation, home environment, parenting skill, etc.

Use an Instrument:

Distance to nearest music school (Z)

Kids closer to a music school are more likely to attend (Z → W)

Distance does not directly improve math scores (Z ↛ Y)

Distance is unrelated to family motivation (ideally)

→ Valid IV.

")

cat("

🧩 How IV works mathematically

Two-Stage Least Squares (2SLS)

Stage 1: Predict treatment using the instrument

W = α + βZ + γX + ϵ

This gives the part of 𝑊^

Stage 2: Predict outcome using predicted treatment
𝑌 = 𝛿 + 𝜃 𝑊^ + 𝜂

The coefficient 𝜃 : θ = causal effect.

")

cat("

library(AER)

# Example data: ivreg requires formula Y ~ W | Z

iv_model <- ivreg(Y ~ W | Z, data = df)

summary(iv_model)

")


cat("

The Two-Stage Least Squares (2SLS) method

")

cat("Examples :

| Setting                            | Treatment (T)              | Outcome (Y)    | Instrument (Z)                          |
| ---------------------------------- | -------------------------- | -------------- | --------------------------------------- |
| Education economics                | Years of education         | Income         | Distance to nearest college             |
| Health                             | Taking a drug              | Blood pressure | Doctor’s random prescription preference |
| Labor economics                    | Job training participation | Wage           | Random assignment or eligibility rule   |
| Genetics (Mendelian Randomization) | BMI                        | Heart disease  | Genetic variant (allele) affecting BMI  |

")

set.seed(123)
n <- 1000

U <- rnorm(n)
Z <- rnorm(n)                   # instrument
T <- 0.8*Z + 0.7*U + rnorm(n)   # treatment depends on Z (relevant) and U (confounded)
Y <- 5 + 2*T + 3*U + rnorm(n)   # outcome depends on T and U

head(U)
head(Z)
head(T)
head(Y)

# Ordinary regression (biased, because Cov(T,U)!=0)
ols <- lm(Y ~ T)
summary(ols)$coef

# Instrumental variable regression
library(AER)
iv <- ivreg(Y ~ T | Z)
summary(iv)

cat("\n :

| Term            | Estimate | Meaning                                                                                                                            | p-value | Significance           |
| --------------- | -------- | ----------------------------------------------------- |
| **(Intercept)** | 5.03     | When ( T = 0 ), the expected value of ( Y ) ≈ 5.0   
                             | < 2e–16 | *** highly significant |
| **T**           | **2.28** | A **1-unit increase in ( T )** causes, on average, 
                               a **2.28-unit increase in ( Y )**, 
                               after accounting for confounding using ( Z ) | < 2e–16 | 
                               *** highly significant |

| Metric                               | Meaning                                                                          | Value                      
| ------------------------------------ | -------------------------------------------------------------------------------- | --------------------------
| **Residual Std. Error = 2.96**       | Typical deviation of observed vs. predicted ( Y )                                | Moderate fit               
| **R² = 0.6846**                      | 68% of the variation in ( Y ) explained by fitted model                          | Strong explanatory power   
| **Wald test (F = 474.9, p < 2e–16)** | Tests whether the instrumented regressor ( T ) has a significant effect on ( Y ) | Highly significant overall 

✅ Summary in plain language :

After removing confounding effects using the instrument Z, 

we estimate that each 1-unit increase in the treatment T 
leads to an average increase of 2.28 units in the outcome 𝑌

This effect is statistically significant (p < 0.000001), 
and the model explains about 68% of the variation in 𝑌

")


cat('

2SLS: Two-Stage Least Squares

2SLS is an extension of OLS designed to fix endogeneity 

— using an instrumental variable (Z) that creates “as-if random” variation in T.


| Concept               | OLS                              | 2SLS                                        |
| --------------------- | -------------------------------- | ------------------------------------------- |
| Assumes exogeneity    | ✅ Yes                            | ✅ Yes, but via instrument                   |
| Handles confounding   | ❌ No                             | ✅ Yes (if valid instrument)                 |
| Accuracy              | Efficient if valid               | Less efficient but consistent               |
| Causal interpretation | Only if no omitted variable bias | Robust to omitted variable bias             |
| Analogy               | “Direct regression”              | “Regression using only exogenous variation” |

')


cat("PARTIAL LEAST SQUARES :")

cat(" 

A note about PLS : 

Partial Least Squares (PLS), which sits between Ordinary Least Squares (OLS) regression and 
Principal Component Analysis (PCA).

PLS is especially powerful when you have many correlated predictors — 

typical in genomics, chemometrics, spectroscopy, 

and other high-dimensional bioinformatics settings.

🧠 The idea behind Partial Least Squares

PLS finds new latent variables (components) — linear combinations of the predictors — that:

capture directions of high variance in X and are most correlated with 𝑌

So unlike PCA (which only looks at variance in  𝑋, PLS looks for components that best explain 𝑌

In other words:

PCA: summarize 𝑋

PLS: summarize 𝑋 to predict Y.

| Feature                         | OLS               | PCA                      | PLS                                            |
| ------------------------------- | ----------------- | ------------------------ | ---------------------------------------------- |
| Goal                            | Fit (Y = X\beta)  | Summarize variance in X  | Predict Y from X                               |
| Handles multicollinearity?      | ❌ No              | ✅ Yes                    | ✅ Yes                                          |
| Uses Y in component extraction? | ✅ Directly        | ❌ No                     | ✅ Yes                                          |
| Good for high-dimensional data? | ❌ No              | ✅                        | ✅                                              |
| Typical application             | Simple regression | Dimensionality reduction | Predictive modeling with correlated predictors |


")

set.seed(123)
library(pls)

# Simulate data
n <- 100
p <- 20
X <- matrix(rnorm(n*p), n, p)
X[, 2:5] <- X[, 1] + rnorm(n*4, sd=0.1)  # make predictors correlated
Y <- 3*X[,1] - 2*X[,2] + rnorm(n)

head(X)
head(Y)

# Fit OLS (unstable)
ols <- lm(Y ~ X)

cat("OLS model")
summary(ols)$r.squared

# Fit PLS
pls_model <- plsr(Y ~ X, ncomp = 3, validation = "CV")

cat("PLS model")
summary(pls_model)

# Predicted R² for cross-validation
R2(pls_model, estimate = "CV")



cat("

| Situation                                   | Why PLS helps                                    |
| ------------------------------------------- | ------------------------------------------------ |
| Many predictors (p > n)                     | Dimensionality reduction built in                |
| Predictors highly correlated                | PLS extracts orthogonal latent variables         |
| You want both prediction and interpretation | PLS balances predictive power and explainability |
| You want supervised dimension reduction     | Unlike PCA, PLS uses Y to guide component choice |

🧠 Intuition summary

OLS: fits a direct line
PCA: summarizes input variance
PLS: finds directions in X that best predict Y.

Imagine you have 10,000 genes, many correlated —

PLS builds a few “synthetic” gene-expression features (latent variables) that best predict your phenotype Y.

")

cat("

| Concept                    | OLS                    | 2SLS                              | PLS                                      |
| -------------------------- | ---------------------- | --------------------------------- | ---------------------------------------- |
| Handles confounding        | ❌                      | ✅ (via instruments)               | ❌                                        |
| Handles multicollinearity  | ❌                      | ⚠️ sometimes                      | ✅                                        |
| Dimension reduction        | ❌                      | ❌                                 | ✅                                        |
| Uses latent components     | ❌                      | ❌                                 | ✅                                        |
| Uses Y to guide components | ✅ directly             | ✅ via stage 2                     | ✅ integrated                             |
| Best for                   | clean causal inference | causal inference with endogeneity | prediction with correlated or high-dim X |
")



cat(" INSTRUMENTAL VARIABLES : Scenario: Effect of Education on Wages ")

library(dplyr)
library(ggplot2)
library(AER)  # For ivreg function

set.seed(123)
n <- 5000

# ==================== 1. SIMULATE DATA ====================
# Unmeasured confounder: ability
data <- data.frame(
  ability = rnorm(n, mean = 100, sd = 15),  # Can't observe this!
  quarter_birth = sample(1:4, n, replace = TRUE)  # Instrument
)

# Education: affected by ability AND quarter of birth
# Quarter 1 births → slightly less education (can drop out earlier)
data$education <- with(data,
  12 +  # Base education
  0.08 * ability +  # Ability increases education
  -0.5 * (quarter_birth == 1) +  # Q1 births get less education
  rnorm(n, 0, 1.5)
)

# Wages: affected by education AND ability
# True causal effect of education: $2000 per year
true_effect <- 2000
data$wage <- with(data,
  20000 +  # Base wage
  true_effect * education +  # CAUSAL effect of education
  300 * ability +  # Ability also affects wages (confounding!)
  rnorm(n, 0, 3000)
)

# ==================== 2. NAIVE REGRESSION (BIASED) ====================
naive_model <- lm(wage ~ education, data = data)

cat("=== NAIVE OLS (Biased due to omitted ability) ===\n")
cat("Estimated effect:", round(coef(naive_model)["education"], 2), "\n")
cat("True effect:", true_effect, "\n")
cat("Bias:", round(coef(naive_model)["education"] - true_effect, 2), "\n\n")

# This is BIASED because education correlates with ability



# ==================== 3. ORACLE REGRESSION (If we could observe ability) ====================
oracle_model <- lm(wage ~ education + ability, data = data)

cat("=== ORACLE (Controls for ability - correct) ===\n")
cat("Estimated effect:", round(coef(oracle_model)["education"], 2), "\n")
cat("True effect:", true_effect, "\n\n")

# ==================== 4. CHECK INSTRUMENT VALIDITY ====================
# Requirement 1: Relevance - Does instrument predict treatment?
first_stage <- lm(education ~ quarter_birth, data = data)
summary(first_stage)

# F-statistic should be > 10 (rule of thumb)
f_stat <- summary(first_stage)$fstatistic[1]
cat("=== First Stage F-statistic ===\n")
cat("F-stat:", round(f_stat, 2), "\n")
if (f_stat > 10) {
  cat("✓ Strong instrument (F > 10)\n\n")
} else {
  cat("✗ Weak instrument (F < 10)\n\n")
}

# Visualize first stage
ggplot(data, aes(x = factor(quarter_birth), y = education)) +
  geom_boxplot(fill = "steelblue", alpha = 0.6) +
  labs(
    title = "First Stage: Instrument Relevance",
    subtitle = "Quarter of birth affects education",
    x = "Quarter of Birth",
    y = "Years of Education"
  ) +
  theme_minimal()

# ==================== 5. IV ESTIMATION (2SLS) ====================
# Two-Stage Least Squares
iv_model <- ivreg(wage ~ education | quarter_birth, data = data)

cat("=== IV Estimation (2SLS) ===\n")
cat("Estimated effect:", round(coef(iv_model)["education"], 2), "\n")
cat("True effect:", true_effect, "\n")
cat("Standard error:", round(summary(iv_model)$coefficients["education", "Std. Error"], 2), "\n\n")

# Compare all methods
comparison <- data.frame(
  Method = c("Naive OLS", "Oracle (with ability)", "IV (2SLS)", "True Effect"),
  Estimate = c(
    coef(naive_model)["education"],
    coef(oracle_model)["education"],
    coef(iv_model)["education"],
    true_effect
  ),
  SE = c(
    summary(naive_model)$coefficients["education", "Std. Error"],
    summary(oracle_model)$coefficients["education", "Std. Error"],
    summary(iv_model)$coefficients["education", "Std. Error"],
    NA
  )
)

print(comparison)


# ==================== 6. VISUALIZE ====================
# Create reduced form plot
reduced_form <- data %>%
  group_by(quarter_birth) %>%
  summarise(
    avg_wage = mean(wage),
    avg_education = mean(education),
    .groups = 'drop'
  )

# Plot 1: Reduced form (instrument → outcome)
p1 <- ggplot(reduced_form, aes(x = factor(quarter_birth), y = avg_wage)) +
  geom_col(fill = "darkgreen", alpha = 0.6) +
  labs(
    title = "Reduced Form: Quarter of Birth → Wages",
    x = "Quarter of Birth",
    y = "Average Wage ($)"
  ) +
  theme_minimal()

# Plot 2: IV estimate visualization
p2 <- ggplot(data, aes(x = education, y = wage)) +
  geom_point(alpha = 0.1) +
  geom_smooth(method = "lm", aes(color = "Naive OLS"), se = FALSE, size = 1.5) +
  geom_abline(intercept = coef(iv_model)[1], 
              slope = coef(iv_model)[2],
              color = "red", size = 1.5, linetype = "dashed") +
  annotate("text", x = 13, y = 80000, 
           label = paste("Naive:", round(coef(naive_model)["education"], 0)),
           color = "blue", size = 5) +
  annotate("text", x = 13, y = 75000,
           label = paste("IV:", round(coef(iv_model)["education"], 0)),
           color = "red", size = 5) +
  labs(
    title = "IV vs Naive OLS",
    x = "Education (years)",
    y = "Wage ($)",
    color = ""
  ) +
  theme_minimal()

print(p1)
print(p2)

cat("Manual 2SLS calculation : Claude AI")

# ==================== MANUAL 2SLS ====================
# Stage 1: Regress education on instrument
stage1 <- lm(education ~ quarter_birth, data = data)
data$education_predicted <- predict(stage1)

# Stage 2: Regress wage on predicted education
stage2 <- lm(wage ~ education_predicted, data = data)

cat("\n=== Manual 2SLS ===\n")
cat("Stage 1 coefficient:", round(coef(stage1)["quarter_birth"], 4), "\n")
cat("Stage 2 coefficient:", round(coef(stage2)["education_predicted"], 2), "\n")

# This matches ivreg() output
cat("\nCompare with ivreg:", round(coef(iv_model)["education"], 2), "\n")


