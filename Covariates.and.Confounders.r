print("COVARIATES.CONFOUNDERS")

print("Covariates vs. Confounders — The Difference")

cat("
| Concept         | Meaning                                                                                                                                                                                      | Example                                                                       | Role in Causal Inference                                           |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Covariates**  | Any observed variables that are *related to the outcome, the treatment, or both*. They are general “control variables” that you might include in a model to improve estimation.              | Age, sex, baseline health score, etc.                                         | Used to improve precision or adjust for imbalances between groups. |

| **Confounders** | A *special subset* of covariates that affect **both** the treatment (or exposure) and the outcome. These create **spurious associations** between treatment and outcome if not adjusted for. | Smoking affects both exercise habits (treatment) and heart disease (outcome). | Must be controlled for to estimate a **causal effect** correctly.  |
")

print("CONFOUNDERS")

cat("Confounders make it seem like there’s a treatment effect even if there isn’t — or hide a real effect that exists.")

cat("Here are the main approaches used in causal inference to deal with confounders:")

cat("How to Handle Confounders (Adjustment Methods)")

cat("
| Approach                               | Intuition                                                                                             | Typical Tools                                       |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------- |

| **Stratification / Blocking**          | Analyze within levels of the confounder (e.g., within each age group).  
                                         | `dplyr` group_by(), or in theory, stratified means. |

| **Regression Adjustment**              | Include confounders as covariates in a regression model:  [ Y = β₀ + β₁T + β₂C + ε ]                  
                                         | `lm()`, `glm()`, `statsmodels`.                     |

| **Matching**                           | Match treated and untreated units with similar confounder profiles 
                                           (e.g., age, sex, baseline health). 
                                         | Propensity score matching (`MatchIt` in R).         |

| **Weighting (IPW)**                    | Weight observations by the inverse of their probability of receiving the treatment 
                                           given confounders.
                                         | `causalweight`, `twang`, `survey` packages.         |

| **Instrumental Variables (IV)**        | Use a variable that affects treatment but not the outcome except through the treatment.               
                                         | `AER::ivreg()` in R.                                |

| **Causal Graph / DAG-based selection** | Use Directed Acyclic Graphs to identify minimal adjustment sets.                                      
                                         | `dagitty` package.                                  |
")




cat("How to Handle Covariates (Non-confounding)")

cat(" 

For covariates that aren’t confounders, adjustment is optional:

If they improve precision (reduce variance of estimates), include them.

If they introduce bias (e.g., mediators or colliders), don’t include them.

")


cat("

Example:

Mediators: variables on the causal path between treatment and outcome — adjusting for them blocks part of the causal effect.

Colliders: variables affected by both treatment and outcome — adjusting for them creates bias.

")

 cat(" Co-Variate types : 

| Variable Type         | Relation to T   | Relation to Y | Should Adjust?  | Why                          |
| --------------------- | --------------- | ------------- | --------------  | ---------------------------- |
| Confounder            | Yes             | Yes           | ✅ Yes          | Removes bias                 |
| Mediator              | Effect of T     | Causes Y      | ❌ No           | Blocks part of effect        |
| Collider              | Effect of T & Y | –             | ❌ No           | Induces spurious correlation |
| Covariate (unrelated) | Maybe           | Maybe         | Optional        | May reduce variance          |

")

cat(" Special Types : 

| **Type**                          | **Description**                            | **Use Case**                       |
| --------------------------------- | ------------------------------------------ | ---------------------------------- |
| **Instrumental Variable (IV)**    | Affects treatment but not outcome directly | Random assignment in RCT           |
| **Proxy Variable**                | Stand-in for an unmeasured factor          | Income → education (proxy for SES) |
| **Latent Variable**               | Not directly observed                      | Intelligence, disease severity     |
| **Derived / Engineered Variable** | Created from other variables               | BMI = weight / height²             |

")



# ============================================================================
# CONFOUNDER EXAMPLE (Claude ai)
# ============================================================================


library(dagitty)
library(ggdag)
library(ggplot2)  # <- Add this!

# ============================================================================
# CONFOUNDER EXAMPLE WITH LABELED DAG
# ============================================================================

# Define the DAG with labels
dag_confounder <- dagify(
  Y ~ X + Z,
  X ~ Z,
  labels = c(X = "Exercise", Y = "Heart Health", Z = "Age"),
  exposure = "X",
  outcome = "Y"
)

# METHOD 1: Basic visualization with labels (SIMPLEST)
ggdag(dag_confounder, text = FALSE, use_labels = "label") + 
  theme_dag()

# METHOD 2: With status colors (exposure/outcome highlighted)
# ggdag_status(dag_confounder, text = FALSE, use_labels = "label") +
#  guides(color = "none") +  # Now this works!
#  theme_dag() +
#  labs(title = "Confounder: Age affects both Exercise and Heart Health")

# METHOD 3: Manual layout for better positioning
# dag_confounder_coords <- dagify(
#  Y ~ X + Z,
#  X ~ Z,
#  labels = c(X = "Exercise", Y = "Heart Health", Z = "Age"),
#  exposure = "X",
#  outcome = "Y",
#  coords = list(
#    x = c(X = 1, Y = 3, Z = 2),
#    y = c(X = 1, Y = 1, Z = 2)
#  )
# )

# ggdag(dag_confounder_coords, text = FALSE, use_labels = "label") +
#  theme_dag() +
#  labs(title = "Confounder Structure")

# METHOD 4: Show adjustment set highlighted
# ggdag_adjustment_set(dag_confounder, 
#                     text = FALSE, 
#                     use_labels = "label",
#                     shadow = TRUE) +
#  theme_dag() +
#  labs(title = "Must Control for Age (shown in color)")

# Identify adjustment sets
cat("\n=== ADJUSTMENT SETS ===\n")
print(adjustmentSets(dag_confounder))

# ============================================================================
# Simulate data showing confounding
# ============================================================================

set.seed(123)
n <- 1000

age <- rnorm(n, mean = 50, sd = 15)
exercise <- 100 - 0.8*age + rnorm(n, sd = 10)
heart_health <- 80 - 0.5*age + 0.3*exercise + rnorm(n, sd = 5)

data_conf <- data.frame(age, exercise, heart_health)
head(data_conf, 3)
tail(data_conf, 3)


cat("

| **Component**    | **Meaning**                                                    | **Use**                                                                          |
| ---------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `$coefficients`  | Estimated regression coefficients (β̂).                         | These are your estimated effects — 
                                                                                            use `coef(model_biased)`.         |
| `$residuals`     | Difference between observed and predicted values (`Y - Ŷ`).    | Used to check model fit (via plots, variance, etc).                              |
| `$fitted.values` | The predicted values (`Ŷ = Xβ̂`).                               | Use `fitted(model_biased)` or `predict()` for them.                              |
| `$rank`          | Number of independent columns in the model matrix X.           | Indicates how many parameters were actually estimated.                           |
| `$df.residual`   | Degrees of freedom for residuals (n − p).                      | Used for standard errors and p-values.                                           |
| `$effects`       | Transformed response vector from QR decomposition.             | Mostly used internally for computation.                                          |
| `$qr`            | QR decomposition of the design matrix X.                       | Internal matrix decomposition used by `lm()` to solve 
                                                                                      least squares efficiently. |
| `$xlevels`       | Levels of categorical predictors (factors).                    | Ensures predictions later use consistent factor levels.                          |
| `$call`          | The original function call used to fit the model.              | Shows exactly what formula and data were used.                                   |
| `$terms`         | The parsed formula (the “blueprint” of the model).             | Used by summary(), predict(), and model.matrix().                                |
| `$model`         | The actual data frame used in the fit (response + predictors). | You can inspect it with `model_biased$model`.                                    |

")

# Biased estimate (not controlling for confounder)
cat("\n=== WITHOUT CONTROLLING FOR CONFOUNDER ===\n")
model_biased <- lm(heart_health ~ exercise, data = data_conf)
cat("Effect of exercise:", coef(model_biased)["exercise"], "\n")
cat("(BIASED because age confounds the relationship)\n\n")

# Correct estimate (controlling for confounder)
cat("=== CONTROLLING FOR CONFOUNDER ===\n")
model_correct <- lm(heart_health ~ exercise + age, data = data_conf)
cat("Effect of exercise:", coef(model_correct)["exercise"], "\n")
cat("(Correctly recovers the true causal effect: 0.3)\n")

cat("
| **Command**             | **Purpose**                                                    |
| ----------------------- | -------------------------------------------------------------- |
| `summary(model_biased)` | Shows coefficients, standard errors, R², p-values, etc.        |
| `coef(model_biased)`    | Extracts just the β̂ coefficients.                             |
| `resid(model_biased)`   | Extracts residuals.                                            |
| `fitted(model_biased)`  | Extracts predicted values.                                     |
| `plot(model_biased)`    | Produces diagnostic plots (residuals vs fitted, QQ plot, etc). |
| `anova(model_biased)`   | Performs an ANOVA test for model significance.                 |
")

summary(model_biased)
coef(model_biased)
# resid(model_biased)
# fitted(model_biased)
plot(model_biased)
anova(model_biased)

# The ANOVA shows that exercise explains about 68% of the variability in heart health.
# The relationship is highly statistically significant (p < 10⁻²⁴⁸).
# In practical terms, people who exercise more have much better heart health scores on average, with a very strong effect size.

cat("

| Tool                    | Checks                                 | Output Interpretation                          | Typical Issue It Reveals                         |
| ----------------------- | -------------------------------------- | ---------------------------------------------- | ------------------------------------------------ |
| **QQ Plot (Residuals)** | Normality assumption                   | Straight line → good; curve → skew/heavy tails | 
                            Non-normal residuals, outliers                   |
| **ANOVA(model_biased)** | Overall model or variable significance | Small p-value → model/predictor is significant | 
                            Uninformative predictors, model misspecification |

")

summary(model_correct)
coef(model_correct)
# resid(model_correct)
# fitted(model_correct)
plot(model_correct)
anova(model_correct)

# So, ~84% of the variation in heart health is explained by exercise and age together.



cat("
| Mediator              | Effect of T     | Causes Y      | ❌ No           | Blocks part of effect        |
| Collider              | Effect of T & Y | –             | ❌ No           | Induces spurious correlation |
| Covariate (unrelated) | Maybe           | Maybe         | Optional        | May reduce variance          |
")



cat("
| Type                      | Example Variable | Relationship          | Should Adjust? | Why                          |
| ------------------------- | ---------------- | --------------------- | -------------- | ---------------------------- |
| **Mediator**              | Body fat %       | Effect of T, Causes Y | ❌ No           | Blocks part of causal effect |
| **Collider**              | Doctor visits    | Effect of T & Y       | ❌ No           | Induces spurious correlation |
| **Covariate (Unrelated)** | Height           | Maybe related to Y    | Optional       | Improves precision only      |
")

cat("

1️⃣ Mediator

Definition:

A variable that lies on the causal path between treatment (T) and outcome (Y).
Adjusting for it blocks part of the real causal effect.

| Role              | Example             |
| ----------------- | ------------------- |
| **Treatment (T)** | Exercise frequency  |
| **Mediator (M)**  | Body fat percentage |
| **Outcome (Y)**   | Blood pressure      |


💡 Path: Exercise → Body Fat → Blood Pressure
If you adjust for Body Fat, you remove part of the true effect of Exercise on Blood Pressure, 
because body fat transmits part of that effect.

Interpretation:

Exercise lowers body fat, which improves blood pressure.

If you control for body fat, you’re essentially asking: “Among people with the same body fat, does exercise still affect BP?” — 
that blocks the indirect (mediated) effect.

")

cat("

⚡ 2️⃣ Collider

Definition:
A variable that is caused by both the treatment and the outcome.
Adjusting for it creates spurious associations (bias).

| Role              | Example                |
| ----------------- | ---------------------- |
| **Treatment (T)** | Exercise               |
| **Outcome (Y)**   | Heart health           |
| **Collider (K)**  | Doctor visits per year |


💡 Path: Exercise → Doctor Visits ← Heart Health
Both exercise and heart health influence how often you see a doctor (collider).

What happens if you adjust for it?

You artificially induce a correlation between exercise and heart health even if one didn’t exist — 
because conditioning on the collider “opens” a non-causal path.

")

cat("

⚙️ 3️⃣ Covariate (Unrelated / Precision Variable)

Definition:
A variable correlated with Y or T but not on the causal path and not a confounder.
You can adjust for it optionally to reduce noise.

| Role              | Example        |
| ----------------- | -------------- |
| **Treatment (T)** | Exercise       |
| **Covariate (C)** | Height         |
| **Outcome (Y)**   | Blood pressure |

💡 Explanation:
Height may correlate weakly with blood pressure, but it does not cause exercise habits or sit on the causal pathway.
Including it as a covariate may slightly improve model precision, but it doesn’t change the causal interpretation.

")




cat("

Linear regression is the most classical way to adjust for confounders, 
but it’s not the only one — and sometimes it’s not the best, 
especially when relationships are nonlinear or the treatment assignment mechanism is complex.

Let’s go through the main families of methods for confounder control beyond linear regression, 
organized by conceptual approach 👇

")



cat(" How to correct for con-founders beside linear regression ?")

cat("

🧱 1️⃣ Matching Methods

🔹 Idea:

Compare treated and untreated individuals who have similar confounder profiles (e.g., same age, sex, health score).
This emulates a randomized experiment by balancing confounders.

| Method                              | Description                                                                                                      | R / Python Implementation |                                              |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------- | -------------------------------------------- |
| **Exact Matching**                  | Match subjects with identical confounder values. Works well only for categorical 
                                        or low-dimensional confounders. | `MatchIt` (R)             |                                              |
| **Propensity Score Matching (PSM)** | Match based on the probability of receiving treatment given confounders:  
                                        ( e(X) = P(T=1 | X) ). | `MatchIt`, `twang`, `causaltoolbox`, `DoWhy` |
| **Mahalanobis Matching**            | Match based on distance in confounder space.  
                                      | `optmatch`, `Matching`    |                                              |

")

cat("

⚖️ 2️⃣ Weighting (IPW / AIPW)

🔹 Inverse Probability Weighting (IPW):

Weights each observation by the inverse of the probability of receiving the treatment actually received.

🔹 Augmented IPW (AIPW):

Combines weighting + outcome regression → double robust: if either model is correct, the estimate is unbiased.

")

cat("

3️⃣ Causal Trees and Forests

🔹 Idea:

Use machine learning to estimate treatment effects while adjusting for confounders automatically.
They partition the covariate space into subgroups with similar causal effects.

🔹 Main Tools:

Method	Description	Packages

Causal Trees / Causal Forests	Estimate heterogeneous treatment effects (HTEs) adjusting for confounders.	grf (R), econml (Python)
Double Machine Learning (DML)	Orthogonalizes treatment and outcome models to remove confounding bias.	DoubleML, econml.dml

")


cat("

🔬 4️⃣ Instrumental Variables (IV)
🔹 Idea:

Find a variable (instrument) that affects treatment but not the outcome except through the treatment.
")

cat("

🧩 5️⃣ Propensity Score Stratification / Blocking
🔹 Idea:

Divide data into bins (strata) by propensity score and compare outcomes within each stratum.
Balances confounders across treatment groups.

")



cat('

1️⃣ What "blocking" means in causal inference

Blocking means:

Adjusting for (or conditioning on) a variable so that a non-causal path between two variables is closed — preventing spurious associations.

In other words:

When a confounder opens an unwanted “backdoor path” between treatment (T) and outcome (Y),

You “block” that path by conditioning on (adjusting for) the confounder.

3️⃣ What does “adjusting” mean in practice?

You can block a path by:

Including the variable in your regression model

lm(Y ~ T + C, data = df)

Matching or weighting on that variable
(so treated and control groups have similar C values)

Stratifying your analysis within levels of C

5️⃣ When blocking helps — and when it hurts

| Case           | Example                                        | Should You Block? | Why                            |
| -------------- | ---------------------------------------------- | ----------------- | ------------------------------ |
| **Confounder** | Smoking → Exercise and Smoking → Heart disease | ✅ Yes             | Blocks backdoor path           |
| **Mediator**   | Exercise → Body fat → Heart disease            | ❌ No              | Blocks part of the true effect |
| **Collider**   | Exercise → Doctor visits ← Heart disease       | ❌ No              | Opens a spurious path          |


')



cat(" BLOCKING : 🎯 Example: The “Coffee and Heart Disease” Paradox ")

cat("Scenario

Suppose we observe that:

People who drink more coffee seem to have higher rates of heart disease.

But… we also know that smoking is more common among heavy coffee drinkers.
Smoking itself increases heart disease risk.

So the relationship looks like this 👇

Coffee  ←  Smoking  →  Heart Disease
")

cat("

🧠 Step 1: Identify the confounder

Treatment (T): Coffee consumption

Outcome (Y): Heart disease

Confounder (C): Smoking (affects both coffee and heart disease)

Here, smoking opens a backdoor path:

T←C→Y

That means coffee and heart disease are associated even if coffee has no direct effect on heart disease.
")

cat("
🚪 Step 2: “Block” the backdoor path

If we condition on or adjust for smoking — for example:

lm(heart_disease ~ coffee + smoking, data = data)


we block that non-causal path.

The DAG now effectively becomes:

Coffee  →  Heart Disease
↑ (smoking blocked)

Now the only open path between coffee and heart disease is the direct causal one (if it exists
")

cat("

💡 Step 3: See what happens numerically

GroupAvg. Coffee Smoking Rate Heart Disease Rate 
Smokers 
         4 cups/day 100%  20% 
Non-smokers 
        1 cup/day 0% 5%

Naïve analysis (no blocking):
High coffee drinkers (smokers) → 20% disease
Low coffee drinkers (non-smokers) → 5% disease
➡️ It looks like coffee causes heart disease.

After blocking for smoking:
Among smokers: disease rate 20% regardless of coffee.
Among non-smokers: disease rate 5% regardless of coffee.
➡️ The apparent “effect” disappears.

✅ By blocking on smoking, we reveal that coffee itself isn’t the culprit — smoking was the confounder driving the association.
")

set.seed(73)

# === Simulate data =====================================================

n <- 1000

# Smoking status (1 = smoker, 0 = non-smoker)
smoking <- rbinom(n, 1, 0.5)

# Coffee consumption depends on smoking (smokers drink more coffee)
coffee <- ifelse(smoking == 1,
                 rnorm(n, mean = 4, sd = 1),   # smokers: ~4 cups/day
                 rnorm(n, mean = 1, sd = 0.5)) # non-smokers: ~1 cup/day

# Heart disease probability depends on smoking but *not* coffee
p_heart <- ifelse(smoking == 1, 0.20, 0.05)
heart_disease <- rbinom(n, 1, p_heart)

data <- data.frame(smoking, coffee, heart_disease)
head(data, 5)
tail(head, 5)

# === Naïve analysis: ignoring the confounder ===========================
cat("\n--- Naïve model: Heart disease ~ Coffee ---\n")
model_naive <- glm(heart_disease ~ coffee, data = data, family = binomial)
summary(model_naive)

# === Correct (blocked) analysis: controlling for smoking ===============
cat("\n--- Blocked model: Heart disease ~ Coffee + Smoking ---\n")
model_blocked <- glm(heart_disease ~ coffee + smoking, data = data, family = binomial)
summary(model_blocked)

# === Group summaries ===================================================
library(dplyr)
cat("\n--- Group summaries ---\n")
data %>%
  group_by(smoking) %>%
  summarise(mean_coffee = mean(coffee),
            heart_disease_rate = mean(heart_disease)) %>%
  mutate(smoking = ifelse(smoking == 1, "Smokers", "Non-smokers")) %>%
  print()

# === Interpretation message ============================================
cat("
💡 Interpretation:
Without adjusting for smoking, coffee appears to increase heart disease risk
because smokers both drink more coffee and have higher disease rates.
After blocking (adjusting) for smoking, the coffee effect disappears —
smoking was the confounder.
")




cat("AIC :

AIC stands for Akaike Information Criterion, and it’s one of the most widely used metrics to compare statistical models 
— especially in regression, GLMs, and mixed models.

📊 1️⃣ Definition

The Akaike Information Criterion (AIC) measures the trade-off between model fit and complexity.

AIC = 2k − 2ln(L)

where:


k = number of parameters in the model (complexity)
L = maximum likelihood of the model (goodness of fit)

where:

⚖️ 2️⃣ Intuition

Good fit → higher likelihood (larger 𝐿), which lowers AIC.

Too complex model → more parameters (higher k), which increases AIC.

So AIC penalizes overfitting:

You want a model that fits the data well without being too complex.

✅ Smaller AIC = better model (among models fitted to the same data).

🧠 3️⃣ How it’s used

You typically compare multiple models.

| ΔAIC (difference from best model) | Interpretation                                            |
| --------------------------------- | --------------------------------------------------------- |
| 0–2                               | Models have **substantial support** (roughly equivalent). |
| 4–7                               | Considerably less support.                                |
| >10                               | Very little support; model is likely poor.                |


")

fit1 <- glm(heart_disease ~ coffee, data = data, family = binomial)
fit2 <- glm(heart_disease ~ coffee + smoking, data = data, family = binomial)

AIC(fit1, fit2)




cat("BIC : Bayesian Information Criterion (also known as the Schwarz criterion")

cat("

📘 1️⃣ Definition

The Bayesian Information Criterion (BIC) is another way to measure the trade-off between model fit and model complexity, 
similar to AIC — but with a stronger penalty for models that have more parameters.

BIC = ln(n)k − 2ln(L)

where:

n = number of observations (sample size)
k = number of estimated parameters in the model

𝐿 = maximum likelihood of the model

⚖️ 2️⃣ The difference between AIC and BIC

Criterion	Formula	Penalty for Complexity	Interpretation

| Criterion | Formula              | Penalty for Complexity  | Interpretation                       |
| --------- | ---------------------| ----------------------- | ------------------------------------ |
| **AIC**   |  2k - 2 ln( L)       | **2 per parameter**     | Predictive accuracy                  |
| **BIC**   |  ln(n)k - 2 ln(L)    | **ln(n)** per parameter | Penalizes complexity more as n grows |

✅ When your sample size n is large, ln(n)>2, so BIC penalizes extra parameters more harshly than AIC.

3️⃣ Intuition

Both AIC and BIC reward models that fit data well (higher likelihood → lower AIC/BIC).

Both penalize models with many parameters.

BIC is more conservative: it prefers simpler models when two fit similarly.

🧮 4️⃣ How to interpret

Smaller BIC = better model (among those compared on the same dataset).

BIC can be interpreted as an approximation to the Bayes factor — it tells you how likely a model is relative to others, given the data.

| ΔBIC | Interpretation                                |
| ---- | --------------------------------------------- |
| 0–2  | Weak evidence (models similar)                |
| 2–6  | Positive evidence against higher BIC model    |
| 6–10 | Strong evidence                               |
| >10  | Very strong evidence against higher BIC model |

")

fit1 <- glm(heart_disease ~ coffee, data = data, family = binomial)
fit2 <- glm(heart_disease ~ coffee + smoking, data = data, family = binomial)

AIC(fit1, fit2)
BIC(fit1, fit2)


cat("

| Model    | Meaning                                 | AIC   | BIC   |
| -------- | --------------------------------------- | ----- | ----- |
| **fit1** | Unadjusted (simpler) model              | 668.1 | 677.9 |
| **fit2** | Adjusted for confounders (more complex) | 667.8 | 682.5 |

So:

fit2 adjusts for confounders (e.g., adds smoking, age, etc.).

AIC says → “fit2 is slightly better.”

BIC says → “fit2 is too complex, stick with fit1.”

⚖️ Why this happens

AIC is designed for predictive accuracy:
It asks, “Which model predicts unseen data better?”
→ It’s fine adding more variables if they improve predictive power even a little.

BIC is designed for model parsimony / true model identification:
It asks, “Which model is more likely to be the true underlying data-generating process?”
→ It penalizes complexity heavily (especially as sample size grows).

So if fit2 adds confounders that only slightly change likelihood,
BIC may reject them because they increase complexity more than they improve fit.

🧠 But here’s the key insight

👉 In causal inference, you don’t use AIC/BIC to decide whether to adjust for confounders.

Because:

Confounders are not just “extra predictors.”

They are necessary controls to get the causal effect right — even if they don’t improve prediction much.

So, if your goal is causal estimation, not just prediction,
then you must include the confounders — regardless of BIC.

🧩 Analogy

Think of BIC as a judge of “parsimony” and AIC as a judge of “prediction.”
But causal inference has a different court entirely — it judges bias vs. variance, not model simplicity.

Even if a confounder adds little predictive gain (hurts BIC),
failing to adjust for it leaves your causal estimate biased — a much bigger problem.

| Concept                    | Explanation                                                          |
| -------------------------- | -------------------------------------------------------------------- |
| **AIC**                    | Optimizes predictive accuracy (out-of-sample).                       |
| **BIC**                    | Optimizes model simplicity (true model identification).              |
| **Causal model selection** | Adjust for *all confounders*, regardless of AIC/BIC.                 |
| **Why?**                   | Omitting confounders biases the estimated effect — not just the fit. |

💡 Takeaway

In causal inference, bias reduction > parsimony.

So even if BIC says your adjusted model is “too complex,” you keep it —
because it’s causally correct, not just statistically efficient.

")

BIC(fit1, fit2)

cat("ANCOVA and Multi-linear regression :
    
ANCOVA and multiple (multilinear) regression, mathematically, they look almost identical, 
but conceptually they serve different purposes and are used in different contexts.

🎯 1️⃣ The short answer

Multiple Linear Regression (MLR)
→ A general predictive model for any continuous outcome using multiple predictors (numeric or categorical).
Goal: describe or predict the outcome.

ANCOVA (Analysis of Covariance)
→ A specific type of regression used to test whether group means differ after adjusting for covariates.
Goal: estimate adjusted group differences (often interpreted causally).

So, ANCOVA = Linear Regression + group comparison focus.

🧮 2️⃣ Model equations

Multiple linear regression (general form): Predictors can be continuous, categorical, or interaction terms.

ANCOVA (special case of regression): Y=α+τT+βX+ϵ

T: categorical factor (treatment, group, or experimental condition)
X: continuous covariate (e.g. baseline, age)
τ: adjusted mean difference between groups, controlling for 

Thus, ANCOVA is a specific regression model with:

at least one CATEGORICAL predictor (the “group”), and

at least one CONTINUOUS covariate (“covariance adjustment”).

")

cat("

⚖️ 3️⃣ Conceptual difference

| Aspect               | Multiple Linear Regression                  | ANCOVA                                                       |
| -------------------- | ------------------------------------------- | ------------------------------------------------------------ |
| **Purpose**          | Prediction or explanation                   | Compare group means controlling for covariates               |
| **Predictors**       | Any mix of continuous/categorical           | At least one categorical (group) + ≥1 continuous covariate   |
| **Interpretation**   | How predictors *relate* to outcome          | Whether group differences *remain* after adjusting           |
| **Primary interest** | Coefficients of all predictors              | Group (treatment) effect (adjusted mean difference)          |
| **Common in**        | Machine learning, econometrics, forecasting | Experimental, clinical, psychological, or biological studies |

🧠 4️⃣ Think of ANCOVA as “Regression for ANOVA users”

ANOVA compares group means.
Regression models continuous relationships.
ANCOVA merges them:

ANCOVA asks: “After adjusting for this covariate, are the adjusted group means still significantly different?”

Mathematically, it’s identical to a linear model — but the interpretation is as an adjusted ANOVA.

Multiple Linear Regression : lm(Y ~ age + weight + income, data = df) [Predict Y from several continuous variables.]

ANCOVA : lm(Y ~ treatment + baseline_score, data = df)

Here, treatment is CATEGORICAL (e.g., control/treatment), and baseline_score is CONTINUOUS.
The coefficient for treatment = adjusted treatment effect.

✅ 8️⃣ Bottom line

Mathematically: identical — ANCOVA is a linear regression with categorical and continuous predictors.

Conceptually: different emphasis.

Regression → “How does Y change with Xs?”

ANCOVA → “Are there group differences in Y after controlling for X?

")

cat("ANCOVA example")

set.seed(123)

n <- 100
# Treatment assignment
group <- factor(rep(c("Control", "Treatment"), each = n/2))

# Baseline blood pressure (covariate)
baseline <- rnorm(n, mean = 130, sd = 10)

# True effect: treatment reduces BP by 8, plus dependence on baseline
true_effect <- -8
final_bp <- 0.6 * baseline + ifelse(group == "Treatment", true_effect, 0) + rnorm(n, 0, 5)

data <- data.frame(group, baseline, final_bp)
head(data)
tail(data)

model <- lm(final_bp ~ group + baseline, data = data)
summary(model)

anova_model <- lm(final_bp ~ group, data = data)
summary(anova_model)


cat("

✅ Interpretation in plain language:

After adjusting for each participant’s baseline blood pressure, 
those in the treatment group had on average 6.5 points lower final blood pressure 
than those in the control group (p < 0.000000001).

Baseline BP also predicts final BP strongly (p < 0.0000000000000002).
Overall, the ANCOVA model explains about 61% of the variation in final blood pressure.

")




