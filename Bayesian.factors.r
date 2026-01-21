cat("
🧠 1️⃣ What is a Bayes Factor?

A Bayes factor (BF) quantifies how much more likely the data are under one hypothesis than another.

BF₁₂ = P(Data | H₁) / P(Data | H₂)

Where:

BF₁₂ = Bayes factor favoring H₁ over H₂
P(Data | H₁) = Probability of observing the data if H₁ is true
P(Data | H₂) = Probability of observing the data if H₂ is true

Simple Example
Suppose you flip a coin 10 times and get 8 heads.

H₁: The coin is fair (p = 0.5)
H₂: The coin is biased (p = 0.7)

Calculate how likely '8 heads out of 10' is under each hypothesis:

")

# Probability under H1 (fair coin)
p_data_h1 <- dbinom(8, size = 10, prob = 0.5)  # = 0.044
p_data_h1

# Probability under H2 (biased coin)
p_data_h2 <- dbinom(8, size = 10, prob = 0.7)  # = 0.233
p_data_h2

# Bayes factor
BF <- p_data_h2 / p_data_h1  # = 5.3
print(BF)

cat("

The Binomial Distribution

Density, distribution function, quantile function and random generation 
for the binomial distribution with parameters size and prob. 

👉 A Bayes Factor of 10 means:

The data are 10× more likely under the alternative than under the null hypothesis.

A Bayes Factor of 0.1 means:

The data are 10× more likely under the null than under the alternative

🧩 3️⃣ Relationship to p-values

| Concept          | What it measures                                                     | Interpretation        |        |                                                 |
| ---------------- | -------------------------------------------------------------------- | --------------------- | ------ | ----------------------------------------------- |
| **p-value**      | Probability of seeing data *as extreme* as observed if (H_0) is true | Reject (H_0) if small |        |                                                 |
| **Bayes Factor** | Ratio of likelihoods ( P(D                                           | H_1)/P(D              | H_0) ) | How much more the data support (H_1) over (H_0) |


")

# ?dbinom

cat("

| Bayes Factor ( BF_{10} ) | Evidence for ( H_1 ) (Alternative)        |
| ------------------------ | ----------------------------------------- |
| **> 100**                | Decisive evidence                         |
| **30 – 100**             | Very strong evidence                      |
| **10 – 30**              | Strong evidence                           |
| **3 – 10**               | Moderate evidence                         |
| **1 – 3**                | Weak evidence                             |
| **= 1**                  | No preference (data equally support both) |
| **< 1/3**                | Moderate evidence for ( H_0 )             |
| **< 1/10**               | Strong evidence for ( H_0 )               |
| **< 1/30**               | Very strong evidence for ( H_0 )          |

👉 A Bayes Factor of 10 means:

The data are 10× more likely under the alternative than under the null hypothesis.

A Bayes Factor of 0.1 means:

The data are 10× more likely under the null than under the alternative

")




library(BayesFactor)

# Simulate data: test scores in control vs treatment groups
set.seed(123)
control <- rnorm(30, mean = 75, sd = 10)
treat   <- rnorm(30, mean = 80, sd = 10)   # treatment improves by +5 points

# === Classical t-test ===
t_test_result <- t.test(treat, control)
cat("\nt test result:\n")
t_test_result

bf_result <- ttestBF(x = treat, y = control)
cat("\nbayesian factors:\n")
bf_result


# ?ttestBF

boxplot(control, treat,
        names = c("Control", "Treatment"),
        col = c("lightblue", "lightgreen"),
        main = "Test Scores by Group")

cat("🧪 1️⃣ Classical t-test 

✅ Interpretation:

The t-statistic (3.08) shows a strong difference between treatment and control.

p-value = 0.003 → much smaller than 0.05 → statistically significant.
→ We reject the null hypothesis (no difference in means).

95% Confidence Interval = [2.54, 11.97] means the true difference in means is very likely between +2.5 and +12 points.

Means:

Treatment group mean = 81.78

Control group mean = 74.53
→ On average, the treatment group scored ≈7.25 points higher.

")

cat(" 🧮 2️⃣ Bayesian Analysis

✅ Interpretation:

The Bayes Factor (BF₁₀ = 12.03) means:

The observed data are 12 times more likely under the alternative hypothesis 
(that there is a true difference in means) than under the null hypothesis (no difference).

This corresponds to “strong evidence” for the alternative hypothesis (H₁).
")

cat("

| Bayes Factor (BF₁₀) | Interpretation            |
| ------------------- | ------------------------- |
| 1–3                 | Anecdotal evidence for H₁ |
| 3–10                | Moderate evidence         |
| **10–30**           | **Strong evidence**       |
| 30–100              | Very strong evidence      |
| >100                | Decisive evidence         |

| Test                           | Result                   | Evidence Strength    | Conclusion                            |
| ------------------------------ | ------------------------ | -------------------- | ------------------------------------- |
| **t-test (p = 0.003)**         | Significant difference   | Strong (frequentist) | Treatment effect likely real          |
| **Bayes Factor (BF₁₀ = 12.0)** | 12× more likely under H₁ | Strong (Bayesian)    | Data strongly support real difference |
| **Mean difference = +7.25**    | CI [2.54, 11.97]         | —                    | Treatment improves mean outcome       |

")

cat("

✅ 4️⃣ In plain English

Both the frequentist and Bayesian analyses agree:
The treatment group performed significantly better than the control group.

Classical test: p = 0.003 → statistically significant.

Bayesian test: BF = 12 → data are 12× more likely if the treatment really has an effect.

Therefore, there is strong, converging evidence that the treatment improves outcomes by roughly 7 points on average.

")



cat("LIKELIHOOD")

cat("

🧠 1️⃣ What is “Likelihood”?

Likelihood tells us how likely the observed data are, given a specific model or parameter value.

Formally:

L(θ∣data)=P(data∣θ)

θ = model parameter(s) (e.g., mean, variance, regression coefficients)

data = what we actually observed

⚖️ In words:

THE LIKELIHOOD IS NOT THE PROBABILITY OF THE PARAMETERS — IT’S THE PROBABILITY OF THE DATA, GIVEN THE PARAMETERS. ✅

🎯 2️⃣ Example: Tossing a coin

Suppose you toss a coin 10 times and get 7 heads.

You want to know how likely that is for different values of 

θ (the probability of heads).

The likelihood for a binomial model is [ FORMULA] : 

| θ (probability of heads) | Likelihood ( L(\theta) ) |
| ------------------------ | ------------------------ |
| 0.2                      | 0.00079                  |
| 0.5                      | 0.117                    |
| **0.7**                  | **0.267**                |
| 0.9                      | 0.012                    |

✅ The highest likelihood is at θ = 0.7
→ So the data are most consistent with a coin that lands heads 70% of the time.

🧮 3️⃣ Likelihood vs Probability

| Concept         | What varies                 | Meaning                                             |
| --------------- | --------------------------- | --------------------------------------------------- |
| **Probability** | Data vary, parameters fixed | “Given θ, how likely are these data?”               |
| **Likelihood**  | Parameters vary, data fixed | “Given these data, which θ makes them most likely?” |

So in inference, we treat the data as fixed and ask:

WHICH PARAMETER VALUE MAKES THE DATA MOST PLAUSIBLE? ✅

📊 4️⃣ Maximum Likelihood Estimation (MLE)

The MLE is the parameter value that maximizes the likelihood function. 

It is the cornerstone of frequentist estimation and the starting point for most machine learning models 
(logistic regression, neural nets, etc.).

So the likelihood is the bridge connecting data to posterior inference in Bayesian statistics.

")

cat("

🧪 Example: Coin toss likelihood

We toss a coin 10 times and get 7 heads.

We want to see for which value of 

θ (probability of heads) the data are most likely ")

# 1️⃣ Simulated experiment
n <- 10          # number of tosses
k <- 7           # number of heads observed

# 2️⃣ Likelihood function for Binomial model
theta <- seq(0, 1, length.out = 200)
likelihood <- dbinom(k, size = n, prob = theta)

head(theta)
head(likelihood)


# 3️⃣ Normalize (optional, for nicer plotting)
likelihood <- likelihood / max(likelihood)

# 4️⃣ Plot likelihood curve
plot(theta, likelihood, type = "l", lwd = 3, col = "blue",
     main = "Likelihood Function for 7 Heads in 10 Tosses",
     xlab = expression(theta),
     ylab = "Relative Likelihood")

# 5️⃣ Mark the Maximum Likelihood Estimate (MLE)
theta_hat <- theta[which.max(likelihood)]
abline(v = theta_hat, col = "red", lwd = 2, lty = 2)
text(theta_hat, 0.9, labels = paste("MLE =", round(theta_hat, 2)),
     pos = 4, col = "red")

cat("

| Concept                      | Meaning                                                                   |
| ---------------------------- | ------------------------------------------------------------------------- |
| **θ (probability of heads)** | Parameter we’re estimating                                                |
| **Likelihood curve**         | How plausible each θ value is given our data (7 heads / 10 tosses)        |
| **Peak (θ̂ = 0.7)**          | The data are most consistent with a coin that lands heads 70% of the time |
| **Likelihood width**         | Reflects uncertainty — flatter curve = more uncertainty                   |

Interpret :

The MLE = observed success rate (0.7)

")

cat("

| Concept                 | Meaning                                                            |
| ----------------------- | ------------------------------------------------------------------ |
| **Likelihood function** | Probability of data given parameters                               |
| **MLE**                 | Parameter values that maximize likelihood                          |
| **Each distribution**   | Has its own formula for the likelihood and its own closed-form MLE |
| **In ML**               | Training = maximizing (log) likelihood = minimizing loss           |

")

cat("

| Model               | Parameter(s)               | MLE finds…                                                               |
| ------------------- | -------------------------- | ------------------------------------------------------------------------ |
| Binomial            | θ = probability of success | θ that makes observed successes most likely                              |
| Normal              | μ, σ²                      | Mean and variance that best fit data                                     |
| Linear regression   | β coefficients             | Values that maximize likelihood (equivalent to minimizing squared error) |
| Logistic regression | β coefficients             | Coefficients that maximize likelihood of observed labels                 |

")








