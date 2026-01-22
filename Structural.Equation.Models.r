# descriptions provided by chatGPT, Grok, Claude, Gemini

cat("

Structural Equation Models (SEMs) are statistical models used to test complex, 
theory-driven relationships among variables—especially when some concepts 
are latent (unobserved) and measured indirectly. 

SEMs combine FACTOR ANALYSIS and REGRESSION to test complex relationships between observed and latent (unobserved) variables. 

Think of it as a way to test theories about how multiple variables influence each other simultaneously.

")

cat("

Two Types of Variables :

* Observed Variables (measured directly):

Gene expression levels
Blood pressure readings

* Latent Variables (not directly measured):

Health (general construct)
Hypoxia response (biological pathway)

")

cat("Simple Example :

Let's say you want to study how stress affects health:

Measurement Model :
├─ Stress (latent)
│  ├─ Cortisol levels (observed)
│  ├─ Heart rate (observed)
│  └─ Self-report anxiety (observed)
│
└─ Health (latent)
   ├─ Blood pressure (observed)
   ├─ Sleep quality (observed)
   └─ Immune markers (observed)

Visual representation :

[Observed Variables]          [Latent Variables]
     
     Cortisol ─────┐
     Heart Rate ───┤──→ STRESS ──→ HEALTH ─┬──→ Blood Pressure
     Anxiety ──────┘                        ├──→ Sleep Quality
                                            └──→ Immune Markers

")

cat("Simple Examples :

Where SEMs shine (examples) : 

Genetics → pathway activity → phenotype

Hypoxia → transcriptional program → disease severity

Microbiome → metabolites → immune activation

Clinical scores measured by multiple questionnaires

Latent variables often map naturally to pathways or gene programs.

")

cat("

Two parts of an SEM : PART 1

1) Measurement model (latent variables) : FACTOR ANALYSIS

Some concepts aren’t directly observed (e.g., inflammation, cell stress). 
Instead, we observe indicators (genes, proteins, scores).

Goal: Define how observed variables reflect latent constructs.

Tool: Confirmatory Factor Analysis (CFA).

Mathematically:

𝑋 = 𝛬 𝜂 + 𝜖

𝑋 : observed indicators

η: latent variables

Λ: loadings (strength of each indicator)

ϵ: measurement error

")

cat("

Two parts of an SEM : PART 2

2) Structural model (causal paths) : REGRESSION

Defines how variables influence each other (latent ↔ latent, observed → latent).

η = Bη + ΓX + ζ

B: relationships among latent variables

Γ: effects of observed variables

ζ: unexplained variance

This is where mediation (A → B → C) and total effects are estimated.

")

# How SEMs are visualized (path diagrams) : 

# Conventions
# Circles = latent variables
# Squares = observed variables
# Single arrows = directional effects
# Double arrows = correlations
# Error terms are explicit

cat("

What SEMs estimate :

Factor loadings (how indicators measure a latent construct)

Direct effects (A → C)

Indirect effects (A → B → C)

Total effects (direct + indirect)

Residual variances

Model fit (does the whole structure explain the data?)

")

cat("

Model fit (crucial)

SEMs evaluate whether your entire hypothesized structure matches observed covariances.

Common metrics:

χ² (Chi-square): Non-significant = good fit
CFI (Comparative Fit Index): ≥ 0.95 = good
RMSEA (Root Mean Square Error): ≤ 0.06 = good
SRMR (Standardized Root Mean Residual): ≤ 0.08 = good

Poor fit usually means the causal assumptions need revision.

")

cat("Simple example :

SNP ─────────┐
             │
             └──→ PATHWAY ACTIVATION ─────────→ Drug Response
                 (latent / unobserved)
                     ↑
Gene Expression ────┤ (measurements of the pathways, not cause of it)
Metabolites ────────┘ (measurements of the pathways, not cause of it)

Direct Effect: SNP ───────────────────────────→ Drug Response

✅ Measurement model

(How the latent pathway activation is measured)

GeneExpression = 𝜆1 ⋅ PathwayActivation + 𝜖1

Metabolites = 𝜆2 ⋅ PathwayActivation + 𝜖2

PathwayActivation is latent (unobserved)

Gene expression and metabolites are noisy indicators

𝜆𝑖 = factor loadings

𝜖𝑖 = measurement error

✅ Structural model

(Causal relationships between variables)

PathwayActivation = 𝛽1 ⋅ SNP + 𝜁1

DrugResponse = 𝛽2 ⋅PathwayActivation + 𝛽3⋅SNP + 𝜁2

DrugResponse as: a measured outcome (eg tumor size)

β1 : SNP → pathway activation
β2 : pathway → drug response (mediated effect)
𝛽3 : SNP → drug response (direct effect)
𝜁𝑖 : unexplained biological variation

Interpretation (important)

Indirect (mediated) effect of SNP on drug response: 𝛽1 × 𝛽2
Direct effect: 𝛽3
Total effect: 𝛽3 + (𝛽1 × 𝛽2)

SEM philosophy:

Latent variables cause measurements, not the other way around.
")

# Install if needed:
# install.packages("lavaan")

library(lavaan)

cat("Simple example:\n\n",
"SNP ─────────┐\n",
"             │\n",
"             └──→ PATHWAY ACTIVATION ─────────→ Drug Response\n",
"                 (latent / unobserved)\n",
"                     ↑\n",
"Gene Expression ────┤ (measurements of the pathway, not cause of it)\n",
"Metabolites ────────┘ (measurements of the pathway, not cause of it)\n\n",
"Direct Effect: SNP ───────────────────────────→ Drug Response\n\n",

"✅ Measurement model\n\n",
    
"(How the latent pathway activation is measured)\n\n",

"GeneExpression = λ1 ⋅ PathwayActivation + ε1\n",
"Metabolites    = λ2 ⋅ PathwayActivation + ε2\n\n",

"PathwayActivation is latent (unobserved)\n",

"Gene expression and metabolites are noisy indicators\n",

"λi = factor loadings\n",
"εi = measurement error\n\n",
    
"✅ Structural model\n\n",

"(Causal relationships between variables)\n\n",

"PathwayActivation = β1 ⋅ SNP + ζ1\n",
"DrugResponse      = β2 ⋅ PathwayActivation + β3 ⋅ SNP + ζ2\n\n",
"DrugResponse is a measured outcome (e.g., tumor size)\n",

"β1 : SNP → pathway activation\n",
"β2 : pathway → drug response (mediated effect)\n",
"β3 : SNP → drug response (direct effect)\n",
"ζi : unexplained biological variation\n\n",

"Interpretation (important)\n\n",
"Indirect (mediated) effect of SNP on drug response: β1 × β2\n",
"Direct effect: β3\n",
"Total effect: β3 + (β1 × β2)\n",
sep = "")


cat("
SEM specification matching the diagram:

# latent factor: PathwayActivation measured by GeneExpression + Metabolites
# structural: PathwayActivation <- SNP ; DrugResponse <- PathwayActivation + SNP

model_sem <- '
  # Measurement model (latent)
  PathwayActivation =~ GeneExpression + Metabolites

  # Structural model
  PathwayActivation ~ b1*SNP
  DrugResponse      ~ b2*PathwayActivation + b3*SNP

  # Effects
  indirect = b1*b2
  direct   = b3
  total    = b3 + (b1*b2)
")

# Example : simulation

set.seed(1)
n <- 500

SNP <- rbinom(n, 2, 0.3)
Pathway_true <-   0.6 * SNP + rnorm(n, 0, 1)
GeneExpression <- 1.0 * Pathway_true + rnorm(n, 0, 0.7)
Metabolites    <- 0.8 * Pathway_true + rnorm(n, 0, 0.7)
DrugResponse   <- 0.7 * Pathway_true + 0.2 * SNP + rnorm(n, 0, 1)

df <- data.frame(SNP, GeneExpression, Metabolites, DrugResponse)
head(df)
tail(df)

# define SEM model

model_sem <- '
  # Measurement model (latent)
  PathwayActivation =~ GeneExpression + Metabolites

  # Structural model
  PathwayActivation ~ b1*SNP
  DrugResponse      ~ b2*PathwayActivation + b3*SNP

  # Effects
  indirect := b1*b2
  direct   := b3
  total    := b3 + (b1*b2)
'

# sem(model = NULL, data = NULL, ...)

fit <- sem(model_sem, data = df, meanstructure = TRUE)
fit
# str(fit)

# ✅  : Model convergence (first sanity check)
# “lavaan 0.6-19 ended normally after 28 iterations” : good

# SEM is happy when N ≥ 10–20 × number of free parameters

# Test statistic (χ²) = 1.117
# Degrees of freedom = 1
# P-value = 0.291

# What this test asks
# “Is the covariance structure implied by your model significantly different from the observed data?”

# ✅ p = 0.291 → very good fit

# We can say : A model in which the SNP affects drug response partly through a latent pathway (measured by gene expression and metabolites), 
# and partly through a direct effect, is fully consistent with the observed data.

summary(fit, standardized = TRUE, fit.measures = TRUE, rsquare = TRUE)

# 1️⃣ Global model fit (excellent)

# From the $fit section:
# χ² test
# χ²(1) = 1.12
# p = 0.29
# ✅ You do not reject the model → the covariance structure implied by your SEM matches the data well.

# | Metric             | Value  | Interpretation     |
# | ------------------ | ------ | ------------------ |
# | **CFI**            | 0.9998 | Excellent (≫ 0.95) |
# | **TLI**            | 0.9986 | Excellent          |
# | **RMSEA**          | 0.015  | Excellent (< 0.05) |
# | **RMSEA p(close)** | 0.54   | Supports close fit |
# | **SRMR**           | 0.007  | Excellent (< 0.08) |

# ✅ Every major fit index indicates an excellent model fit.

# 2️⃣ Measurement model (latent pathway quality)
# Loadings on PathwayActivation
# Indicator	Estimate	Std.lv	p-value	Meaning
# GeneExpression	1.00 (fixed)	1.08	—	Reference indicator
# Metabolites	0.83	0.89	< 1e−15	Strong loading

# 3️⃣ Structural model (causal paths)

# A. SNP → PathwayActivation (β₁)
# | Estimate  | z    | p-value     | Std.lv |
# | --------- | ---- | ----------- | ------ |
# | **0.546** | 6.80 | **1.0e−11** | 0.51   |

# ✅ Strong effect : the SNP robustly perturbs pathway activation.

# B. PathwayActivation → DrugResponse (β₂)

# | Estimate  | z     | p-value     | Std.lv |
# | --------- | ----- | ----------- | ------ |
# | **0.806** | 11.73 | **< 1e−15** | 0.87   |

# ✅ Strong effect : pathway activation is the dominant driver of drug response.

# C. SNP → DrugResponse (direct, β₃)

# | Estimate  | z    | p-value       | Std.lv |
# | --------- | ---- | ------------- | ------ |
# | **0.071** | 0.85 | **0.40 (NS)** | 0.07   |

# ❌ Not significant : Once pathway activation is accounted for, the SNP has no detectable direct effect on drug response.

# 4️⃣ Mediation: the key biological conclusion

# Because:

# β₁ is significant
# β₂ is significant
# β₃ is not significant

# We have full mediation.

# Indirect effect
# SNP → Pathway → DrugResponse = 𝛽1 × 𝛽2 ≈ 0.55 × 0.81 ≈ 0.45
# 📌 This means: The SNP influences drug response entirely through pathway activation.





parameterEstimates(fit, standardized = TRUE)[
  parameterEstimates(fit)$label %in% c("b1","b2","b3") | parameterEstimates(fit)$lhs %in% c("indirect","direct","total"),
]

# Mediation type (important conclusion)

# Because:
# Indirect effect is significant
# Direct effect is NOT significant
# Total effect is significant

# 👉 This is full mediation.



cat(" Differences between SEM and Regression methods:

| Regression                | SEM                       |
| ------------------------- | ------------------------- |
| One outcome at a time     | Multiple outcomes         |
| No latent variables       | Latent variables allowed  |
| No mediation by default   | Explicit mediation        |
| Measurement error ignored | Measurement error modeled |
| Descriptive               | Hypothesis-driven         |

")





cat(" Differences between SEM and Bayesian Networks :

Conceptual difference (most important) :

SEM answers:

“If my causal theory is correct, would the observed covariance structure look like this?”

Bayesian Networks answer:

“What probabilistic dependencies (and possibly causal structure) best explain the data?”

Latent variables: the biggest divider

SEM : Latent variables are first-class citizens

Bayesian Networks : Nodes are usually observed variables

")


cat("

Biomedical example contrast : 

* SEM example

Genotype → latent pathway activity → phenotype
(measured via RNA, protein, metabolite)

* Bayesian Network example

Gene expression ⟂ protein ⟂ metabolite ⟂ phenotype
(conditional dependencies inferred from data)

")



cat(" The LATENT VARIABLE models : the full landscape - simplified taxonomy

Thinking in two dimensions:

Is the latent variable discrete or continuous?

Does it evolve over time or not?

A. Latent states (discrete, evolving) : 'The system switches between modes.'

1. Hidden Markov Models (HMMs)

Discrete states
Markov dynamics
Examples: disease stages, copy-number states

2. Switching Linear Dynamical Systems

Discrete regimes
Continuous observations
Common in neuroscience, finance

3. Hierarchical HMMs

States within states
Speech, behavior

Key idea:
At any moment, the system is in one of a few regimes.

B. Latent trajectories (continuous, evolving) : 'There is a hidden process unfolding over time.'

1. Kalman Filters / Linear Dynamical Systems

Continuous latent state
Gaussian assumptions

2. General State-Space Models

Nonlinear / non-Gaussian versions

Key idea:
Latent variable = smooth hidden dynamics.

C. Latent factors (continuous, static) : 'Hidden causes of variation.'

1. PCA / Factor Analysis

Explains covariance
No causality, no time

2. Structural Equation Models (SEM)

Latent factors + causal structure
Measurement + structural components

Hypothesis-driven

3. MOFA / Group Factor Analysis

Multi-omics latent factors
Mostly correlational

Key idea:
Latent variable = compressed explanation of variation.

D. Latent classes (discrete, static) : 'Hidden subtypes exist.'

1. Mixture models (e.g., GMMs)
Soft clustering

2. Latent Class Analysis

Often categorical data

Key idea:
Latent variable = unobserved group membership.

E. Latent representations (deep learning) : 'Latents are useful, not predefined.'

1. VAEs (e.g., scVI, totalVI, OmiVAE)
Learned latent space
Weak identifiability

2. Diffusion / flow models
Excellent generation
Poor semantic meaning

Key idea:
Latent variable = mathematical convenience, not concept.

F. Hybrid / modern models : Boundaries are blurred.

1. Dynamic VAEs (VAE + state-space)
2. Neural HMMs (discrete states + neural emissions)
3. Bayesian causal + temporal models

Key idea:
Combine interpretability with flexibility.

")



# An example provided by Grok

# ────────────────────────────────────────────────────────────────
#   Example: Structural Equation Model in R (lavaan)
#   Topic: Stress → Burnout → Turnover Intention
#   Date: January 2026 style
# ───────────────────────────────────────────────────────────

library(lavaan)
# library(semPlot)
library(dplyr)

# ─── 1. Generate clean synthetic data with good measurement properties ───
set.seed(20260121)

n <- 450

# True latents (with mediation structure)
stress_latent   <- rnorm(n, mean = 0, sd = 1)
burnout_latent  <- 0.65 * stress_latent + rnorm(n, 0, sqrt(1 - 0.65^2))
turnover_latent <- 0.20 * stress_latent + 0.60 * burnout_latent + 
                   rnorm(n, 0, sqrt(1 - (0.20^2 + 0.60^2 + 2*0.20*0.60*0.65)))

# Measurement (loadings ~0.7-0.9 after standardization, some error)
stress_items   <- 4 + outer(stress_latent,   c(1.2, 1.3, 1.25, 1.15)) + rnorm(n*4, 0, 0.9)
burnout_items  <- 3 + outer(burnout_latent,  c(1.4, 1.35, 1.45))     + rnorm(n*3, 0, 1.0)
turnover_items <- 2.5 + outer(turnover_latent, c(1.3, 1.4, 1.35))    + rnorm(n*3, 0, 0.95)

dat <- data.frame(
  stress1   = round(stress_items[,1], 1),
  stress2   = round(stress_items[,2], 1),
  stress3   = round(stress_items[,3], 1),
  stress4   = round(stress_items[,4], 1),
  burnout1  = round(burnout_items[,1], 1),
  burnout2  = round(burnout_items[,2], 1),
  burnout3  = round(burnout_items[,3], 1),
  quit1     = round(turnover_items[,1], 1),
  quit2     = round(turnover_items[,2], 1),
  quit3     = round(turnover_items[,3], 1)
)

# Quick sanity check
round(cor(dat), 2)
psych::alpha(dat[,1:4])$total$raw_alpha     # Stress ~0.90+
psych::alpha(dat[,5:7])$total$raw_alpha      # Burnout ~0.88+
psych::alpha(dat[,8:10])$total$raw_alpha     # Turnover ~0.90+

# ─── 2. Model syntax (same as before) ─────────────────────────────────
model <- '
  # Measurement model
  Stress  =~ stress1 + stress2 + stress3 + stress4
  Burnout =~ burnout1 + burnout2 + burnout3
  Turnover =~ quit1 + quit2 + quit3
  
  # Structural model
  Burnout  ~ a*Stress
  Turnover ~ c*Stress + b*Burnout
  
  # Indirect & total effect
  indirect := a * b
  total    := c + (a * b)
'

# ─── 3. Fit with convergence-friendly settings ────────────────────────
fit <- sem(model,
           data       = dat,
           std.lv     = TRUE,           # ← most important for stability
           estimator  = "MLR",          # robust standard errors
           optim.method = "BFGS",       # often better than nlminb near boundaries
           control    = list(maxit = 20000, trace = 2),  # show progress
           missing    = "fiml")

# ─── 4. Check convergence & results ───────────────────────────────────
summary(fit, 
        fit.measures = TRUE, 
        standardized = TRUE, 
        rsquare      = TRUE)

# Should now show converged = TRUE and sensible fit indices
fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))

# Parameter table with bootstrapped CI for indirect effect
parameterEstimates(fit, 
                   boot.ci.type = "bca.simple", 
                   ci = TRUE, level = 0.95)


# ─── Optional: bootstrap for better indirect effect inference ────────
# fit_boot <- sem(model, data = dat, std.lv = TRUE, estimator = "MLR",
#                 se = "bootstrap", bootstrap = 2000)
# parameterEstimates(fit_boot, boot.ci.type = "bca.simple", ci = TRUE)




