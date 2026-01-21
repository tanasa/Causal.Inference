cat("Bayes theorem")

cat("The Core Idea: Bayes' Theorem

Bayes' Theorem is the mathematical backbone of Bayesian statistics. 

It shows how to calculate the Posterior Probability (the new, updated belief) using three components: 

the Prior, 
the Likelihood
the Evidence.

Let H be our hypothesis (e.g., 'It will be good weather') 

and D be the data/evidence we observe (e.g., 'Wind and Sun'). 

The formula is:

P(H | D) = P(D | H) * P(H) / P(D)

Term,Name,Meaning in the Weather Example

P(H,D),Posterior

P(H),Prior : The initial, pre-existing probability of Good Weather before looking outside. (Your starting belief)

P(D,H),Likelihood

P(D),Evidence : The overall probability of observing Wind and Sun (regardless of whether the weather is good or bad).

")

cat("

| Term        | Name                               | Meaning in the Weather Example                                                            
| :---------- | :--------------------------------  | ----------------------------------------------------------------------------------------|
| (P(H | D))  | Posterior                          | The probability of good weather *after* observing wind and sun. (Updated belief)          
| (P(H))      | Prior                              | The probability of good weather *before* seeing any evidence. (Starting belief)      
| (P(D | H))  | Likelihood (conditional likelihood)| The probability of observing wind and sun *if* the weather is good.                       
| (P(D))      | Evidence (marginal likelihood)     | The overall probability of observing wind and sun, under all possible weather conditions. 

")

# P(D) : It’s the MARGINAL PROBABILITY of the data, integrating/summing over all hypotheses or models.

# P(D | H) : It's a CONDITIONAL PROBABILITY
# and in the context of Bayesian inference, it’s called the Likelihood 
# (or more precisely, the Conditional Likelihood of the data given the hypothesis).

# Mathematically, 
# P(D∣H) is a conditional probability, but conceptually, we reinterpret it as a function of the hypothesis (H) for fixed data (D).


cat("

POSTERIOR = LIKELIHOOD * PRIOR / EVIDENCE

")

cat("

You can think of PRIOR as the kind of EDUCATED GUESS (or belief) before you see any new data.

So yes — it’s like your initial guess, but ideally it’s not random:

It’s based on previous knowledge, experience, or historical data.

It encodes your expectation about what’s likely to be true.

The prior is your starting belief — a kind of SMART GUESS about what’s true, before seeing any new evidence.

The posterior is what you believe after combining your prior with real data.

")


cat("

🌤 Example: Weather

If you live in a sunny city like San Diego:

🔹 Before looking outside, you might believe there’s an 80% chance of good weather → 

P(H) = 0.8 : that’s your prior belief, based on past experience.

If you live somewhere rainy like London:

P(H) = 0.3 (only a 30% chance of good weather).

Same idea - different prior, based on what you already know.

🔹 After you see new data

Once you look outside and see wind and sun (your data), you update your belief using Bayes’ Theorem to get the posterior.

")

cat("

Example Scenario: Predicting “Good Weather”

Let’s define:

𝐻1 : “The weather is good”

𝐻0 : “The weather is bad”

You want to estimate 

𝑃(Good weather ∣ sunny and not windy)

P(Bad weather ∣ sunny and not windy).

")

print("Case A: Evidence = “sunny and calm”")

cat("

Step 1 — Prior belief

From past experience:

P(Good weather) = 0.6  (60% prior chance of good weather)
P(Bad weather)  = 0.4  (40% prior chance of bad weather) 

Step 2 — Likelihood (how evidence behaves under each hypothesis)

From meteorological data:

P(sunny and calm∣Good weather) = 0.8  # if the day is actually a “good” day, 80% of those days show sunny+calm.

P(sunny and calm∣Bad weather)  = 0.2  # if the day is “bad”, still 20% of those days happen to be sunny+calm.

")

cat("

Step 3 — Evidence probability : Compute P(sunny and calm)

The probability of seeing “sunny and calm” on any given day is:

P(sunny and calm) = 0.8 × 0.6 + 0.2 × 0.4 = 0.48 + 0.08 = 0.56

Interpretation: before knowing whether the day is good or bad, the overall chance of seeing “sunny and calm” is 56%.

")

cat("

Step 4 — Posterior belief (update your belief)

Now apply Bayes’ rule:

P(Good weather∣sunny and calm) = P(sunny and calm∣Good weather) P(Good weather) /P(sunny and calm) =

                               = 0.8 × 0.6 / 0.56 = 0.48 / 0.56 ≈ 0.857
	​
✅ Interpretation: after observing sunny+calm, your belief that it is a “good” day rises from 0.60 to ≈0.857 (≈85.7%).

")

print("Case B: Evidence = “rainy and windy”")

cat("

Case B : If the evidence were opposite (rain + wind)

Let’s say:

P(rainy and windy∣Good weather) = 0.1

P(rainy and windy∣Bad weather) = 0.7

Compute again:

P(rainy and windy) = 0.1 × 0.6 + 0.7 × 0.4 = 0.06 + 0.28 = 0.34

Then:

P(Good weather∣rainy and windy)= 0.1 × 0.6 / 0.34 = 0.06 / 0.34 ≈ 0.176

✅ Interpretation:

If it’s rainy and windy, your belief in good weather drops from 60% to ~18%.

")

cat("

In plain words :

“If 60% of the time the weather is good (and on those days 80% are sunny and calm), 
and 40% of the time it’s bad (but 20% of those are still sunny and calm), 
then overall there’s a 56% chance that any given day is sunny and calm.”

This step is crucial in Bayesian inference because it gives the denominator in Bayes’ formula:

P(H∣D) = P(D∣H) P(H) /  P(D) 
	
P(D) = 0.56 — the normalizing constant that ensures all posterior probabilities sum to 1.

")

cat("

| Hypothesis (H)      | Evidence (D)  | Expression for Conditional Likelihood | Example Value | Interpretation                                                  
| :------------------ | :------------ | :------------------------------------ | :------------ | :---------------------------------------------------------------
| (H_1): Good weather | sunny & calm  | (P(D | H_1) = 0.8)                 | 0.8           | If the day is *actually* good, 80% of those days are sunny and calm.
| (H_0): Bad weather  | sunny & calm  | (P(D | H_0) = 0.2)                 | 0.2           | If the day is bad, 20% of those days are still sunny and calm.      
| (H_1): Good weather | rainy & windy | (P(D | H_1) = 0.1)                 | 0.1           | On good-weather days, 10% are rainy and windy.                      
| (H_0): Bad weather  | rainy & windy | (P(D | H_0) = 0.7)                 | 0.7           | On bad-weather days, 70% are rainy and windy.                       

")



# We observe 10 days of weather with two recorded features:

# Weather (H): "Good" or "Bad"
# Condition (D): "SunnyCalm" or "RainyWindy"

# Example dataset: observed 10 days
weather_df <- data.frame(
  Weather   = c("Good","Good","Bad","Good","Bad","Good","Bad","Good","Bad","Good"),
  Condition = c("SunnyCalm","SunnyCalm","RainyWindy","SunnyCalm",
                "RainyWindy","RainyWindy","SunnyCalm","SunnyCalm",
                "RainyWindy","RainyWindy")
)

print(weather_df)

# Create contingency table
tbl <- table(weather_df$Condition, weather_df$Weather)
print(tbl)

cond_probs <- prop.table(tbl, margin = 2)  # margin=2 → condition on columns (Weather)
print(cond_probs)

# ? prop.table
# Returns conditional proportions given margins, i.e. entries of x, divided by the appropriate marginal sums. 

# margin 	
# A vector giving the margins to split by. E.g., 
# 1 indicates rows, 
# 2 indicates columns, 
# c(1, 2) indicates rows and columns. 

P_D_given_Good <- cond_probs["SunnyCalm", "Good"]
P_D_given_Bad  <- cond_probs["SunnyCalm", "Bad"]

print(P_D_given_Good)
print(P_D_given_Bad)

# 3) Priors P(H)

priors <- prop.table(table(weather_df$Weather))
print(priors)

P_Good <- priors["Good"]
P_Bad  <- priors["Bad"]

# 4) Posterior for D = "SunnyCalm" via Bayes

print(P_Good)
print(P_Bad)

P_Good_given_D <- (P_D_given_Good * P_Good) / ((P_D_given_Good * P_Good) + (P_D_given_Bad * P_Bad))

cat("Posterior P(Good | SunnyCalm) =", round(P_Good_given_D, 3), "\n")



cat("CONDITIONAL PROBABILITY")

library(tidyverse)
library(ggplot2)

# Create a visual example
set.seed(789)
n <- 200

visual_data <- tibble(
  id = 1:n,
  event_B = rbinom(n, 1, 0.4),  # B occurs 40% of time
  event_A = case_when(
    event_B == 1 ~ rbinom(n, 1, 0.7),  # A occurs 70% when B occurs
    event_B == 0 ~ rbinom(n, 1, 0.2)   # A occurs 20% when B doesn't occur
  )
) %>%
  mutate(
    category = case_when(
      event_A == 1 & event_B == 1 ~ "Both A and B",
      event_A == 1 & event_B == 0 ~ "A only",
      event_A == 0 & event_B == 1 ~ "B only",
      event_A == 0 & event_B == 0 ~ "Neither"
    )
  )

# Count each category
counts <- visual_data %>% count(category)

# Calculate probabilities
p_B <- mean(visual_data$event_B)
p_A_and_B <- mean(visual_data$event_A == 1 & visual_data$event_B == 1)
p_A_given_B <- mean(visual_data$event_A[visual_data$event_B == 1])

cat("=== VISUAL UNDERSTANDING ===\n\n")
cat("Total observations:", n, "\n")
cat("Event B occurs:", sum(visual_data$event_B), "times (", 
    round(p_B * 100, 1), "%)\n", sep = "")
cat("A and B together:", sum(visual_data$event_A == 1 & visual_data$event_B == 1), 
    "times (", round(p_A_and_B * 100, 1), "%)\n\n", sep = "")

cat("CONDITIONAL PROBABILITY:\n")
cat("P(A | B) = P(A and B) / P(B)\n")
cat("         = ", round(p_A_and_B, 3), " / ", round(p_B, 3), "\n", sep = "")
cat("         = ", round(p_A_given_B, 3), "\n\n", sep = "")

cat("INTERPRETATION:\n")
cat("Out of the ", sum(visual_data$event_B), " times B occurred,\n", sep = "")
cat("A also occurred ", sum(visual_data$event_A == 1 & visual_data$event_B == 1), 
    " times\n", sep = "")
cat("So P(A|B) = ", sum(visual_data$event_A == 1 & visual_data$event_B == 1), "/", 
    sum(visual_data$event_B), " = ", round(p_A_given_B, 3), "\n\n", sep = "")

# Create Venn diagram style visualization
ggplot(visual_data, aes(x = event_B, y = event_A)) +
  geom_jitter(aes(color = category), width = 0.3, height = 0.3, 
              alpha = 0.6, size = 3) +
  scale_color_manual(values = c("Both A and B" = "purple",
                                "A only" = "blue",
                                "B only" = "red",
                                "Neither" = "gray")) +
  scale_x_continuous(breaks = c(0, 1), labels = c("B doesn't occur", "B occurs")) +
  scale_y_continuous(breaks = c(0, 1), labels = c("A doesn't occur", "A occurs")) +
  labs(title = "Conditional Probability Visualization",
       subtitle = paste0("P(A|B) = ", round(p_A_given_B, 3), 
                        " (proportion of purple among B=1 column)"),
       x = "Event B",
       y = "Event A",
       color = "Category") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Bar chart showing the calculation
calc_data <- tibble(
  step = c("1. All observations", "2. Filter to B=1", "3. Count A=1 among B=1"),
  count = c(n, sum(visual_data$event_B), 
            sum(visual_data$event_A == 1 & visual_data$event_B == 1)),
  proportion = c(1, p_B, p_A_and_B)
)

ggplot(calc_data, aes(x = step, y = count, fill = step)) +
  geom_col(alpha = 0.7) +
  geom_text(aes(label = paste0(count, "\n(", round(proportion * 100, 1), "%)")), 
            vjust = -0.5, fontface = "bold") +
  labs(title = "Calculating P(A|B) Step by Step",
       subtitle = paste0("Final: ", sum(visual_data$event_A == 1 & visual_data$event_B == 1),
                        "/", sum(visual_data$event_B), " = ", round(p_A_given_B, 3)),
       y = "Count",
       x = "") +
  theme_minimal() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 15, hjust = 1))





library(tidyverse)

cat("=== FROM CONDITIONAL PROBABILITY TO BAYES' THEOREM ===\n\n")

cat("We know:\n")
cat("P(A | B) = P(A ∩ B) / P(B)   ... (1)\n")
cat("P(B | A) = P(A ∩ B) / P(A)   ... (2)\n\n")

cat("From (1): P(A ∩ B) = P(A | B) × P(B)\n")
cat("From (2): P(A ∩ B) = P(B | A) × P(A)\n\n")

cat("Therefore: P(A | B) × P(B) = P(B | A) × P(A)\n\n")

cat("Rearranging: P(A | B) = [P(B | A) × P(A)] / P(B)\n\n")

cat("This is BAYES' THEOREM!\n\n")

# Example: Disease testing (again)
cat("=== EXAMPLE: DISEASE TESTING ===\n\n")

# Given information
p_disease <- 0.01  # P(Disease) = Prior
p_pos_given_disease <- 0.95  # P(Test+ | Disease) = Sensitivity
p_pos_given_no_disease <- 0.10  # P(Test+ | No Disease) = 1 - Specificity (FPR)

cat("GIVEN:\n")
cat("P(Disease) = ", p_disease, "\n", sep = "")
cat("P(Test+ | Disease) = ", p_pos_given_disease, "\n", sep = "")
cat("P(Test+ | No Disease) = ", p_pos_given_no_disease, "\n\n", sep = "")

cat("WANT:\n")
cat("P(Disease | Test+) = ?\n\n")

# Calculate using Bayes' theorem
# First need P(Test+) using law of total probability
p_pos <- p_pos_given_disease * p_disease + 
         p_pos_given_no_disease * (1 - p_disease)

cat("STEP 1: Calculate P(Test+) using law of total probability\n")
cat("P(Test+) = P(Test+|Disease)×P(Disease) + P(Test+|No Disease)×P(No Disease)\n")
cat("         = ", p_pos_given_disease, "×", p_disease, " + ", 
    p_pos_given_no_disease, "×", (1-p_disease), "\n", sep = "")
cat("         = ", round(p_pos, 4), "\n\n", sep = "")

# Apply Bayes' theorem
p_disease_given_pos <- (p_pos_given_disease * p_disease) / p_pos

cat("STEP 2: Apply Bayes' Theorem\n")
cat("P(Disease | Test+) = P(Test+ | Disease) × P(Disease) / P(Test+)\n")
cat("                   = ", p_pos_given_disease, " × ", p_disease, " / ", 
    round(p_pos, 4), "\n", sep = "")
cat("                   = ", round(p_disease_given_pos, 4), "\n\n", sep = "")

cat("ANSWER: P(Disease | Test+) = ", round(p_disease_given_pos * 100, 1), 
    "%\n\n", sep = "")

cat("Even with 95% sensitivity, only ", round(p_disease_given_pos * 100, 1), 
    "% of positive tests\n", sep = "")
cat("indicate actual disease because the disease is rare!\n")



# Given information

cat("

p_disease <- 0.01               # P(Disease) = Prior

p_pos_given_disease <- 0.95     # P(Test+ | Disease) = Sensitivity (TRUE POSITIVE RATE)

p_pos_given_no_disease <- 0.10  # P(Test+ | No Disease) = 1 - Specificity (FALSE POSITIVE RATE)

p_pos_given_no_disease <- 0.10  # P(Test+ | No Disease) = 1 - Specificity

This line defines the false positive rate, which is the complement of specificity:

Specificity = 1 − 𝑃 ( Test+ ∣ ¬ Disease)
= 1 − 0.10
= 0.90

So:

Specificity = 90%

This means 90% of healthy people correctly test negative.

10% of healthy people incorrectly test positive (false positives).

Sensitivity tells you how well the test detects disease (true positives).

Specificity tells you how well the test avoids false alarms (true negatives).

In Bayesian inference, both are used to calculate the posterior probability of disease given a positive test.

")

cat("Claude AI")

library(tidyverse)

cat("=== THE DISEASE TESTING PROBLEM ===\n\n")

# Given information
p_disease <- 0.01
p_healthy <- 1 - p_disease
sensitivity <- 0.95  # P(Test+ | Disease)
false_positive_rate <- 0.10  # P(Test+ | Healthy)

cat("GIVEN INFORMATION:\n")
cat("─────────────────\n")
cat("1. P(Disease) = 0.01 = 1%\n")
cat("   → Out of 100 people, 1 has the disease\n\n")

cat("2. P(Healthy) = 1 - 0.01 = 0.99 = 99%\n")
cat("   → Out of 100 people, 99 are healthy\n\n")

cat("3. P(Test+ | Disease) = 0.95 = 95%\n")
cat("   → If you HAVE the disease, test is positive 95% of the time\n")
cat("   → This is SENSITIVITY (TRUE POSITIVE RATE)\n\n")

cat("4. P(Test+ | Healthy) = 0.10 = 10%\n")
cat("   → If you're HEALTHY, test is STILL positive 10% of the time\n")
cat("   → This is FALSE POSITIVE rate\n\n")

cat("THE QUESTION:\n")
cat("─────────────\n")
cat("If someone tests positive, what's the probability they ACTUALLY have the disease?\n")
cat("In symbols: P(Disease | Test+) = ?\n\n")

library(tidyverse)

cat("=== STEP 1: LAW OF TOTAL PROBABILITY ===\n\n")

cat("There are TWO ways to get a positive test:\n\n")

cat("WAY 1: You HAVE disease AND test correctly identifies it\n")
cat("       P(Test+ AND Disease) = P(Test+ | Disease) × P(Disease)\n")
cat("                            = 0.95 × 0.01\n")

# Calculate
way1 <- 0.95 * 0.01
cat("                            = ", way1, "\n\n", sep = "")

cat("       Interpretation: ", way1 * 100, "% of ALL people are\n", sep = "")
cat("       both diseased AND test positive (TRUE POSITIVES)\n\n")

cat("WAY 2: You're HEALTHY but test gives false positive\n")
cat("       P(Test+ AND Healthy) = P(Test+ | Healthy) × P(Healthy)\n")
cat("       [Note: 'Healthy' = 'No Disease' = ¬Disease]\n")
cat("                            = 0.10 × 0.99\n")

# Calculate
way2 <- 0.10 * 0.99
cat("                            = ", way2, "\n\n", sep = "")

cat("       Interpretation: ", way2 * 100, "% of ALL people are\n", sep = "")
cat("       healthy BUT still test positive (FALSE POSITIVES)\n\n")

cat("TOTAL probability of testing positive:\n")
cat("P(Test+) = Way 1 + Way 2\n")
cat("         = ", way1, " + ", way2, "\n", sep = "")

p_test_positive <- way1 + way2
cat("         = ", p_test_positive, "\n\n", sep = "")

cat("This means ", round(p_test_positive * 100, 2), "% of ALL people test positive\n", sep = "")
cat("(whether they have the disease or not)\n\n")

# Visualize with a population
cat("=== CONCRETE EXAMPLE: 10,000 PEOPLE ===\n\n")

n_total <- 10000
n_diseased <- n_total * 0.01
n_healthy <- n_total * 0.99

cat("Total people:", n_total, "\n")
cat("├─ Diseased:", n_diseased, "people (1%)\n")
cat("└─ Healthy:", n_healthy, "people (99%)\n\n")

# Among diseased
n_diseased_test_pos <- n_diseased * 0.95
n_diseased_test_neg <- n_diseased * 0.05

cat("Among the", n_diseased, "diseased people:\n")
cat("├─ Test positive:", n_diseased_test_pos, "(95% of 100) ← TRUE POSITIVES\n")
cat("└─ Test negative:", n_diseased_test_neg, "(5% of 100)  ← FALSE NEGATIVES\n\n")

# Among healthy
n_healthy_test_pos <- n_healthy * 0.10
n_healthy_test_neg <- n_healthy * 0.90

cat("Among the", n_healthy, "healthy people:\n")
cat("├─ Test positive:", n_healthy_test_pos, "(10% of 9900) ← FALSE POSITIVES\n")
cat("└─ Test negative:", n_healthy_test_neg, "(90% of 9900) ← TRUE NEGATIVES\n\n")

# Total positives
n_total_positive <- n_diseased_test_pos + n_healthy_test_pos

cat("TOTAL people who test positive:\n")
cat("= ", n_diseased_test_pos, " (true positives) + ", 
    n_healthy_test_pos, " (false positives)\n", sep = "")
cat("= ", n_total_positive, " people\n\n", sep = "")

cat("Proportion who test positive:\n")
cat("= ", n_total_positive, " / ", n_total, "\n", sep = "")
cat("= ", n_total_positive / n_total, "\n", sep = "")
cat("= ", round((n_total_positive / n_total) * 100, 2), "%\n\n", sep = "")

cat("This matches our calculation: P(Test+) = ", p_test_positive, " ✓\n\n", sep = "")

# Create visualization
population_data <- tibble(
  status = c(rep("Diseased", n_diseased_test_pos),
             rep("Diseased", n_diseased_test_neg),
             rep("Healthy", n_healthy_test_pos),
             rep("Healthy", n_healthy_test_neg)),
  test_result = c(rep("Positive", n_diseased_test_pos),
                  rep("Negative", n_diseased_test_neg),
                  rep("Positive", n_healthy_test_pos),
                  rep("Negative", n_healthy_test_neg)),
  category = c(rep("True Positive", n_diseased_test_pos),
               rep("False Negative", n_diseased_test_neg),
               rep("False Positive", n_healthy_test_pos),
               rep("True Negative", n_healthy_test_neg))
)

# Count for visualization
counts <- population_data %>% count(category)

ggplot(counts, aes(x = "", y = n, fill = category)) +
  geom_col(width = 1) +
  geom_text(aes(label = paste0(category, "\n", n, " people")),
            position = position_stack(vjust = 0.5),
            color = "white", fontface = "bold", size = 4) +
  scale_fill_manual(values = c("True Positive" = "darkgreen",
                                "False Positive" = "orange",
                                "False Negative" = "red",
                                "True Negative" = "lightblue")) +
  labs(title = "Population of 10,000 People",
       subtitle = "Distribution of test results") +
  coord_flip() +
  theme_void() +
  theme(legend.position = "none")

library(tidyverse)

cat("=== STEP 2: BAYES' THEOREM ===\n\n")

cat("BAYES' THEOREM FORMULA:\n")
cat("─────────────────────────\n\n")

cat("P(Disease | Test+) = P(Test+ | Disease) × P(Disease)\n")
cat("                     ──────────────────────────────────\n")
cat("                              P(Test+)\n\n")

cat("In words:\n")
cat("Posterior = (Likelihood × Prior) / Evidence\n\n")

cat("COMPONENTS:\n")
cat("───────────\n\n")

# Numerator
cat("NUMERATOR (Joint Probability):\n")
cat("P(Test+ | Disease) × P(Disease)\n")
cat("= 0.95 × 0.01\n")

numerator <- 0.95 * 0.01
cat("= ", numerator, "\n\n", sep = "")

cat("This is the probability of BOTH:\n")
cat("- Having the disease AND\n")
cat("- Testing positive\n\n")

# Denominator
cat("DENOMINATOR (Total Probability of Test+):\n")
cat("P(Test+) = ", p_test_positive, " (from Step 1)\n\n", sep = "")

cat("This is the probability of testing positive\n")
cat("(regardless of whether you have disease or not)\n\n")

# Division
cat("DIVISION:\n")
cat("P(Disease | Test+) = ", numerator, " / ", p_test_positive, "\n", sep = "")

posterior <- numerator / p_test_positive
cat("                   = ", posterior, "\n", sep = "")
cat("                   = ", round(posterior * 100, 2), "%\n\n", sep = "")

cat("=" , rep("=", 50), "\n", sep = "")
cat("FINAL ANSWER: ", round(posterior * 100, 1), "%\n", sep = "")
cat("=" , rep("=", 50), "\n\n", sep = "")

cat("If you test POSITIVE, there's only an ", round(posterior * 100, 1), 
    "% chance\n", sep = "")
cat("you actually HAVE the disease!\n\n")



cat("Naive Bayes is just Bayes' Theorem applied to classification, with one key assumption: features are conditionally independent.")

cat("FEATURES are CONDITIONALLY INDEPENDENT")

library(tidyverse)

cat("=== NAIVE BAYES: SPAM EMAIL CLASSIFICATION ===\n\n")

cat("GIVEN DATA (from training emails):\n")
cat("──────────────────────────────────\n\n")

# Prior probabilities
p_spam <- 0.4
p_not_spam <- 0.6

cat("PRIORS (base rates):\n")
cat("P(Spam) = 0.4 = 40%\n")
cat("  → Out of 100 emails, 40 are spam\n\n")

cat("P(Not Spam) = 0.6 = 60%\n")
cat("  → Out of 100 emails, 60 are legitimate\n\n")

# Likelihoods for each word
cat("LIKELIHOODS (word frequencies):\n\n")

cat("In SPAM emails:\n")
cat("  P('free' | Spam) = 0.8   → 'free' appears in 80% of spam\n")
cat("  P('win' | Spam) = 0.6    → 'win' appears in 60% of spam\n")
cat("  P('money' | Spam) = 0.7  → 'money' appears in 70% of spam\n\n")

cat("In NOT SPAM emails:\n")
cat("  P('free' | Not Spam) = 0.1   → 'free' appears in 10% of legitimate emails\n")
cat("  P('win' | Not Spam) = 0.05   → 'win' appears in 5% of legitimate emails\n")
cat("  P('money' | Not Spam) = 0.2  → 'money' appears in 20% of legitimate emails\n\n")

cat("NEW EMAIL RECEIVED: 'free win money'\n\n")

cat("QUESTION: Is this spam?\n")
cat("Calculate: P(Spam | 'free', 'win', 'money') = ?\n\n")

library(tidyverse)

cat("=== STEP 1: BAYES' THEOREM ===\n\n")

cat("STANDARD BAYES' THEOREM:\n")
cat("────────────────────────\n\n")

cat("P(Spam | free, win, money) = P(free, win, money | Spam) × P(Spam)\n")
cat("                             ─────────────────────────────────────\n")
cat("                                    P(free, win, money)\n\n")

cat("Components:\n")
cat("  • POSTERIOR: P(Spam | free, win, money) ← What we want!\n")
cat("  • LIKELIHOOD: P(free, win, money | Spam) ← Probability of seeing these words in spam\n")
cat("  • PRIOR: P(Spam) = 0.4 ← Base rate of spam\n")
cat("  • EVIDENCE: P(free, win, money) ← Overall probability of these words\n\n")

cat("THE PROBLEM:\n")
cat("How do we calculate P(free, win, money | Spam)?\n")
cat("This is complicated because words might depend on each other!\n\n")

cat("THE NAIVE ASSUMPTION:\n")
cat("Assume words are INDEPENDENT given the class\n")
cat("(This is the 'naive' part!)\n\n")

cat("P(free, win, money | Spam) = P(free|Spam) × P(win|Spam) × P(money|Spam)\n\n")

cat("This simplifies the calculation dramatically!\n\n")

library(tidyverse)

cat("=== STEP 2: NAIVE ASSUMPTION → MULTIPLY LIKELIHOODS ===\n\n")

cat("We assume words are INDEPENDENT given the class.\n")
cat("This means we can multiply individual probabilities!\n\n")

# For SPAM
p_free_given_spam <- 0.8
p_win_given_spam <- 0.6
p_money_given_spam <- 0.7

cat("FOR SPAM CLASS:\n")
cat("───────────────\n\n")

cat("P(free, win, money | Spam)\n")
cat("  = P(free | Spam) × P(win | Spam) × P(money | Spam)\n")
cat("  = ", p_free_given_spam, " × ", p_win_given_spam, " × ", p_money_given_spam, "\n", sep = "")

likelihood_spam <- p_free_given_spam * p_win_given_spam * p_money_given_spam

cat("\n")
cat("CALCULATION:\n")
cat("  Step 1: 0.8 × 0.6 = ", 0.8 * 0.6, "\n", sep = "")
cat("  Step 2: ", 0.8 * 0.6, " × 0.7 = ", likelihood_spam, "\n\n", sep = "")

cat("RESULT: P(free, win, money | Spam) = ", likelihood_spam, "\n\n", sep = "")

cat("INTERPRETATION:\n")
cat("  If an email is spam, there's a ", likelihood_spam * 100, 
    "% chance\n", sep = "")
cat("  it contains all three words: 'free', 'win', and 'money'\n\n\n")

# For NOT SPAM
p_free_given_not <- 0.1
p_win_given_not <- 0.05
p_money_given_not <- 0.2

cat("FOR NOT SPAM CLASS:\n")
cat("───────────────────\n\n")

cat("P(free, win, money | Not Spam)\n")
cat("  = P(free | Not Spam) × P(win | Not Spam) × P(money | Not Spam)\n")
cat("  = ", p_free_given_not, " × ", p_win_given_not, " × ", p_money_given_not, "\n", sep = "")

likelihood_not_spam <- p_free_given_not * p_win_given_not * p_money_given_not

cat("\n")
cat("CALCULATION:\n")
cat("  Step 1: 0.1 × 0.05 = ", 0.1 * 0.05, "\n", sep = "")
cat("  Step 2: ", 0.1 * 0.05, " × 0.2 = ", likelihood_not_spam, "\n\n", sep = "")

cat("RESULT: P(free, win, money | Not Spam) = ", likelihood_not_spam, "\n\n", sep = "")

cat("INTERPRETATION:\n")
cat("  If an email is legitimate, there's only a ", likelihood_not_spam * 100, 
    "% chance\n", sep = "")
cat("  it contains all three words together\n\n")

cat("COMPARISON:\n")
cat("───────────\n")
cat("Likelihood if SPAM:     ", likelihood_spam, " (", 
    round(likelihood_spam * 100, 1), "%)\n", sep = "")
cat("Likelihood if NOT SPAM: ", likelihood_not_spam, " (", 
    round(likelihood_not_spam * 100, 1), "%)\n\n", sep = "")

cat("These words are ", round(likelihood_spam / likelihood_not_spam, 1), 
    "× more likely to appear together in spam!\n\n", sep = "")

library(tidyverse)

cat("=== STEP 3: COMPUTE EVIDENCE (NORMALIZING CONSTANT) ===\n\n")

cat("We need the TOTAL probability of seeing these words,\n")
cat("regardless of whether the email is spam or not.\n\n")

cat("LAW OF TOTAL PROBABILITY:\n")
cat("─────────────────────────\n\n")

cat("P(free, win, money) = P(free, win, money | Spam) × P(Spam)\n")
cat("                    + P(free, win, money | Not Spam) × P(Not Spam)\n\n")

# Values we have
p_spam <- 0.4
p_not_spam <- 0.6
likelihood_spam <- 0.336
likelihood_not_spam <- 0.001

cat("Plug in the values:\n\n")

# Calculate each part
part1 <- likelihood_spam * p_spam
part2 <- likelihood_not_spam * p_not_spam

cat("PART 1 (from spam emails):\n")
cat("  P(words | Spam) × P(Spam)\n")
cat("  = ", likelihood_spam, " × ", p_spam, "\n", sep = "")
cat("  = ", part1, "\n\n", sep = "")

cat("PART 2 (from legitimate emails):\n")
cat("  P(words | Not Spam) × P(Not Spam)\n")
cat("  = ", likelihood_not_spam, " × ", p_not_spam, "\n", sep = "")
cat("  = ", part2, "\n\n", sep = "")

evidence <- part1 + part2

cat("TOTAL EVIDENCE:\n")
cat("  P(free, win, money) = ", part1, " + ", part2, "\n", sep = "")
cat("                      = ", evidence, "\n\n", sep = "")

cat("INTERPRETATION:\n")
cat("  Out of ALL emails, ", round(evidence * 100, 2), 
    "% contain all three words\n", sep = "")
cat("  'free', 'win', and 'money' together\n\n")

cat("BREAKDOWN:\n")
cat("  • ", round(part1 * 100, 2), "% from spam emails\n", sep = "")
cat("  • ", round(part2 * 100, 2), "% from legitimate emails\n\n", sep = "")

cat("Most of these emails (", round((part1/evidence) * 100, 1), 
    "%) are spam!\n\n", sep = "")

library(tidyverse)

cat("=== STEP 4: POSTERIOR PROBABILITY ===\n\n")

cat("Now we apply Bayes' Theorem to get our answer!\n\n")

cat("FORMULA:\n")
cat("────────\n\n")

cat("P(Spam | data) = P(data | Spam) × P(Spam)\n")
cat("                 ────────────────────────\n")
cat("                       P(data)\n\n")

# Our values
likelihood_spam <- 0.336
p_spam <- 0.4
evidence <- 0.135

cat("SUBSTITUTE VALUES:\n")
cat("──────────────────\n\n")

cat("P(Spam | free, win, money) = (0.336 × 0.4) / 0.135\n\n")

# Calculate numerator
numerator <- likelihood_spam * p_spam

cat("STEP 1: Calculate numerator\n")
cat("  0.336 × 0.4 = ", numerator, "\n\n", sep = "")

# Calculate final result
posterior_spam <- numerator / evidence

cat("STEP 2: Divide by evidence\n")
cat("  ", numerator, " / ", evidence, " = ", posterior_spam, "\n\n", sep = "")

cat("Convert to percentage:\n")
cat("  ", posterior_spam, " × 100 = ", round(posterior_spam * 100, 1), "%\n\n", sep = "")

cat("=" , rep("=", 60), "\n", sep = "")
cat("FINAL ANSWER: P(Spam | free, win, money) ≈ ", round(posterior_spam, 3), 
    " = ", round(posterior_spam * 100, 1), "%\n", sep = "")
cat("=" , rep("=", 60), "\n\n", sep = "")

cat("RESULT:\n")
cat("───────\n")
cat("This email has a ", round(posterior_spam * 100, 1), 
    "% chance of being SPAM!\n\n", sep = "")

cat("DECISION: Classify as SPAM ✓\n\n")

# Calculate probability it's NOT spam
posterior_not_spam <- 1 - posterior_spam

cat("For completeness:\n")
cat("P(Not Spam | data) = 1 - ", round(posterior_spam, 3), 
    " = ", round(posterior_not_spam, 3), "\n", sep = "")
cat("                   = ", round(posterior_not_spam * 100, 1), 
    "% chance it's legitimate\n\n", sep = "")



cat("\n=== KEY TAKEAWAYS ===\n\n")

cat("1. NAIVE BAYES = BAYES' THEOREM + INDEPENDENCE ASSUMPTION\n\n")

cat("2. THE FORMULA:\n")
cat("   P(Class | features) ∝ P(Class) × ∏ P(feature_i | Class)\n")
cat("                         [Prior]   [Product of likelihoods]\n\n")

cat("3. THE STEPS:\n")
cat("   a) Calculate likelihood for each class\n")
cat("      (multiply individual feature probabilities)\n")
cat("   b) Multiply by prior P(Class)\n")
cat("   c) Normalize (divide by evidence)\n")
cat("   d) Choose class with highest probability\n\n")

cat("4. WHY MULTIPLY?\n")
cat("   Independence assumption:\n")
cat("   P(A,B,C|Class) = P(A|Class) × P(B|Class) × P(C|Class)\n\n")

cat("5. THE 'NAIVE' PART:\n")
cat("   Assumes features are independent (usually wrong!)\n")
cat("   But works surprisingly well in practice\n\n")

cat("6. SAME MATH AS DISEASE EXAMPLE:\n")
cat("   Disease test: 1 feature (test result)\n")
cat("   Spam filter: Multiple features (words)\n")
cat("   Both use Bayes' Theorem!\n")


