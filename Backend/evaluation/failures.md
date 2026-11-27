# Evaluation failures

## dense

- question: What is the company policy on bringing pets to the office?
  - type: unanswerable
  - expected: (none) pages none
  - first correct rank: none
  - outcome: wrongly answered
- question: How much is the annual performance bonus?
  - type: unanswerable
  - expected: (none) pages none
  - first correct rank: none
  - outcome: wrongly answered
- question: What is the price of a NimbusDB enterprise license?
  - type: unanswerable
  - expected: (none) pages none
  - first correct rank: none
  - outcome: wrongly answered
- question: What is the capital of France?
  - type: unanswerable
  - expected: (none) pages none
  - first correct rank: none
  - outcome: wrongly answered

## bm25

- question: What is the company policy on bringing pets to the office?
  - type: unanswerable
  - expected: (none) pages none
  - first correct rank: none
  - outcome: wrongly answered
- question: How much is the annual performance bonus?
  - type: unanswerable
  - expected: (none) pages none
  - first correct rank: none
  - outcome: wrongly answered
- question: What is the price of a NimbusDB enterprise license?
  - type: unanswerable
  - expected: (none) pages none
  - first correct rank: none
  - outcome: wrongly answered
- question: What is the capital of France?
  - type: unanswerable
  - expected: (none) pages none
  - first correct rank: none
  - outcome: wrongly answered

## hybrid

- question: What is the company policy on bringing pets to the office?
  - type: unanswerable
  - expected: (none) pages none
  - first correct rank: none
  - outcome: wrongly answered
- question: How much is the annual performance bonus?
  - type: unanswerable
  - expected: (none) pages none
  - first correct rank: none
  - outcome: wrongly answered
- question: What is the price of a NimbusDB enterprise license?
  - type: unanswerable
  - expected: (none) pages none
  - first correct rank: none
  - outcome: wrongly answered
- question: What is the capital of France?
  - type: unanswerable
  - expected: (none) pages none
  - first correct rank: none
  - outcome: wrongly answered

## hybrid_reranked

- question: When do I need a doctor's note for being ill?
  - type: paraphrase
  - expected: Acme_HR_Policy.pdf pages 4
  - first correct rank: 1
  - top rerank score: -7.096
  - outcome: wrongly refused

## hybrid_reranked rerank scores

| answerable | type | top rerank score | question |
|---|---|---:|---|
| yes | direct | 8.750 | How many days of annual leave do employees get each year? |
| yes | direct | 5.358 | What is the probation period for new employees? |
| yes | direct | 7.849 | How many hours is the standard work week? |
| yes | direct | 5.779 | How much is the daily meal allowance during business travel? |
| yes | direct | 4.751 | How many weeks of maternity leave are paid? |
| yes | direct | 6.565 | When do carried-forward leave days expire? |
| yes | paraphrase | -0.313 | How many vacation days can staff save for next year? |
| yes | paraphrase | -0.444 | How long do I have to hand in my travel cost reimbursement requests? |
| yes | paraphrase | -1.340 | What pay rate applies when someone works extra hours? |
| yes | paraphrase | -7.096 | When do I need a doctor's note for being ill? |
| yes | exact_term | 2.428 | What is the form code for expense claims? |
| yes | exact_term | 0.190 | Which form is used to declare gifts? |
| yes | exact_term | 5.221 | What is the ethics hotline extension? |
| yes | direct | 7.746 | What is the default port for NimbusDB? |
| yes | direct | 4.968 | How often are automatic backups created? |
| yes | direct | 4.623 | How long are backups retained? |
| yes | direct | 6.296 | What is the default value of max_connections? |
| yes | direct | 0.899 | How many failed logins cause an account lockout? |
| yes | paraphrase | 2.456 | How much memory does the server need at minimum? |
| yes | paraphrase | 7.668 | How long is a login token valid? |
| yes | exact_term | 7.154 | What does error ERR-503 mean? |
| yes | exact_term | 5.693 | Which command applies the migration when upgrading to 4.8? |
| yes | exact_term | 7.691 | What does ERR-912 mean? |
| no | unanswerable | -8.461 | What is the company policy on bringing pets to the office? |
| no | unanswerable | -8.321 | How much is the annual performance bonus? |
| no | unanswerable | -3.905 | What is the price of a NimbusDB enterprise license? |
| no | unanswerable | -11.088 | What is the capital of France? |
