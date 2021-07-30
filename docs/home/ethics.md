# Ethics checklist

!!! note
    This checklist will guide you through the ethical aspects to consider when approaching a problem.<br>
    You should always read this list, make sure you are at least aware of this point. And ideally respect them.<br>
    *This checklist is inspired from the [Deon checklist](https://deon.drivendata.org/)*

!!! tip
    To update/edit the list and tick what you have verified, update the markdown file ``ethics.md`` and tick each point by adding a ``[ x ]``  


## Data Collection
 - [ ] **Informed consent**: If there are human subjects, have they given informed consent, where subjects affirmatively opt-in and have a clear understanding of the data uses to which they consent?
 - [ ] **Collection bias**: Have we considered sources of bias that could be introduced during data collection and survey design and taken steps to mitigate those?
 - [ ] **Limit PII exposure**: Have we considered ways to minimize exposure of personally identifiable information (PII) for example through anonymization or not collecting information that isn't relevant for analysis?

## Data Storage
- [ ] **Data security**: Do we have a plan to protect and secure data (e.g., encryption at rest and in transit, access controls on internal users and third parties, access logs, and up-to-date software)?
- [ ] **Right to be forgotten**: Do we have a mechanism through which an individual can request their personal information be removed?
- [ ] **Data retention plan**: Is there a schedule or plan to delete the data after it is no longer needed?

## Analysis
- [ ] **Missing perspectives**: Have we sought to address blindspots in the analysis through engagement with relevant stakeholders (e.g., checking assumptions and discussing implications with affected communities and subject matter experts)?
- [ ] **Dataset bias**: Have we examined the data for possible sources of bias and taken steps to mitigate or address these biases (e.g., stereotype perpetuation, confirmation bias, imbalanced classes, or omitted confounding variables)?
- [ ] **Honest representation**: Are our visualizations, summary statistics, and reports designed to honestly represent the underlying data?
- [ ] **Privacy in analysis**: Have we ensured that data with PII are not used or displayed unless necessary for the analysis?
- [ ] **Auditability**: Is the process of generating the analysis well documented and reproducible if we discover issues in the future?
## Modeling
- [ ] **Proxy discrimination**: Have we ensured that the model does not rely on variables or proxies for variables that are unfairly discriminatory?
- [ ] **Fairness across groups**: Have we tested model results for fairness with respect to different affected groups (e.g., tested for disparate error rates)?
- [ ] **Metric selection**: Have we considered the effects of optimizing for our defined metrics and considered additional metrics?
- [ ] **Explainability**: Can we explain in understandable terms a decision the model made in cases where a justification is needed?
- [ ] **Communicate bias**: Have we communicated the shortcomings, limitations, and biases of the model to relevant stakeholders in ways that can be generally understood?
## Deployment
- [ ] **Redress**: Have we discussed with our organization a plan for response if users are harmed by the results (e.g., how does the data science team evaluate these cases and update analysis and models to prevent future harm)?
- [ ] **Roll back**: Is there a way to turn off or roll back the model in production if necessary?
- [ ] **Concept drift**: Do we test and monitor for concept drift to ensure the model remains fair over time?
- [ ] **Unintended use**: Have we taken steps to identify and prevent unintended uses and abuse of the model and do we have a plan to monitor these once the model is deployed?