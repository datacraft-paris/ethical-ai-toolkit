# Quickstart

```python

from ethicalai.eda import FacetsVisualizer
from ethicalai.fairness import AutoBiasDetector,BiasEvaluator,BiasMitigator,AutoBiasMitigator
from ethicalai.fairness import BiasComparator


df = pd.read_excel(....)
target = "credit_accepted"


facets = FacetsVisualizer()
facets.visualize(df,target)
# First intuition, there is some bias with age and gender, let's confirm that

evaluator = BiasEvaluator(metrics = ["disparate_impact",...])

auto = AutoBiasDetector(evaluator = evaluator) # exclude parameters
results,fig = auto.evaluate(df,target)
# cols_to_study = results.sort_values("disparate_impact")["feature"].head(2)

cols_to_study = ["age","gender"]


# Model run
model_run = ModelRun(
    model = RandomForestClassifier(...),
    data = df,
    target = target,
    split = ...
)

model = RandomForestClassifier(...)
X_train,X_test,.... = train_test_split(df.drop(columns = target),df[target])
model.fit(X_train,y_train)
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)





evaluator.evaluate_bias(model_run)

# Mitigator
mitigator = BiasMitigator()
study1 = mitigator.pre_mitigate(experiment)
study2 = mitigator.pre_mitigate2(experiment)

# Comparator
comparator = BiasComparator()
comparator.compare([study1,study2,...])

# AutoMitigate
# ???


 



```















```