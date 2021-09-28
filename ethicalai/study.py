import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.explainers import MetricTextExplainer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_auc_score, roc_curve

# Bias mitigation techniques
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.inprocessing import PrejudiceRemover

from .utils import get_categorical_cols


class ClassificationStudy:
    def __init__(self, data, target, favorable_classes, protected_attribute_names, privileged_classes, split=0.7,
                 **kwargs):

        self.dataset = StandardDataset(
            data,
            label_name=target,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            categorical_features=get_categorical_cols(data, [*protected_attribute_names, target]),
            **kwargs
        )

    def compute_bias_metrics(self, dataset=None):

        if dataset is None:
            dataset = self.dataset

        cols = ['disparate_impact', 'statistical_parity_difference', ]
        obj_fairness = [[0, 1]]

        fair_metrics = pd.DataFrame(data=obj_fairness, index=['objective'], columns=cols)

        for attr in dataset.protected_attribute_names:
            idx = dataset.protected_attribute_names.index(attr)
            privileged_groups = [{attr: dataset.privileged_protected_attributes[idx][0]}]
            unprivileged_groups = [{attr: dataset.unprivileged_protected_attributes[idx][0]}]

            metric_pred = BinaryLabelDatasetMetric(
                dataset,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups
            )

            row = pd.DataFrame([[
                metric_pred.disparate_impact(),
                metric_pred.mean_difference(),
            ]],
                columns=cols,
                index=[attr]
            )
            fair_metrics = fair_metrics.append(row)

        fair_metrics = fair_metrics.replace([-np.inf, np.inf], 2)

        return fair_metrics

    def train_test_split(self, split=0.7):
        return self.dataset.split([split], shuffle=True)

    def fit(self, dataset=None, method="lr", model=None, **kwargs):

        assert method in ["lr", "rf"]

        if dataset is None:
            dataset = self.dataset

        if model is None:
            if method == "lr":

                self.model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(solver=kwargs.get("solver", "liblinear"), **kwargs)
                )

                fit_params = {'logisticregression__sample_weight': dataset.instance_weights}

            elif method == "rf":

                self.model = make_pipeline(
                    StandardScaler(),
                    RandomForestClassifier(
                        n_estimators=kwargs.get("n_estimators", 100),
                        **kwargs
                    )
                )

                fit_params = {'randomforestclassifier__sample_weight': dataset.instance_weights}

            X = dataset.features
            y = dataset.labels.ravel()

            self.model.fit(X, y, **fit_params)
            return self.model

    def evaluate_performances(self, test, plot_roc_curve=True):

        pred = self.model.predict(test.features)
        probas = self.model.predict_proba(test.features)[:, 1]
        true = test.labels.ravel()

        metrics = {}

        metrics["confusion_matrix"] = confusion_matrix(true, pred)
        metrics["accuracy"] = accuracy_score(true, pred)
        metrics["recall"] = recall_score(true, pred)
        metrics["roc_auc"] = roc_auc_score(true, pred)

        if plot_roc_curve:
            fpr, tpr, _ = roc_curve(true, probas)

            plt.figure(figsize=(5, 5))
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % metrics["roc_auc"])
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.show()

        return metrics

    def _make_groups(self, dataset=None):

        if dataset is None:
            dataset = self.dataset

        privileged_groups = {}
        unprivileged_groups = {}

        for feature in self.dataset.protected_attribute_names:
            idx = self.dataset.protected_attribute_names.index(feature)

            privileged_groups[feature] = self.dataset.privileged_protected_attributes[idx][0]
            unprivileged_groups[feature] = self.dataset.unprivileged_protected_attributes[idx][0]

        return [privileged_groups], [unprivileged_groups]

    def evaluate_bias(self, dataset=None, model=None):

        thresh_arr = np.linspace(0.01, 0.8, 100)

        if model is None:
            model = self.model

        if dataset is None:
            dataset = self.dataset

        privileged_groups, unprivileged_groups = self._make_groups(dataset)

        try:
            # sklearn classifier
            y_val_pred_prob = model.predict_proba(dataset.features)  # [:,1]
            pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
        except AttributeError:
            # aif360 inprocessing algorithm
            y_val_pred_prob = model.predict(dataset).scores
            pos_ind = 0

        metric_arrs = defaultdict(list)
        for thresh in thresh_arr:
            y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

            dataset_pred = dataset.copy()
            dataset_pred.labels = y_val_pred
            metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

            metric_arrs['balanced_accuracy'].append((metric.true_positive_rate() + metric.true_negative_rate()) / 2)
            metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
            metric_arrs['disparate_impact'].append(metric.disparate_impact())
            metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
            metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
            metric_arrs['theil_ind'].append(metric.theil_index())
            metric_arrs['true_positive_rate'].append(metric.true_negative_rate())
            metric_arrs['true_negative_rate'].append(metric.true_negative_rate())
            metric_arrs['recall'].append(metric.recall())
            metric_arrs['precision'].append(metric.precision())
            metric_arrs['f1_score'].append(
                (2 * metric.precision() * metric.recall()) / (metric.precision() + metric.recall()))

        disp_imp = np.array(metric_arrs["disparate_impact"])
        disp_imp_err = 1 - np.minimum(disp_imp, 1 / disp_imp)
        metric_arrs["balanced_disparate_impact"] = disp_imp_err
        metric_arrs["thresholds"] = thresh_arr

        return metric_arrs

    def plot_bias_metrics(self, metrics, y_right="balanced_disparate_impact", y_left="balanced_accuracy",
                          mitigated_metrics=None, labels=None):

        if not isinstance(metrics, list): metrics = [metrics]

        fig, ax1 = plt.subplots(figsize=(10, 7))

        reds = ["red", "salmon", "lightcoral"]
        blues = ["blue", "royalblue", "cornflowerblue"]

        for i, metrics_set in enumerate(metrics):

            if len(metrics) == 1:
                label = ""
            else:
                if labels is None:
                    label = str(i)
                else:
                    label = labels[i]

            x = metrics_set["thresholds"]

            y_left_values = metrics_set[y_left]
            y_right_values = metrics_set[y_right]

            ax1.plot(x, y_left_values, c=blues[i], label=f"{y_left} {label}".strip())
            ax1.set_xlabel("Classification Threshold", fontsize=16, fontweight='bold')
            ax1.set_ylabel(y_left, color='b', fontsize=16, fontweight='bold')
            ax1.xaxis.set_tick_params(labelsize=14)
            ax1.yaxis.set_tick_params(labelsize=14)
            ax1.set_ylim(0.5, 0.8)

            ax1.legend(loc="upper left")

            ax2 = ax1.twinx()
            ax2.plot(x, y_right_values, color=reds[i], label=f"{y_right} {label}".strip())
            ax2.set_ylabel(y_right, color='r', fontsize=16, fontweight='bold')
            if 'disparate' in y_right:
                ax2.set_ylim(0., 0.7)
            else:
                ax2.set_ylim(-0.25, 0.1)

            best_ind = np.argmax(y_left_values)

            if len(metrics) == 1:
                ax2.axvline(np.array(x)[best_ind], color='k', linestyle=':')

            ax2.yaxis.set_tick_params(labelsize=14)
            ax2.grid(True)
            ax2.legend(loc="upper right")

    def pre_mitigate_reweighing(self, dataset):
        privileged_groups, unprivileged_groups = self._make_groups(dataset)
        rw = Reweighing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )

        new_dataset = rw.fit_transform(dataset)
        return new_dataset

    def pre_mitigate_disparate_impact_remover(self, dataset):

        disp_remover = DisparateImpactRemover(repair_level=1.0)
        new_dataset = disp_remover.fit_transform(dataset)
        return new_dataset

    def in_mitigate_prejudice_remover(self, dataset, eta=25.0):

        # Training Prejudice Remover model

        sensitive_attr = self.dataset.protected_attribute_names

        model = PrejudiceRemover(sensitive_attr=sensitive_attr, eta=25.0)
        scaler = StandardScaler()

        new_dataset = dataset.copy()
        new_dataset.features = scaler.fit_transform(dataset.features)

        model.fit(new_dataset)

        return model
