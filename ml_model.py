import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
import shap
import joblib
import logging
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from utils import create_folder

log = logging.getLogger(__name__)


def plot_roc_curve(fpr, tpr, roc_auc, model_path, split_set):
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUC ROC curve")
    plt.legend(loc="lower right")
    plt.savefig(f'{model_path}/model_lgb_{split_set}_auc.png', dpi=150)


def evaluate_model(model, X_data, y_data, model_path, genie, target, split_set):
    y_pred = model.predict(X_data)
    df_preds = pd.DataFrame()
    df_preds["prediction"] = y_pred
    df_preds = df_preds.round({'prediction': 4})
    df_preds["target"] = target
    df_preds["split_set"] = split_set

    df_preds[["target", "split_set", "prediction"]].to_csv(f"{model_path}/predictions_{target}_{split_set}.csv", sep=";", index=False)
    threshold = 0.5
    predictions = [1 if case > threshold else 0 for case in y_pred]

    log.info(f'Evaluating model {split_set}')
    accuracy = metrics.accuracy_score(y_data, predictions)
    precision = metrics.precision_score(y_data, predictions, average='macro')
    recall = metrics.recall_score(y_data, predictions, average='macro')
    classification_report = metrics.classification_report(y_data, predictions)
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = metrics.roc_curve(y_data, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, model_path, split_set)

    log.info(f'AUC: {roc_auc}')
    log.info(f'Overall Accuracy: {accuracy}')
    log.info(f'Overall Precision: {precision}')
    log.info(f'Overall Recall: {recall}')
    log.info(f'Classification report: {classification_report}')

    log.info('Saving metrics')
    if os.path.isfile(genie["models_metrics_file"]):
        with open(genie["models_metrics_file"], "a") as fwrite:
            new_line = f"{target};{split_set};{roc_auc};{accuracy};{precision};{recall};\n"
            fwrite.write(new_line)
    else:
        with open(genie["models_metrics_file"], "w") as fwrite:
            new_line = f"target;split_set;roc_auc;accuracy;precision;recall\n"
            fwrite.write(new_line)
            new_line = f"{target};{split_set};{roc_auc};{accuracy};{precision};{recall}\n"
            fwrite.write(new_line)


def model_interpretability(model_gbm, X_test, model_path, genie, target):
    try:
        tree_to_plot = 0
        tree = lgb.create_tree_digraph(model_gbm, orientation="vertical",
                                       tree_index=tree_to_plot)
        graphviz.Source(tree.source,
                        filename=f'{model_path}/tree_{tree_to_plot}.gv',
                        format="png")
    except:
        log.info('Error creating tree diagraph')
    shap_values = shap.TreeExplainer(model_gbm).shap_values(X_test)
    class_selected = 1

    shap_values_class_1 = np.abs(shap_values[1]).mean(0)
    features_shap = model_gbm.feature_name()
    df_importances_shap = pd.DataFrame()
    df_importances_shap["feature"] = features_shap
    df_importances_shap["shap_value"] = shap_values_class_1
    df_importances_shap.sort_values("shap_value", inplace=True, ascending=False)
    df_importances_shap.to_csv(f'{model_path}/feature_importance_shap_lgb.csv', sep=";", index=False)

    for max_display_shap in genie["max_display_shap_importance"]:
        log.info(f'Top-{max_display_shap} feature importance of class: {class_selected}')
        plt.figure()
        fig_summary = shap.summary_plot(shap_values[class_selected],
                                        X_test,
                                        max_display=max_display_shap,
                                        show=False)
        plt.savefig(f'{model_path}/shap_class_{class_selected}_{max_display_shap}.png',
                    dpi=150)
        plt.close()

    log.info(f'Bar plot feature importance of class: {class_selected}')
    plt.figure()
    fig_importance = shap.summary_plot(shap_values[class_selected],
                                       X_test,
                                       plot_type='bar',
                                       show=False)
    plt.savefig(f'{model_path}/shap_class_{class_selected}_bar_plot.png', dpi=150)
    plt.close()

    gene_mutations = genie["gene_mutations_shap_dependence"]
    for gene_mutation in gene_mutations:
        log.info(f'SHAP dependence plot: {gene_mutation}')
        if gene_mutation == target:
            log.info(
                f'{gene_mutation} mutation is the target model {target}')
            continue
        shap_importance = df_importances_shap[df_importances_shap["feature"] == gene_mutation]["shap_value"].values[0]
        if shap_importance < genie["gene_mutations_thresholds_importance"]:
            log.info(f'{gene_mutation} mutation has a shap value of {shap_importance} \n It is not important for the '
                     f'model')
            continue
        else:
            try:
                plt.figure()
                fig_dependence = shap.dependence_plot(gene_mutation,
                                                      shap_values[1],
                                                      X_test.values,
                                                      feature_names=X_test.columns,
                                                      show=False)
                plt.savefig(f'{model_path}/shap_dependence_plot_{gene_mutation}.png',
                            dpi=150,
                            show=False)
                plt.close()
                log.info(f'SHAP dependence plot: {gene_mutation} created')
            except:
                log.info(f'Error SHAP dependence plot: {gene_mutation}')


def train_model(genie, datasets, target):
    log.info(f"Training model for {target} at {genie['aggregation_level']} level")
    model_path = f"{genie['models_path']}model_{target}"
    create_folder(model_path)
    if genie["aggregation_level"] == "patient":
        log.info('Using dataset_by_patient')
        ds = datasets["dataset_by_patient"]
    elif genie["aggregation_level"] == "sample":
        log.info('Using dataset_by_sample')
        ds = datasets["dataset_by_sample"]
    else:
        raise Exception(f"aggregation_level not defined")

    ds = ds.fillna(0)
    # TODO parametrize path
    with open('data/genie_mutations_features.txt') as f:
        mutation_cols = f.read().splitlines()
    features = set(mutation_cols) - set([target])
    features = list(features)

    # TODO parametrize path
    with open('data/genie_cancer_types_features.txt') as f:
        cancer_types_cols = f.read().splitlines()
    features = set(features) - set(cancer_types_cols)
    features = list(features)

    features = features + genie['categorical_features']
    features = features + genie['clinical_features']

    ds["PRIMARY_RACE"] = ds["PRIMARY_RACE"].astype('category')
    ds["PATIENT_ID"] = ds["PATIENT_ID"].astype('category')
    ds["SEX"] = ds["SEX"].apply(lambda x: 1 if x == "Female" else 0)

    log.info(f'Target distribution \n {ds[target].value_counts()}')

    X_trainable, X_test, y_trainable, y_test = train_test_split(ds[features], ds[target],
                                                                test_size=0.10, random_state=42)

    X_train, X_validation, y_train, y_validation = train_test_split(X_trainable, y_trainable,
                                                                    test_size=0.20, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train,
                             categorical_feature=genie['categorical_features'])
    validation_data = lgb.Dataset(X_validation, label=y_validation,
                                  categorical_feature=genie['categorical_features'])

    params_k = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'subsample': 0.8,
        'subsample_freq': 1,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'max_bin': 300,
        'n_estimators': 2000,
        'boost_from_average': False,
        "random_seed": 42}

    model_gbm = lgb.train(params_k, train_data, valid_sets=[validation_data],
                          num_boost_round=5000, early_stopping_rounds=25,
                          verbose_eval=50)
    joblib.dump(model_gbm, f'{model_path}/model_lgb.pkl')

    evaluate_model(model_gbm, X_train, y_train, model_path, genie, target, "train")

    evaluate_model(model_gbm, X_validation, y_validation, model_path, genie, target, "validation")

    evaluate_model(model_gbm, X_test, y_test, model_path, genie, target, "test")

    model_interpretability(model_gbm, X_test, model_path, genie, target)

    return model_gbm
