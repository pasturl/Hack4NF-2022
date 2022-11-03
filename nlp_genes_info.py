import logging
import pandas as pd
from bertopic import BERTopic
from nltk.corpus import stopwords
from utils import create_folder

log = logging.getLogger(__name__)

stopwords = stopwords.words('english')
stopwords = stopwords+["et", "al", "http", "href", "omim", "org"]


def read_gene_info(genie):
    df_genes = pd.read_csv(genie["processed_data"]+genie["processed_files"]["gene_info"]["file_name"],
                           sep=";")
    df_genes = df_genes[~df_genes["description"].isna()].reset_index(drop=True)

    return df_genes


def read_importance_features(genie, target, df_genes):
    model_path = f"{genie['models_path']}model_{target}"
    df_features = pd.read_csv(f'{model_path}/feature_importance_shap_lgb.csv', sep=";")
    df_features = df_features[df_features["shap_value"] >= genie["nlp_thresholds_importance"]]
    df_features["Hugo_Symbol"] = df_features["feature"].apply(lambda x: x.split("_")[0])
    # Remove repeated genes and keep most important variant
    df_features.drop_duplicates(subset="Hugo_Symbol", keep="first", inplace=True)
    df_features.reset_index(drop=True, inplace=True)
    df = pd.merge(df_features, df_genes,
                  on="Hugo_Symbol",
                  how="inner")
    return df


def clean_text(df, stopwords):
    df["description_clean"] = df["description"].str.lower().str.replace('[^a-zA-Z]', ' ')
    df["description_clean"] = df["description_clean"].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
    return df


def apply_bertopic(genie, df, model_path):
    docs = []
    for case in df["description_clean"]:
        docs.append(case)
    n_docs = len(docs)
    log.info(f'Total docs after filtering by threshold importance: {n_docs}')
    topic_model = BERTopic(min_topic_size=genie["bert_min_topic_size"])
    topics, probs = topic_model.fit_transform(docs)
    df["topic"] = topics
    df.to_csv(f"{model_path}/bert_documents_topics.csv",
              sep=";", index=False)
    df_topic_model = topic_model.get_topic_info()
    df_topic_model.to_csv(f"{model_path}/bert_topics.csv",
                          sep=";", index=False)
    barchart = topic_model.visualize_barchart(n_words=10,
                                              height=600,
                                              top_n_topics=50)
    barchart.write_html(f"{model_path}/bert_topic_words_score.html")
    topics = topic_model.visualize_topics()
    topics.write_html(f"{model_path}/bert_topics.html")
    heatmap = topic_model.visualize_heatmap()
    heatmap.write_html(f"{model_path}/bert_heatmap.html")
    hierarchical_topics = topic_model.hierarchical_topics(docs)
    hierarchy = topic_model.visualize_hierarchy()
    hierarchy.write_html(f"{model_path}/bert_hierarchy.html")


def bertopic_to_important_genes(genie):
    create_folder(genie["nlp_path"])
    log.info('Loading gene info')
    df_genes = read_gene_info(genie)
    log.info('Loading model feature importance')
    # Execute all cancer types if targets contains "All"
    if "All" in genie["targets"]:
        with open('data/genie_cancer_types_features.txt') as f:
            cancer_types_cols = f.read().splitlines()
        for target in cancer_types_cols:
            log.info(f'Applying BERTopic to {target} model important features')
            model_path = genie["nlp_path"] + target
            create_folder(model_path)
            try:
                df = read_importance_features(genie, target, df_genes)
                df = clean_text(df, stopwords)
                apply_bertopic(genie, df, model_path)
            except:
                log.info(f'Error applying BERTopic to {target} model important features')

    else:
        for target in genie["targets"]:
            log.info(f'Applying BERTopic to {target} model important features')
            model_path = genie["nlp_path"] + target
            create_folder(model_path)
            try:
                df = read_importance_features(genie, target, df_genes)
                df = clean_text(df, stopwords)
                apply_bertopic(genie, df, model_path)
            except:
                log.info(f'Error applying BERTopic to {target} model important features')
