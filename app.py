import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from PIL import Image

models_path = "trained_model/"
df = pd.read_csv(f"{models_path}shap_importances_model_targets.csv", sep=";")

targets = list(df["model_target"].unique())
targets = [target.split("_")[1] for target in targets]
features = list(df["feature"].unique())


img_logo = Image.open("static/image/logo.png")
img_main = Image.open("static/image/logo_AI4NF.jpg")


st.set_page_config(
    page_title="Hack4NF",
    page_icon=img_logo,
    layout="wide"
)


st.markdown("<h1 style='text-align: center;'>Hack4NF 2022 Machine Learning</h1>", unsafe_allow_html=True)

with st.expander("ℹ️ - About this app", expanded=False):

    st.markdown(
        """
        ### This app is based in Machine Learning pipeline to predict cancer type based in mutations features  
        \n[Project webpage](https://hack4nf-platform.bemyapp.com/#/projects/634fdae469c573321d684dcf)  
        \n[The Github repository](https://github.com/pasturl/Hack4NF-2022)    
        \nThis project was developed for the Challenge 1 of the Hack4NF 2022.
        The Genie dataset was used and a Machine Learning pipeline with the following steps:  
        1. Download data: Download genie data using synapse id to create structured csv files.  
        2. Create dataset: Transformed data to generate the dataset with all features.  
        3. Supervised ML model: Train a supervised ML model to predict each cancer type using genomic features.  
        4. Model interpretability: Use model explainability techniques (SHAP) to understand feature importance.  
        5. Natural Language Processing: Get gene info from marrvel and apply BERT (state-of-the-art ML model), to detect topics (groups of similar genes).  
        """
    )

    st.markdown("")

st.markdown("")

st.sidebar.image(img_main, width=240)
feature_option = st.sidebar.selectbox("Feature", features, index=0)
target = st.sidebar.selectbox("Model target", targets, index=0)

target_option = "model_" + target

with st.expander("Model feature importance using SHAP", expanded=False):

    st.markdown(f"### Importance of {feature_option}")
    st.dataframe(df[df["feature"] == feature_option])

    st.markdown(f"### Feature importance for {target}")
    st.dataframe(df[df["model_target"] == target_option])

    st.markdown("#### SHAP feature importance")
    shap_bar_plot = f"{models_path}{target_option}/shap_class_1_bar_plot.png"
    st.image(shap_bar_plot, width=700)

    st.markdown("### SHAP feature correlation")
    shap_plot_25 = f"{models_path}{target_option}/shap_class_1_25.png"
    st.image(shap_plot_25, width=700)


with st.expander("Model metrics", expanded=False):
    splits = ["test", "validation"]
    col1, col2 = st.columns(2)
    with col1:
        model_lgb_test_auc = f"{models_path}{target_option}/model_lgb_{splits[0]}_auc.png"
        st.image(model_lgb_test_auc, caption=f"Area Under the Curve ROC for {splits[0]} split", width=400)
    with col2:
        model_lgb_val_auc = f"{models_path}{target_option}/model_lgb_{splits[1]}_auc.png"
        st.image(model_lgb_val_auc, caption=f"Area Under the Curve ROC for {splits[1]} split", width=400)



with st.expander("BERTopic analysis of important genes", expanded=False):
    st.markdown("#### Topic heatmap")
    HtmlFile_heatmap = open(f"nlp/{target}/bert_heatmap.html", 'r', encoding='utf-8')
    source_code_heatmap = HtmlFile_heatmap.read()
    print(source_code_heatmap)
    components.html(source_code_heatmap, height=800, scrolling=True)

    st.markdown("#### Topic words score")
    HtmlFile_hierarchy = open(f"nlp/{target}/bert_hierarchy.html", 'r', encoding='utf-8')
    source_code_hierarchy = HtmlFile_hierarchy.read()
    print(source_code_hierarchy)
    components.html(source_code_hierarchy, height=800, scrolling=True)


    st.markdown("#### Topic words score")
    HtmlFile_words = open(f"nlp/{target}/bert_topic_words_score.html", 'r', encoding='utf-8')
    source_code_words = HtmlFile_words.read()
    print(source_code_words)
    components.html(source_code_words, height=1200, scrolling=True)
