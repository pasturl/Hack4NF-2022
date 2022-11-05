import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

models_path = "trained_model/"
df = pd.read_csv(f"{models_path}shap_importances_model_targets.csv", sep=";")

targets = list(df["model_target"].unique())
features = list(df["feature"].unique())
st.set_page_config(
    page_title="Hack4NF",
    page_icon="🎈",
    layout="wide"
)



st.title("Hack4NF")
st.header("This app ...")



with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """Model feature importance"""
    )

    st.markdown("")

st.markdown("")
st.markdown("## Machine Learning model feature importance using SHAP")

feature_option = st.selectbox("Feature", features, index=0)

st.write('You selected feature:', feature_option)
st.dataframe(df[df["feature"] == feature_option])

target_option = st.selectbox("Model target", targets, index=0)

st.write('You selected model target:', target_option)
st.dataframe(df[df["model_target"] == target_option])

st.markdown("#### SHAP feature importance")
shap_bar_plot = f"{models_path}{target_option}/shap_class_1_bar_plot.png"
st.image(shap_bar_plot)

st.markdown("### SHAP feature correlation")
shap_plot_25 = f"{models_path}{target_option}/shap_class_1_25.png"
st.image(shap_plot_25)

st.markdown("### Model metrics")
model_lgb_test_auc = f"{models_path}{target_option}/model_lgb_test_auc.png"
st.image(model_lgb_test_auc)

target = target_option.split("_")[1]

st.header("BERTopic analysis of important genes")

st.markdown("#### Topic heatmap")
HtmlFile_heatmap = open(f"nlp/{target}/bert_heatmap.html", 'r', encoding='utf-8')
source_code_heatmap = HtmlFile_heatmap.read()
print(source_code_heatmap)
components.html(source_code_heatmap, height=1200, scrolling=True)

st.markdown("#### Topic words score")
HtmlFile_hierarchy = open(f"nlp/{target}/bert_hierarchy.html", 'r', encoding='utf-8')
source_code_hierarchy = HtmlFile_hierarchy.read()
print(source_code_hierarchy)
components.html(source_code_hierarchy, height=1200, scrolling=True)


st.markdown("#### Topic words score")
HtmlFile_words = open(f"nlp/{target}/bert_topic_words_score.html", 'r', encoding='utf-8')
source_code_words = HtmlFile_words.read()
print(source_code_words)
components.html(source_code_words, height=1200, scrolling=True)
