import streamlit as st

st.set_page_config(page_title="NEMESIS Application", layout="wide")
st.markdown(
    """
    <style>
    /* Centre le texte de toutes les balises <h1>, <h2>, etc. */
    h1, h2, h3, h4, h5, h6, p {
        text-align: center;
    }
    /* Centre les images et autres éléments dans leur conteneur */
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    </style>
    """,
    unsafe_allow_html=True
)




st.title("NEMESIS Application")
st.write("Welcome to our modeling and simulation environment for the Georgia State Energy Infrastructure.")
st.write("Use the sidebar to navigate between pages.")
left_co, cent_co, last_co = st.columns(3)
with cent_co:
    st.image("Main_app/PICTURES/nemesis.jpg", caption="Personal Image", width=350)

