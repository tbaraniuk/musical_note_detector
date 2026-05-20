import streamlit as st


pages = {
    "Musical Note Detector": [
        st.Page("pages/notes.py", title="Note Explorer"),
        st.Page("pages/detector.py", title="Note Detector"),
        st.Page("pages/bayesian.py", title="Bayesian Updating")
    ]
}

pg = st.navigation(pages)
pg.run()
