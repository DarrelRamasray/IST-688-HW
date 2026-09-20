#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#HW Manager

import streamlit as st

hw1_page = st.Page("HW/HW1.py", title="HW1", icon=":material/description:")
hw2_page = st.Page("HW/HW2.py", title="HW2", icon=":material/description:")
hw3_page = st.Page("HW/HW3.py", title="HW3", icon=":material/description:")
hw4_page = st.Page("HW/HW4.py", title="HW4", icon=":material/description:", default=True)  #Default page

pg = st.navigation([hw4_page,hw3_page, hw2_page, hw1_page])  #HW4 listed first so it appears at the top of the sidebar
st.set_page_config(page_title="HW Manager", page_icon=":material/edit:")
pg.run()